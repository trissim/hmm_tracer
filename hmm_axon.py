from skimage.io import imread, imsave
import pudb
from skimage.feature import canny
from skimage.feature import blob_dog as local_max
from skimage.filters import *
from skimage.morphology import remove_small_objects, skeletonize
import numpy as np
import networkx as nx
import alva_machinery.markov.aChain as alva_MCMC
import alva_machinery.branching.aWay as alva_branch
import pandas as pd 
import skimage
import math
import os
import multiprocessing as mp
from functools import partial

def normalize(img,percentile=99.9):
    percentile_value = np.percentile(img, percentile)
    img = img / percentile_value  # Scale the image to the nth percentile value
    img = np.clip(img, 0, 100)  # You can change 1 to 100 if you want percentages
    #img = img - img.min()
    #img = img / img.max()
    return img

def boundary_masking_canny(image):
    bool_im_axon_edit = canny(image)
    bool_im_axon_edit[:,:2] = False
    bool_im_axon_edit[:,-2:] = False
    bool_im_axon_edit[:2,:] = False
    bool_im_axon_edit[-2:,:] = False
    return np.array(bool_im_axon_edit,dtype=np.int64)

def boundary_masking_threshold(image,threshold=threshold_li,min_size=2):
    threshed=threshold(image)
    bool_image = image > threshed
    bool_image[:,:2] = False
    bool_image[:,-2:] = False
    bool_image[:2,:] = False
    bool_image[-2:,:] = False
    cleaned_bool_im_axon_edit = skeletonize(bool_image)
    return np.array(bool_image,dtype=np.int64)

def boundary_masking_blob(image,min_sigma = 1, max_sigma = 2, threshold = 0.02):
    if min_sigma is None:
        min_sigma = 1
    if max_sigma is None:
        max_sigma = 2
    if threshold is None:
        threshold = 0.02

    image_median = median(image)
    galaxy = local_max(image_median, min_sigma = min_sigma, max_sigma = max_sigma, threshold = threshold)
    yy = np.int64(galaxy[:, 0])
    xx = np.int64(galaxy[:, 1])
    boundary_mask = np.copy(image) * 0
    boundary_mask[yy, xx] = 1
    return boundary_mask

def random_seed_by_edge_map(edge_map):
    yy, xx = edge_map.nonzero()
    seed_index = np.random.choice(len(xx),len(xx))
    seed_xx = xx[seed_index]
    seed_yy = yy[seed_index]
    return seed_xx, seed_yy

def get_growth_cone_positions(image):
    #image = image[:,:,np.newaxis]
    #image = skimage.img_as_ubyte(skimage.color.rgb2gray(image))
    
    # Apply a median filter to remove noise
    #image = skimage.filters.median(image)
    
    # Threshold the image to create a binary mask
    mask = image > skimage.filters.threshold_otsu(image)
    
    # Use morphological closing to fill in small gaps in the mask
    mask = skimage.morphology.closing(mask, skimage.morphology.disk(3))
    labeled = skimage.measure.label(mask)
    imsave("./mask.png",mask)
    props = skimage.measure.regionprops(labeled)
    positions = []
    seed_xx = []
    seed_yy = []
    for prop in props:
        seed_xx.append(prop.centroid[0])
        seed_yy.append(prop.centroid[1])
    print(str(len(seed_xx)) + " growth cones")
    return seed_xx,seed_yy

def selected_seeding(image,seed_xx,seed_yy,chain_level=1.05,total_node=8,node_r=None,line_length_min=32):
    im_copy=np.copy(image)
    alva_HMM = alva_MCMC.AlvaHmm(im_copy,
                                total_node = total_node,
                                total_path = None,
                                node_r = node_r,
                                node_angle_max = None,)
    chain_HMM_1st, pair_chain_HMM, pair_seed_xx, pair_seed_yy = alva_HMM.pair_HMM_chain(seed_xx = seed_xx,
                                                                                        seed_yy = seed_yy,
                                                                                        chain_level = chain_level,)
    for chain_i in [0, 1]:
                chain_HMM = [chain_HMM_1st, pair_chain_HMM][chain_i]
                real_chain_ii, real_chain_aa, real_chain_xx, real_chain_yy = chain_HMM[0:4]
                seed_node_xx, seed_node_yy = chain_HMM[4:6]

    chain_im_fine = alva_HMM.chain_image(chain_HMM_1st, pair_chain_HMM,)
    return alva_branch.connect_way(chain_im_fine,
                                    line_length_min = line_length_min,
                                    free_zone_from_y0 = None,)

def euclidian_distance(x1, y1, x2, y2):
  distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
  return distance

def extract_graph(root_tree_xx,root_tree_yy):
    graph = nx.Graph()
    for path_x,path_y in zip(root_tree_xx,root_tree_yy):
        for x,y in zip(path_x,path_y):
            graph.add_node((x,y))
        for i in range(len(path_x)-1):
            distance=euclidian_distance(path_x[i], path_y[i], path_x[i + 1], path_y[i + 1])
            graph.add_edge((path_x[i], path_y[i]), (path_x[i + 1], path_y[i + 1]),weight=distance)
    return graph

def graph_to_length(graph):
    total_distance = 0
    for u, v, data in graph.edges(data=True):
        total_distance += data['weight']
    return total_distance

def graph_to_image(G, image_path,original_image):
    # Get the maximum x and y values of the nodes in the graph
    max_y, max_x = original_image.shape
    # Initialize the 2D array with all zeros
    image = np.zeros((max_y+1, max_x+1))
    # Iterate over the edges in the graph and set the corresponding elements in the image array to 1
    for u, v in G.edges:
        y1, x1 = u
        y2, x2 = v
        image[y1, x1] = 1
        image[y2, x2] = 1
    # Save the image
    imsave(image_path, image, plugin='pil', format_str='png')

def process_file(filename, input_folder, output_folder, chain_level=1.05,node_r=None,total_node=None,line_length_min=32,min_sigma = 1, max_sigma = 2, threshold = 0.02, debug=True):
    print(f"Analyzing: {filename}")
    print(f"Processing {filename} with:")
    print(f"  - input_folder: {input_folder}")
    print(f"  - output_folder: {output_folder}")
    print(f"  - chain_level: {chain_level}")
    print(f"  - node_r: {node_r}")
    print(f"  - total_node: {total_node}")
    print(f"min_sigma: {min_sigma}, max_sigma: {max_sigma}, threshold: {threshold}")

    im_axon_path = os.path.join(input_folder, filename)
    print(im_axon_path)
    im_axon = skimage.io.imread(im_axon_path)
    im_axon = normalize(im_axon)
    im_axon_edit = np.copy(im_axon)

    edge_map = boundary_masking_blob(im_axon,min_sigma = min_sigma, max_sigma = max_sigma, threshold = threshold)

    if debug:
        debug_folder=input_folder.rstrip('/')+'_debug/'
        os.makedirs(debug_folder,exist_ok=True)
        edge_image_path = os.path.join(debug_folder, filename.replace(".tif", "_edge.png"))
        norm_image_path = os.path.join(debug_folder, filename.replace(".tif", "_norm.png"))
        imsave(edge_image_path, edge_map, plugin='pil', format_str='png')
    #    imsave(norm_image_path, im_axon, plugin='pil', format_str='png')

    seed_xx, seed_yy = random_seed_by_edge_map(edge_map)
    root_tree_yy, root_tree_xx, root_tip_yy, root_tip_xx = selected_seeding(im_axon, seed_xx, seed_yy, chain_level=chain_level,node_r=node_r,total_node=total_node,line_length_min=line_length_min)

    graph = extract_graph(root_tree_yy, root_tree_xx)
    distance = graph_to_length(graph)

    output_image_path = os.path.join(output_folder, filename.replace(".tif", ".png"))
    graph_to_image(graph, output_image_path, im_axon)

    return filename, distance  

def main():
    num_cores = mp.cpu_count() - 4
    filenames = []
    distances = []
    input_folder = '/home/ts/nvme_usb/IMX/processed_test-time-mfd-hips-4x-wide_Plate_12993/samples/03_axons_background'
    input_folder = "/home/ts/nvme_usb/IMX/processed_test-time-mfd-hips-4x-wide_Plate_12993/samples/03_axons"
    input_folder = "/home/ts/nvme_usb/IMX/processed_test-time-mfd-hips-4x-wide_Plate_12993/samples/axons_clear"
    input_folder = '/home/ts/nvme_usb/IMX/processed_test-time-mfd-hips-10x_Plate_12990/analysis/axons/10x_bin'
    filenames = os.listdir(input_folder)
    output_folder=input_folder.rstrip('/')+'_traced/'
    #decent
#    chain_level=1.05
#    total_node=64
#    node_r=30
#    line_length_min=16

    #hmm
    #chain_level=1.02
    #total_node=32
    #node_r=8
    #line_length_min=16
    #debug=True
    #hmm

    chain_level=1.1
    total_node=None
    node_r=4
    line_length_min=32
    #edge_map
    min_sigma=1
    max_sigma=64
    threshold=0.015
    debug=True

    task_args = [(filename, 
                  input_folder,
                  output_folder,
                  chain_level,
                  node_r,
                  total_node,
                  line_length_min,
                  min_sigma,
                  max_sigma,
                  threshold,
                  debug) for filename in filenames]

    os.makedirs(output_folder,exist_ok=True)

    with mp.Pool(processes=num_cores) as pool:
        results = pool.starmap(process_file, task_args)
    df = pd.DataFrame(results, columns=['filename', 'Distance'])
    df.to_csv('results.csv', index=False)

main()

