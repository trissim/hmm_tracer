"""
Simple example of using HMM Tracer to trace axons in a single image.
"""

import os
import sys
import numpy as np
from skimage.io import imread
from hmm_tracer.core.preprocessing import ImagePreprocessor
from hmm_tracer.core.tracing import AxonTracer
from hmm_tracer.core.graph import NeuriteGraph

def trace_single_image(image_path, output_path):
    """
    Trace axons in a single image and save the result.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the output image
    """
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    print(f"Processing image: {image_path}")
    
    # Load and preprocess image
    image = imread(image_path)
    normalized = ImagePreprocessor.normalize(image)
    
    # Generate edge map
    edge_map = ImagePreprocessor.apply_edge_detection(
        normalized,
        method="blob",
        min_sigma=1,
        max_sigma=64,
        threshold=0.015
    )
    
    # Save edge map for debugging
    edge_path = os.path.splitext(output_path)[0] + "_edge.png"
    ImagePreprocessor.save_image(edge_path, edge_map)
    print(f"Edge map saved to: {edge_path}")
    
    # Create tracer and trace axons
    tracer = AxonTracer(
        chain_level=1.1,
        node_r=4,
        line_length_min=32,
        debug=True
    )
    
    # Trace image
    root_tree_yy, root_tree_xx, root_tip_yy, root_tip_xx = tracer.trace_image(
        normalized,
        edge_map,
        seed_method="random"
    )
    
    # Create graph
    neurite_graph = NeuriteGraph.from_paths(root_tree_yy, root_tree_xx)
    distance = neurite_graph.get_total_length()
    
    # Get graph statistics
    stats = neurite_graph.get_graph_statistics()
    
    # Save output image
    neurite_graph.save_as_image(output_path, normalized)
    
    print(f"Tracing completed successfully!")
    print(f"Output saved to: {output_path}")
    print(f"Total axon length: {distance}")
    print(f"Graph statistics:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 3:
        print("Usage: python simple_tracing.py <input_image> <output_image>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    trace_single_image(input_path, output_path)
