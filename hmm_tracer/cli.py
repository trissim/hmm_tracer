"""
Command-line interface for HMM Tracer.

This module provides a command-line interface for batch processing
of microscopy images for axon tracing.
"""

import os
import argparse
import multiprocessing as mp
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from skimage.io import imread
import numpy as np

from hmm_tracer.core.preprocessing import ImagePreprocessor
from hmm_tracer.core.tracing import AxonTracer
from hmm_tracer.core.graph import NeuriteGraph

class BatchProcessor:
    """
    Class for batch processing of multiple images.
    
    This class handles the processing of multiple images in parallel,
    using the refactored preprocessing, tracing, and graph modules.
    """
    
    def __init__(
        self,
        input_folder: str,
        output_folder: str,
        chain_level: float = 1.1,
        total_node: Optional[int] = None,
        node_r: Optional[int] = 4,
        line_length_min: int = 32,
        min_sigma: float = 1,
        max_sigma: float = 64,
        threshold: float = 0.015,
        debug: bool = True,
        num_cores: Optional[int] = None
    ):
        """
        Initialize the BatchProcessor.
        
        Args:
            input_folder: Folder containing input images
            output_folder: Folder to save output images
            chain_level: Parameter controlling the HMM chain level
            total_node: Number of nodes in the HMM chain
            node_r: Node radius parameter
            line_length_min: Minimum line length to consider
            min_sigma: Minimum sigma for blob detection
            max_sigma: Maximum sigma for blob detection
            threshold: Threshold for blob detection
            debug: Whether to save debug images
            num_cores: Number of CPU cores to use (default: CPU count - 4)
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.chain_level = chain_level
        self.total_node = total_node
        self.node_r = node_r
        self.line_length_min = line_length_min
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.threshold = threshold
        self.debug = debug
        self.num_cores = num_cores if num_cores is not None else max(1, mp.cpu_count() - 4)
        
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Create debug folder if needed
        if debug:
            self.debug_folder = input_folder.rstrip('/') + '_debug/'
            os.makedirs(self.debug_folder, exist_ok=True)
    
    def process_file(
        self,
        filename: str
    ) -> Tuple[str, float]:
        """
        Process a single image file.
        
        Args:
            filename: Name of the image file
            
        Returns:
            Tuple of (filename, distance)
        """
        print(f"Analyzing: {filename}")
        print(f"Processing {filename} with:")
        print(f"  - input_folder: {self.input_folder}")
        print(f"  - output_folder: {self.output_folder}")
        print(f"  - chain_level: {self.chain_level}")
        print(f"  - node_r: {self.node_r}")
        print(f"  - total_node: {self.total_node}")
        print(f"  - min_sigma: {self.min_sigma}, max_sigma: {self.max_sigma}, threshold: {self.threshold}")
        
        # Load and preprocess image
        im_axon_path = os.path.join(self.input_folder, filename)
        im_axon = imread(im_axon_path)
        im_axon = ImagePreprocessor.normalize(im_axon)
        
        # Generate edge map
        edge_map = ImagePreprocessor.apply_edge_detection(
            im_axon,
            method="blob",
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            threshold=self.threshold
        )
        
        # Save debug images if requested
        if self.debug:
            edge_image_path = os.path.join(
                self.debug_folder,
                filename.replace(".tif", "_edge.png")
            )
            ImagePreprocessor.save_image(edge_image_path, edge_map)
        
        # Create tracer and trace axons
        tracer = AxonTracer(
            chain_level=self.chain_level,
            total_node=self.total_node,
            node_r=self.node_r,
            line_length_min=self.line_length_min,
            debug=self.debug
        )
        
        # Trace image
        root_tree_yy, root_tree_xx, root_tip_yy, root_tip_xx = tracer.trace_image(
            im_axon,
            edge_map,
            seed_method="random"
        )
        
        # Create graph
        neurite_graph = NeuriteGraph.from_paths(root_tree_yy, root_tree_xx)
        distance = neurite_graph.get_total_length()
        
        # Save output image
        output_image_path = os.path.join(
            self.output_folder,
            filename.replace(".tif", ".png")
        )
        neurite_graph.save_as_image(output_image_path, im_axon)
        
        return filename, distance
    
    def process_all(self) -> pd.DataFrame:
        """
        Process all image files in the input folder.
        
        Returns:
            DataFrame with results
        """
        # Get list of files
        filenames = os.listdir(self.input_folder)
        
        # Filter for image files
        image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
        filenames = [f for f in filenames if any(f.lower().endswith(ext) for ext in image_extensions)]
        
        # Process files in parallel
        with mp.Pool(processes=self.num_cores) as pool:
            results = pool.map(self.process_file, filenames)
        
        # Create DataFrame and save results
        df = pd.DataFrame(results, columns=['filename', 'Distance'])
        df.to_csv('results.csv', index=False)
        
        return df


def main():
    """
    Main function for the command-line interface.
    """
    parser = argparse.ArgumentParser(description='Trace axons in microscopy images.')
    
    parser.add_argument('--input', '-i', required=True,
                        help='Input folder containing images')
    parser.add_argument('--output', '-o', required=True,
                        help='Output folder for traced images')
    parser.add_argument('--chain-level', type=float, default=1.1,
                        help='HMM chain level parameter')
    parser.add_argument('--total-node', type=int, default=None,
                        help='Number of nodes in HMM chain')
    parser.add_argument('--node-r', type=int, default=4,
                        help='Node radius parameter')
    parser.add_argument('--line-length-min', type=int, default=32,
                        help='Minimum line length to consider')
    parser.add_argument('--min-sigma', type=float, default=1,
                        help='Minimum sigma for blob detection')
    parser.add_argument('--max-sigma', type=float, default=64,
                        help='Maximum sigma for blob detection')
    parser.add_argument('--threshold', type=float, default=0.015,
                        help='Threshold for blob detection')
    parser.add_argument('--debug', action='store_true',
                        help='Save debug images')
    parser.add_argument('--cores', type=int, default=None,
                        help='Number of CPU cores to use')
    
    args = parser.parse_args()
    
    # Create batch processor
    processor = BatchProcessor(
        input_folder=args.input,
        output_folder=args.output,
        chain_level=args.chain_level,
        total_node=args.total_node,
        node_r=args.node_r,
        line_length_min=args.line_length_min,
        min_sigma=args.min_sigma,
        max_sigma=args.max_sigma,
        threshold=args.threshold,
        debug=args.debug,
        num_cores=args.cores
    )
    
    # Process all files
    results = processor.process_all()
    
    print(f"Processed {len(results)} files")
    print(f"Total axon length: {results['Distance'].sum()}")


if __name__ == '__main__':
    main()
