"""
Example of batch processing multiple images with HMM Tracer.
"""

import os
import sys
from hmm_tracer.cli import BatchProcessor

def batch_process_images(input_folder, output_folder):
    """
    Process all images in a folder using HMM Tracer.
    
    Args:
        input_folder: Path to the folder containing input images
        output_folder: Path to save the output images
    """
    # Check if the input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder not found at {input_folder}")
        return
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Processing images in: {input_folder}")
    print(f"Saving results to: {output_folder}")
    
    # Create a batch processor with default parameters
    processor = BatchProcessor(
        input_folder=input_folder,
        output_folder=output_folder,
        chain_level=1.1,
        node_r=4,
        line_length_min=32,
        min_sigma=1,
        max_sigma=64,
        threshold=0.015,
        debug=True
    )
    
    # Process all images
    results = processor.process_all()
    
    # Print results
    print(f"Processed {len(results)} images")
    print(f"Total axon length: {results['Distance'].sum()}")
    
    # Save results to CSV
    results_path = os.path.join(output_folder, "results.csv")
    results.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 3:
        print("Usage: python batch_processing.py <input_folder> <output_folder>")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    
    batch_process_images(input_folder, output_folder)
