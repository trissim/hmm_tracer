"""
Example of cell body detection and connection to traced neurites.
"""

import os
import sys
import numpy as np
from skimage.io import imread
from hmm_tracer.core.preprocessing import ImagePreprocessor
from hmm_tracer.core.tracing import AxonTracer
from hmm_tracer.core.graph import NeuriteGraph
from hmm_tracer.cellbody.detection import CellBodyDetector
from hmm_tracer.cellbody.connection import NeuriteConnector

def detect_and_connect(image_path, output_dir):
    """
    Detect cell bodies and connect them to traced neurites.
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save output images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess image
    print(f"Loading image: {image_path}")
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
    
    # Save edge map
    edge_path = os.path.join(output_dir, "edge_map.png")
    ImagePreprocessor.save_image(edge_path, edge_map)
    print(f"Edge map saved to: {edge_path}")
    
    # Trace neurites
    print("Tracing neurites...")
    tracer = AxonTracer(
        chain_level=1.1,
        node_r=4,
        line_length_min=32
    )
    
    root_tree_yy, root_tree_xx, root_tip_yy, root_tip_xx = tracer.trace_image(
        normalized,
        edge_map,
        seed_method="random"
    )
    
    # Create neurite graph
    neurite_graph = NeuriteGraph.from_paths(root_tree_yy, root_tree_xx)
    
    # Save neurite graph
    neurite_path = os.path.join(output_dir, "neurites.png")
    neurite_graph.save_as_image(neurite_path, normalized)
    print(f"Neurite graph saved to: {neurite_path}")
    
    # Detect cell bodies
    print("Detecting cell bodies...")
    detector = CellBodyDetector(
        method="threshold",
        min_size=100,
        max_size=1000
    )
    
    cell_bodies = detector.detect(normalized)
    print(f"Detected {len(cell_bodies)} cell bodies")
    
    # Visualize cell bodies
    cell_path = os.path.join(output_dir, "cell_bodies.png")
    detector.visualize(normalized, cell_bodies, cell_path)
    print(f"Cell body visualization saved to: {cell_path}")
    
    # Connect cell bodies to neurites
    print("Connecting cell bodies to neurites...")
    connector = NeuriteConnector(
        max_distance=50.0,
        connection_method="nearest"
    )
    
    connected_graph = connector.connect(cell_bodies, neurite_graph)
    
    # Analyze connections
    analysis = connector.analyze_connections(connected_graph)
    print("Connection analysis:")
    print(f"  - Number of cells: {analysis['overall_stats']['num_cells']}")
    print(f"  - Number of connected cells: {analysis['overall_stats']['num_connected_cells']}")
    print(f"  - Total neurite length: {analysis['overall_stats']['total_neurite_length']:.2f}")
    
    # Visualize connections
    connection_path = os.path.join(output_dir, "connections.png")
    connector.visualize(normalized, connected_graph, connection_path)
    print(f"Connection visualization saved to: {connection_path}")

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 3:
        print("Usage: python cell_body_detection.py <input_image> <output_directory>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    detect_and_connect(image_path, output_dir)
