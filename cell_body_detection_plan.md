# Cell Body Detection Feature Plan

## Overview

This plan outlines the approach for adding cell body detection functionality to the hmm_tracer package. This feature will allow the detection of neuronal cell bodies in microscopy images, which can then be connected to the traced axons.

## Current Status

1. The hmm_tracer package has been refactored with a modular structure
2. Axon tracing functionality is working using the alvahmm package
3. No cell body detection functionality exists yet

## Feature Goals

1. Implement robust cell body detection in microscopy images
2. Connect detected cell bodies to traced axons
3. Enable analysis of neurite outgrowth from specific cell bodies
4. Provide visualization of cell bodies and their connected neurites

## Implementation Approach

### 1. Create Cell Body Detection Module

Create a new module in the hmm_tracer package:

```
hmm_tracer/
└── cellbody/
    ├── __init__.py
    ├── detection.py     # Cell body detection algorithms
    └── connection.py    # Connecting cell bodies to neurites
```

### 2. Implement Cell Body Detection Algorithms

The `detection.py` module will implement various algorithms for cell body detection:

```python
# hmm_tracer/cellbody/detection.py
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from skimage.measure import regionprops, label
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import closing, disk, remove_small_objects
from skimage.feature import blob_dog

class CellBodyDetector:
    """
    Class for detecting cell bodies in microscopy images.
    """
    
    def __init__(
        self,
        method: str = "threshold",
        min_size: int = 100,
        max_size: Optional[int] = None,
        threshold_method: str = "otsu",
        local_block_size: int = 35,
        min_sigma: float = 5.0,
        max_sigma: float = 20.0,
        sigma_ratio: float = 1.6,
        threshold: float = 0.05,
        overlap: float = 0.5
    ):
        """
        Initialize the cell body detector.
        
        Args:
            method: Detection method ('threshold', 'local_threshold', or 'blob')
            min_size: Minimum size of cell bodies to detect
            max_size: Maximum size of cell bodies to detect
            threshold_method: Method for global thresholding
            local_block_size: Block size for local thresholding
            min_sigma: Minimum sigma for blob detection
            max_sigma: Maximum sigma for blob detection
            sigma_ratio: Sigma ratio for blob detection
            threshold: Threshold for blob detection
            overlap: Overlap threshold for blob detection
        """
        self.method = method
        self.min_size = min_size
        self.max_size = max_size
        self.threshold_method = threshold_method
        self.local_block_size = local_block_size
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.sigma_ratio = sigma_ratio
        self.threshold = threshold
        self.overlap = overlap
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect cell bodies in an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of dictionaries containing cell body properties
        """
        if self.method == "threshold":
            return self._detect_threshold(image)
        elif self.method == "local_threshold":
            return self._detect_local_threshold(image)
        elif self.method == "blob":
            return self._detect_blob(image)
        else:
            raise ValueError(f"Unknown detection method: {self.method}")
    
    def _detect_threshold(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect cell bodies using global thresholding.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of dictionaries containing cell body properties
        """
        # Apply threshold
        if self.threshold_method == "otsu":
            thresh = threshold_otsu(image)
        else:
            # Implement other threshold methods
            thresh = threshold_otsu(image)
        
        # Create binary mask
        mask = image > thresh
        
        # Apply morphological operations
        mask = closing(mask, disk(3))
        
        # Remove small objects
        mask = remove_small_objects(mask, min_size=self.min_size)
        
        # Label connected components
        labeled = label(mask)
        
        # Extract region properties
        props = regionprops(labeled)
        
        # Filter regions by size
        cell_bodies = []
        for prop in props:
            if self.max_size is None or prop.area <= self.max_size:
                cell_bodies.append({
                    "centroid": prop.centroid,
                    "area": prop.area,
                    "bbox": prop.bbox,
                    "label": prop.label,
                    "perimeter": prop.perimeter,
                    "eccentricity": prop.eccentricity,
                    "intensity_mean": np.mean(image[prop.coords[:, 0], prop.coords[:, 1]])
                })
        
        return cell_bodies
    
    def _detect_local_threshold(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect cell bodies using local thresholding.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of dictionaries containing cell body properties
        """
        # Apply local threshold
        local_thresh = threshold_local(image, block_size=self.local_block_size)
        mask = image > local_thresh
        
        # Apply morphological operations
        mask = closing(mask, disk(3))
        
        # Remove small objects
        mask = remove_small_objects(mask, min_size=self.min_size)
        
        # Label connected components
        labeled = label(mask)
        
        # Extract region properties
        props = regionprops(labeled)
        
        # Filter regions by size
        cell_bodies = []
        for prop in props:
            if self.max_size is None or prop.area <= self.max_size:
                cell_bodies.append({
                    "centroid": prop.centroid,
                    "area": prop.area,
                    "bbox": prop.bbox,
                    "label": prop.label,
                    "perimeter": prop.perimeter,
                    "eccentricity": prop.eccentricity,
                    "intensity_mean": np.mean(image[prop.coords[:, 0], prop.coords[:, 1]])
                })
        
        return cell_bodies
    
    def _detect_blob(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect cell bodies using blob detection.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of dictionaries containing cell body properties
        """
        # Detect blobs
        blobs = blob_dog(
            image,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            sigma_ratio=self.sigma_ratio,
            threshold=self.threshold,
            overlap=self.overlap
        )
        
        # Convert blobs to cell bodies
        cell_bodies = []
        for i, blob in enumerate(blobs):
            y, x, r = blob
            r = r * np.sqrt(2)  # Convert sigma to radius
            
            # Calculate approximate area
            area = np.pi * (r ** 2)
            
            # Skip if too small or too large
            if area < self.min_size or (self.max_size is not None and area > self.max_size):
                continue
            
            # Calculate bounding box
            y_min = max(0, int(y - r))
            y_max = min(image.shape[0], int(y + r))
            x_min = max(0, int(x - r))
            x_max = min(image.shape[1], int(x + r))
            
            # Calculate mean intensity
            mask = np.zeros(image.shape, dtype=bool)
            y_coords, x_coords = np.ogrid[:image.shape[0], :image.shape[1]]
            dist = np.sqrt((y_coords - y) ** 2 + (x_coords - x) ** 2)
            mask[dist <= r] = True
            intensity_mean = np.mean(image[mask])
            
            cell_bodies.append({
                "centroid": (y, x),
                "area": area,
                "bbox": (y_min, x_min, y_max, x_max),
                "label": i + 1,
                "radius": r,
                "intensity_mean": intensity_mean
            })
        
        return cell_bodies
```

### 3. Implement Cell Body to Neurite Connection

The `connection.py` module will implement algorithms for connecting cell bodies to traced neurites:

```python
# hmm_tracer/cellbody/connection.py
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import networkx as nx
from hmm_tracer.core.graph import NeuriteGraph

class NeuriteConnector:
    """
    Class for connecting cell bodies to traced neurites.
    """
    
    def __init__(
        self,
        max_distance: float = 50.0,
        connection_method: str = "nearest"
    ):
        """
        Initialize the neurite connector.
        
        Args:
            max_distance: Maximum distance for connecting cell bodies to neurites
            connection_method: Method for connecting ('nearest', 'all_within_distance')
        """
        self.max_distance = max_distance
        self.connection_method = connection_method
    
    def connect(
        self,
        cell_bodies: List[Dict[str, Any]],
        neurite_graph: NeuriteGraph
    ) -> nx.DiGraph:
        """
        Connect cell bodies to neurites.
        
        Args:
            cell_bodies: List of cell body dictionaries
            neurite_graph: NeuriteGraph instance
            
        Returns:
            Directed graph with cell bodies and neurites
        """
        if self.connection_method == "nearest":
            return self._connect_nearest(cell_bodies, neurite_graph)
        elif self.connection_method == "all_within_distance":
            return self._connect_all_within_distance(cell_bodies, neurite_graph)
        else:
            raise ValueError(f"Unknown connection method: {self.connection_method}")
    
    def _connect_nearest(
        self,
        cell_bodies: List[Dict[str, Any]],
        neurite_graph: NeuriteGraph
    ) -> nx.DiGraph:
        """
        Connect each cell body to its nearest neurite point.
        
        Args:
            cell_bodies: List of cell body dictionaries
            neurite_graph: NeuriteGraph instance
            
        Returns:
            Directed graph with cell bodies and neurites
        """
        # Create a directed graph
        digraph = nx.DiGraph()
        
        # Add all nodes and edges from the neurite graph
        for node in neurite_graph.graph.nodes():
            digraph.add_node(node, type="neurite")
        
        for u, v, data in neurite_graph.graph.edges(data=True):
            digraph.add_edge(u, v, **data)
            digraph.add_edge(v, u, **data)  # Add reverse edge
        
        # Add cell bodies as nodes
        for i, cell_body in enumerate(cell_bodies):
            cell_node = (f"cell_{i}", "body")
            digraph.add_node(cell_node, **cell_body, type="cell_body")
            
            # Find the nearest neurite point
            min_dist = float('inf')
            nearest_node = None
            
            for node in neurite_graph.graph.nodes():
                y, x = node
                cell_y, cell_x = cell_body["centroid"]
                dist = np.sqrt((y - cell_y) ** 2 + (x - cell_x) ** 2)
                
                if dist < min_dist and dist <= self.max_distance:
                    min_dist = dist
                    nearest_node = node
            
            # Connect the cell body to the nearest neurite point
            if nearest_node is not None:
                digraph.add_edge(cell_node, nearest_node, weight=min_dist, type="connection")
        
        return digraph
    
    def _connect_all_within_distance(
        self,
        cell_bodies: List[Dict[str, Any]],
        neurite_graph: NeuriteGraph
    ) -> nx.DiGraph:
        """
        Connect each cell body to all neurite points within max_distance.
        
        Args:
            cell_bodies: List of cell body dictionaries
            neurite_graph: NeuriteGraph instance
            
        Returns:
            Directed graph with cell bodies and neurites
        """
        # Create a directed graph
        digraph = nx.DiGraph()
        
        # Add all nodes and edges from the neurite graph
        for node in neurite_graph.graph.nodes():
            digraph.add_node(node, type="neurite")
        
        for u, v, data in neurite_graph.graph.edges(data=True):
            digraph.add_edge(u, v, **data)
            digraph.add_edge(v, u, **data)  # Add reverse edge
        
        # Add cell bodies as nodes
        for i, cell_body in enumerate(cell_bodies):
            cell_node = (f"cell_{i}", "body")
            digraph.add_node(cell_node, **cell_body, type="cell_body")
            
            # Connect to all neurite points within max_distance
            for node in neurite_graph.graph.nodes():
                y, x = node
                cell_y, cell_x = cell_body["centroid"]
                dist = np.sqrt((y - cell_y) ** 2 + (x - cell_x) ** 2)
                
                if dist <= self.max_distance:
                    digraph.add_edge(cell_node, node, weight=dist, type="connection")
        
        return digraph
    
    def analyze_connections(
        self,
        connected_graph: nx.DiGraph
    ) -> Dict[str, Any]:
        """
        Analyze the connections between cell bodies and neurites.
        
        Args:
            connected_graph: Directed graph with cell bodies and neurites
            
        Returns:
            Dictionary of analysis results
        """
        # Find all cell body nodes
        cell_nodes = [node for node, attr in connected_graph.nodes(data=True) 
                     if attr.get("type") == "cell_body"]
        
        # Calculate statistics for each cell body
        cell_stats = {}
        for cell_node in cell_nodes:
            # Get all neurites connected to this cell body
            connected_neurites = list(connected_graph.successors(cell_node))
            
            if not connected_neurites:
                cell_stats[cell_node] = {
                    "connected": False,
                    "num_neurites": 0,
                    "total_length": 0,
                    "max_distance": 0
                }
                continue
            
            # Calculate total neurite length
            total_length = 0
            visited = set()
            
            for neurite in connected_neurites:
                # Perform DFS from each connected neurite
                stack = [(neurite, 0)]
                while stack:
                    current, length = stack.pop()
                    if current in visited:
                        continue
                    visited.add(current)
                    
                    # Add outgoing edges
                    for neighbor in connected_graph.successors(current):
                        if neighbor != cell_node and neighbor not in visited:
                            edge_weight = connected_graph[current][neighbor]["weight"]
                            stack.append((neighbor, length + edge_weight))
                
                # Update total length
                total_length += length
            
            # Calculate maximum distance
            max_distance = max(connected_graph[cell_node][neurite]["weight"] 
                              for neurite in connected_neurites)
            
            cell_stats[cell_node] = {
                "connected": True,
                "num_neurites": len(connected_neurites),
                "total_length": total_length,
                "max_distance": max_distance
            }
        
        # Calculate overall statistics
        overall_stats = {
            "num_cells": len(cell_nodes),
            "num_connected_cells": sum(1 for stats in cell_stats.values() if stats["connected"]),
            "total_neurite_length": sum(stats["total_length"] for stats in cell_stats.values()),
            "avg_neurites_per_cell": np.mean([stats["num_neurites"] for stats in cell_stats.values() if stats["connected"]])
        }
        
        return {
            "cell_stats": cell_stats,
            "overall_stats": overall_stats
        }
```

### 4. Update the CLI Module

Update the CLI module to include cell body detection:

```python
# Add to hmm_tracer/cli.py

def main():
    """
    Main function for the command-line interface.
    """
    parser = argparse.ArgumentParser(description='Trace axons in microscopy images.')
    
    # Existing arguments...
    
    # Add cell body detection arguments
    parser.add_argument('--detect-cell-bodies', action='store_true',
                        help='Detect cell bodies in the images')
    parser.add_argument('--cell-method', choices=['threshold', 'local_threshold', 'blob'],
                        default='threshold', help='Cell body detection method')
    parser.add_argument('--cell-min-size', type=int, default=100,
                        help='Minimum size of cell bodies to detect')
    parser.add_argument('--cell-max-size', type=int, default=None,
                        help='Maximum size of cell bodies to detect')
    parser.add_argument('--connect-method', choices=['nearest', 'all_within_distance'],
                        default='nearest', help='Method for connecting cell bodies to neurites')
    parser.add_argument('--max-distance', type=float, default=50.0,
                        help='Maximum distance for connecting cell bodies to neurites')
    
    # Process arguments and run...
```

### 5. Create Visualization Functions

Add visualization functions for cell bodies and connections:

```python
# Add to hmm_tracer/utils/visualization.py

def visualize_cell_bodies(
    image: np.ndarray,
    cell_bodies: List[Dict[str, Any]],
    output_path: str,
    color: Tuple[int, int, int] = (255, 0, 0)
) -> None:
    """
    Visualize detected cell bodies on an image.
    
    Args:
        image: Input image
        cell_bodies: List of cell body dictionaries
        output_path: Path to save the visualization
        color: RGB color for cell body outlines
    """
    # Implementation...

def visualize_connections(
    image: np.ndarray,
    connected_graph: nx.DiGraph,
    output_path: str,
    cell_color: Tuple[int, int, int] = (255, 0, 0),
    neurite_color: Tuple[int, int, int] = (0, 255, 0),
    connection_color: Tuple[int, int, int] = (0, 0, 255)
) -> None:
    """
    Visualize cell bodies and their connections to neurites.
    
    Args:
        image: Input image
        connected_graph: Directed graph with cell bodies and neurites
        output_path: Path to save the visualization
        cell_color: RGB color for cell bodies
        neurite_color: RGB color for neurites
        connection_color: RGB color for connections
    """
    # Implementation...
```

## Testing Approach

1. Create synthetic test images with known cell bodies and neurites
2. Test each detection method on the synthetic images
3. Test the connection algorithms with different parameters
4. Validate on real microscopy images with manual annotations

## Validation

To validate the cell body detection:

1. Compare detected cell bodies with manual annotations
2. Calculate precision, recall, and F1 score
3. Evaluate the quality of connections between cell bodies and neurites
4. Verify that the analysis results match expectations

## Next Steps

After implementing the cell body detection feature:

1. Optimize the algorithms for performance
2. Add more detection methods if needed
3. Improve the visualization capabilities
4. Integrate with the existing axon tracing workflow
