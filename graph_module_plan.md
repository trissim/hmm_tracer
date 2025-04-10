# Graph Module Plan

## Current Implementation

The current implementation includes several functions in `hmm_axon.py` related to graph operations:

```python
def euclidian_distance(x1, y1, x2, y2):
  distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
  return distance

def extract_graph(root_tree_xx, root_tree_yy):
    graph = nx.Graph()
    for path_x, path_y in zip(root_tree_xx, root_tree_yy):
        for x, y in zip(path_x, path_y):
            graph.add_node((x, y))
        for i in range(len(path_x)-1):
            distance = euclidian_distance(path_x[i], path_y[i], path_x[i + 1], path_y[i + 1])
            graph.add_edge((path_x[i], path_y[i]), (path_x[i + 1], path_y[i + 1]), weight=distance)
    return graph

def graph_to_length(graph):
    total_distance = 0
    for u, v, data in graph.edges(data=True):
        total_distance += data['weight']
    return total_distance

def graph_to_image(G, image_path, original_image):
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
```

## Refactoring Goals

1. Create a dedicated module for graph operations
2. Improve code organization and readability
3. Add proper type hints and documentation
4. Create a class-based approach for graph operations

## Proposed Implementation

### Module Structure

```
hmm_tracer/
└── core/
    └── graph.py  # Graph operations module
```

### Implementation Details

```python
# hmm_tracer/core/graph.py
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import networkx as nx
import math
from skimage.io import imsave

class NeuriteGraph:
    """
    Class for handling graph operations on traced neurites.
    
    This class provides methods for creating, analyzing, and visualizing
    graph representations of traced neurites.
    """
    
    def __init__(self, graph: Optional[nx.Graph] = None):
        """
        Initialize the NeuriteGraph.
        
        Args:
            graph: Optional existing networkx Graph
        """
        self.graph = graph if graph is not None else nx.Graph()
    
    @staticmethod
    def euclidean_distance(
        x1: float, y1: float, x2: float, y2: float
    ) -> float:
        """
        Calculate the Euclidean distance between two points.
        
        Args:
            x1: X coordinate of first point
            y1: Y coordinate of first point
            x2: X coordinate of second point
            y2: Y coordinate of second point
            
        Returns:
            Euclidean distance
        """
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    
    @classmethod
    def from_paths(
        cls,
        root_tree_xx: List[List[int]],
        root_tree_yy: List[List[int]]
    ) -> 'NeuriteGraph':
        """
        Create a NeuriteGraph from traced paths.
        
        Args:
            root_tree_xx: List of lists containing x coordinates of paths
            root_tree_yy: List of lists containing y coordinates of paths
            
        Returns:
            NeuriteGraph instance
        """
        graph = nx.Graph()
        
        # Add nodes and edges for each path
        for path_x, path_y in zip(root_tree_xx, root_tree_yy):
            # Add nodes
            for x, y in zip(path_x, path_y):
                graph.add_node((x, y))
            
            # Add edges with distance weights
            for i in range(len(path_x) - 1):
                distance = cls.euclidean_distance(
                    path_x[i], path_y[i], path_x[i + 1], path_y[i + 1]
                )
                graph.add_edge(
                    (path_x[i], path_y[i]),
                    (path_x[i + 1], path_y[i + 1]),
                    weight=distance
                )
        
        return cls(graph)
    
    def get_total_length(self) -> float:
        """
        Calculate the total length of all neurites in the graph.
        
        Returns:
            Total length as sum of edge weights
        """
        total_distance = 0
        for u, v, data in self.graph.edges(data=True):
            total_distance += data['weight']
        return total_distance
    
    def save_as_image(
        self,
        image_path: str,
        original_image: np.ndarray,
        value: float = 1.0
    ) -> None:
        """
        Save the graph as an image.
        
        Args:
            image_path: Path to save the image
            original_image: Original image to get dimensions
            value: Value to use for graph pixels (default: 1.0)
        """
        # Get dimensions from original image
        max_y, max_x = original_image.shape
        
        # Create empty image
        image = np.zeros((max_y + 1, max_x + 1))
        
        # Draw edges
        for u, v in self.graph.edges:
            y1, x1 = u
            y2, x2 = v
            
            # Ensure coordinates are within bounds
            if 0 <= y1 < image.shape[0] and 0 <= x1 < image.shape[1]:
                image[y1, x1] = value
            
            if 0 <= y2 < image.shape[0] and 0 <= x2 < image.shape[1]:
                image[y2, x2] = value
        
        # Save image
        imsave(image_path, image, plugin='pil', format_str='png')
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Calculate various statistics about the graph.
        
        Returns:
            Dictionary of graph statistics
        """
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'total_length': self.get_total_length(),
            'connected_components': nx.number_connected_components(self.graph),
            'average_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes()
        }
    
    def get_endpoints(self) -> List[Tuple[int, int]]:
        """
        Get all endpoints (nodes with degree 1) in the graph.
        
        Returns:
            List of endpoint coordinates
        """
        return [node for node, degree in self.graph.degree() if degree == 1]
    
    def get_branch_points(self) -> List[Tuple[int, int]]:
        """
        Get all branch points (nodes with degree > 2) in the graph.
        
        Returns:
            List of branch point coordinates
        """
        return [node for node, degree in self.graph.degree() if degree > 2]
```

### Usage Example

```python
from hmm_tracer.core.preprocessing import ImagePreprocessor
from hmm_tracer.core.tracing import AxonTracer
from hmm_tracer.core.graph import NeuriteGraph
from skimage.io import imread

# Load and preprocess image
image = imread("input_image.tif")
normalized = ImagePreprocessor.normalize(image)
edge_map = ImagePreprocessor.apply_edge_detection(
    normalized,
    method="blob",
    min_sigma=1,
    max_sigma=64,
    threshold=0.015
)

# Trace axons
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

# Create graph
neurite_graph = NeuriteGraph.from_paths(root_tree_xx, root_tree_yy)

# Calculate total length
total_length = neurite_graph.get_total_length()
print(f"Total neurite length: {total_length}")

# Save as image
neurite_graph.save_as_image("output.png", normalized)

# Get statistics
stats = neurite_graph.get_graph_statistics()
print(f"Graph statistics: {stats}")
```

## Validation

The refactored code maintains the same functionality as the original implementation but with several improvements:

1. Better organization with a dedicated class
2. Proper type hints and documentation
3. Additional utility methods for graph analysis
4. Clear separation of concerns

The implementation doesn't duplicate any existing code in the codebase and follows Python best practices.

## Next Steps

After implementing the graph module, we'll proceed with:

1. [CLI Module Plan](cli_module_plan.md)
2. Integration of all modules
