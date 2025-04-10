"""
Cell body to neurite connection module for HMM Tracer.

This module provides algorithms for connecting detected cell bodies to traced neurites.
"""

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
    
    def visualize(
        self,
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
        from skimage.color import gray2rgb
        from skimage.draw import circle, line
        from skimage.io import imsave
        import numpy as np
        
        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            vis_image = gray2rgb(image)
        else:
            vis_image = image.copy()
        
        # Normalize image if needed
        if vis_image.max() <= 1.0:
            vis_image = (vis_image * 255).astype(np.uint8)
        
        # Draw neurites
        for u, v, data in connected_graph.edges(data=True):
            if data.get("type") != "connection":
                # Draw neurite edge
                if isinstance(u, tuple) and u[1] == "body":
                    continue
                if isinstance(v, tuple) and v[1] == "body":
                    continue
                
                y1, x1 = u
                y2, x2 = v
                rr, cc = line(int(y1), int(x1), int(y2), int(x2))
                vis_image[rr, cc] = neurite_color
        
        # Draw cell bodies and connections
        for node, attr in connected_graph.nodes(data=True):
            if attr.get("type") == "cell_body":
                # Draw cell body
                y, x = attr["centroid"]
                radius = int(np.sqrt(attr["area"] / np.pi))
                rr, cc = circle(int(y), int(x), radius, shape=vis_image.shape)
                vis_image[rr, cc] = cell_color
                
                # Draw connections
                for neighbor in connected_graph.successors(node):
                    ny, nx = neighbor
                    rr, cc = line(int(y), int(x), int(ny), int(nx))
                    vis_image[rr, cc] = connection_color
        
        # Save visualization
        imsave(output_path, vis_image)
