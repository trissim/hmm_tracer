"""
Tests for the graph operations module.
"""

import unittest
import numpy as np
import networkx as nx
from hmm_tracer.core.graph import NeuriteGraph

class TestNeuriteGraph(unittest.TestCase):
    """Test cases for the NeuriteGraph class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test graph
        self.graph = nx.Graph()
        self.graph.add_node((0, 0))
        self.graph.add_node((0, 1))
        self.graph.add_node((1, 1))
        self.graph.add_edge((0, 0), (0, 1), weight=1.0)
        self.graph.add_edge((0, 1), (1, 1), weight=1.0)
        
        # Create a NeuriteGraph from the test graph
        self.neurite_graph = NeuriteGraph(self.graph)
        
        # Create test paths for from_paths method
        self.root_tree_xx = [[0, 0, 1], [1, 2, 3]]
        self.root_tree_yy = [[0, 1, 1], [1, 1, 2]]
    
    def test_euclidean_distance(self):
        """Test the euclidean_distance method."""
        # Test with simple cases
        self.assertEqual(NeuriteGraph.euclidean_distance(0, 0, 3, 4), 5.0)
        self.assertEqual(NeuriteGraph.euclidean_distance(1, 1, 4, 5), 5.0)
        
        # Test with floating point values
        self.assertAlmostEqual(
            NeuriteGraph.euclidean_distance(0.5, 0.5, 1.5, 1.5),
            1.4142135623730951
        )
    
    def test_from_paths(self):
        """Test the from_paths method."""
        # Create a NeuriteGraph from paths
        neurite_graph = NeuriteGraph.from_paths(
            self.root_tree_xx,
            self.root_tree_yy
        )
        
        # Check that the graph has the expected number of nodes and edges
        self.assertEqual(neurite_graph.graph.number_of_nodes(), 6)
        self.assertEqual(neurite_graph.graph.number_of_edges(), 4)
        
        # Check that the nodes are correct
        expected_nodes = [(0, 0), (0, 1), (1, 1), (1, 1), (2, 1), (3, 2)]
        self.assertEqual(set(neurite_graph.graph.nodes()), set(expected_nodes))
    
    def test_get_total_length(self):
        """Test the get_total_length method."""
        # Check the total length of the test graph
        self.assertEqual(self.neurite_graph.get_total_length(), 2.0)
        
        # Create a graph with different weights
        graph = nx.Graph()
        graph.add_node((0, 0))
        graph.add_node((0, 1))
        graph.add_node((1, 1))
        graph.add_edge((0, 0), (0, 1), weight=2.0)
        graph.add_edge((0, 1), (1, 1), weight=3.0)
        
        neurite_graph = NeuriteGraph(graph)
        self.assertEqual(neurite_graph.get_total_length(), 5.0)
    
    def test_get_graph_statistics(self):
        """Test the get_graph_statistics method."""
        # Get statistics for the test graph
        stats = self.neurite_graph.get_graph_statistics()
        
        # Check that the statistics are correct
        self.assertEqual(stats['num_nodes'], 3)
        self.assertEqual(stats['num_edges'], 2)
        self.assertEqual(stats['total_length'], 2.0)
        self.assertEqual(stats['connected_components'], 1)
        self.assertAlmostEqual(stats['average_degree'], 4/3)
    
    def test_get_endpoints(self):
        """Test the get_endpoints method."""
        # Check the endpoints of the test graph
        endpoints = self.neurite_graph.get_endpoints()
        
        # The test graph has two endpoints: (0, 0) and (1, 1)
        self.assertEqual(set(endpoints), {(0, 0), (1, 1)})
    
    def test_get_branch_points(self):
        """Test the get_branch_points method."""
        # The test graph has no branch points
        branch_points = self.neurite_graph.get_branch_points()
        self.assertEqual(branch_points, [])
        
        # Create a graph with a branch point
        graph = nx.Graph()
        graph.add_node((0, 0))
        graph.add_node((0, 1))
        graph.add_node((1, 1))
        graph.add_node((0, 2))
        graph.add_edge((0, 0), (0, 1), weight=1.0)
        graph.add_edge((0, 1), (1, 1), weight=1.0)
        graph.add_edge((0, 1), (0, 2), weight=1.0)
        
        neurite_graph = NeuriteGraph(graph)
        branch_points = neurite_graph.get_branch_points()
        
        # The graph has one branch point: (0, 1)
        self.assertEqual(branch_points, [(0, 1)])

if __name__ == "__main__":
    unittest.main()
