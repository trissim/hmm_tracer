"""
Tests for the cell body detection module.
"""

import unittest
import numpy as np
from hmm_tracer.cellbody.detection import CellBodyDetector
from hmm_tracer.cellbody.connection import NeuriteConnector
from hmm_tracer.core.graph import NeuriteGraph

class TestCellBodyDetector(unittest.TestCase):
    """Test cases for the CellBodyDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test image with cell bodies
        self.test_image = np.zeros((200, 200), dtype=np.float32)
        
        # Add some cell-like features
        # Cell 1: Large bright spot
        y1, x1 = 50, 50
        r1 = 20
        y_grid, x_grid = np.ogrid[-r1:r1+1, -r1:r1+1]
        mask = y_grid**2 + x_grid**2 <= r1**2
        self.test_image[y1-r1:y1+r1+1, x1-r1:x1+r1+1][mask] = 1.0
        
        # Cell 2: Medium bright spot
        y2, x2 = 150, 150
        r2 = 15
        y_grid, x_grid = np.ogrid[-r2:r2+1, -r2:r2+1]
        mask = y_grid**2 + x_grid**2 <= r2**2
        self.test_image[y2-r2:y2+r2+1, x2-r2:x2+r2+1][mask] = 0.8
        
        # Add some noise
        np.random.seed(42)
        self.test_image += np.random.normal(0, 0.1, size=self.test_image.shape)
        
        # Ensure values are positive
        self.test_image = np.clip(self.test_image, 0, 1.0)
        
        # Create a detector
        self.detector = CellBodyDetector(
            method="threshold",
            min_size=100,
            max_size=2000
        )
    
    def test_detect_threshold(self):
        """Test the threshold detection method."""
        # Detect cell bodies
        cell_bodies = self.detector.detect(self.test_image)
        
        # Check that we detected at least one cell body
        self.assertGreater(len(cell_bodies), 0)
        
        # Check that the cell bodies have the expected properties
        for cell in cell_bodies:
            self.assertIn("centroid", cell)
            self.assertIn("area", cell)
            self.assertIn("bbox", cell)
            self.assertIn("label", cell)
    
    def test_detect_blob(self):
        """Test the blob detection method."""
        # Create a blob detector
        blob_detector = CellBodyDetector(
            method="blob",
            min_sigma=5.0,
            max_sigma=20.0,
            threshold=0.05
        )
        
        # Detect cell bodies
        cell_bodies = blob_detector.detect(self.test_image)
        
        # Check that we detected at least one cell body
        self.assertGreater(len(cell_bodies), 0)
        
        # Check that the cell bodies have the expected properties
        for cell in cell_bodies:
            self.assertIn("centroid", cell)
            self.assertIn("area", cell)
            self.assertIn("bbox", cell)
            self.assertIn("label", cell)
            self.assertIn("radius", cell)

class TestNeuriteConnector(unittest.TestCase):
    """Test cases for the NeuriteConnector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test graph
        self.graph = nx.Graph()
        
        # Add some nodes representing neurites
        for i in range(10):
            for j in range(10):
                self.graph.add_node((i*20, j*20))
        
        # Add edges between adjacent nodes
        for i in range(10):
            for j in range(9):
                self.graph.add_edge((i*20, j*20), (i*20, (j+1)*20), weight=20.0)
                self.graph.add_edge((j*20, i*20), ((j+1)*20, i*20), weight=20.0)
        
        # Create a NeuriteGraph
        self.neurite_graph = NeuriteGraph(self.graph)
        
        # Create some cell bodies
        self.cell_bodies = [
            {
                "centroid": (50, 50),
                "area": 400,
                "bbox": (40, 40, 60, 60),
                "label": 1
            },
            {
                "centroid": (150, 150),
                "area": 300,
                "bbox": (140, 140, 160, 160),
                "label": 2
            }
        ]
        
        # Create a connector
        self.connector = NeuriteConnector(
            max_distance=50.0,
            connection_method="nearest"
        )
    
    def test_connect_nearest(self):
        """Test the nearest connection method."""
        # Connect cell bodies to neurites
        connected_graph = self.connector.connect(self.cell_bodies, self.neurite_graph)
        
        # Check that the graph has the expected number of nodes
        expected_nodes = len(self.graph.nodes()) + len(self.cell_bodies)
        self.assertEqual(connected_graph.number_of_nodes(), expected_nodes)
        
        # Check that each cell body is connected to at least one neurite
        cell_nodes = [node for node, attr in connected_graph.nodes(data=True) 
                     if attr.get("type") == "cell_body"]
        
        for cell_node in cell_nodes:
            # Get connected neurites
            connected_neurites = list(connected_graph.successors(cell_node))
            
            # Check that there is at least one connected neurite
            self.assertGreater(len(connected_neurites), 0)
    
    def test_analyze_connections(self):
        """Test the connection analysis."""
        # Connect cell bodies to neurites
        connected_graph = self.connector.connect(self.cell_bodies, self.neurite_graph)
        
        # Analyze connections
        analysis = self.connector.analyze_connections(connected_graph)
        
        # Check that the analysis has the expected structure
        self.assertIn("cell_stats", analysis)
        self.assertIn("overall_stats", analysis)
        
        # Check overall stats
        overall_stats = analysis["overall_stats"]
        self.assertEqual(overall_stats["num_cells"], len(self.cell_bodies))
        self.assertGreaterEqual(overall_stats["num_connected_cells"], 0)

if __name__ == "__main__":
    unittest.main()
