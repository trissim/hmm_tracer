"""
Tests for the axon tracing module.
"""

import unittest
import numpy as np
from hmm_tracer.core.tracing import AxonTracer

class TestAxonTracer(unittest.TestCase):
    """Test cases for the AxonTracer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test image
        self.test_image = np.zeros((100, 100), dtype=np.float32)
        
        # Add some features to the test image
        self.test_image[40:60, 40:60] = 1.0
        self.test_image[20:30, 70:80] = 0.5
        
        # Add some noise
        np.random.seed(42)
        self.test_image += np.random.normal(0, 0.1, size=self.test_image.shape)
        
        # Ensure values are positive
        self.test_image = np.clip(self.test_image, 0, None)
        
        # Create a simple edge map
        self.edge_map = np.zeros_like(self.test_image, dtype=np.int64)
        self.edge_map[45, 45:55] = 1
        self.edge_map[45:55, 45] = 1
    
    def test_generate_random_seeds(self):
        """Test the generate_random_seeds method."""
        # Generate random seeds
        seed_xx, seed_yy = AxonTracer.generate_random_seeds(self.edge_map)
        
        # Check that the seeds are within the edge map
        for x, y in zip(seed_xx, seed_yy):
            self.assertEqual(self.edge_map[y, x], 1)
        
        # Check that the number of seeds matches the number of edge points
        self.assertEqual(len(seed_xx), np.sum(self.edge_map))
    
    def test_tracer_initialization(self):
        """Test the initialization of the AxonTracer class."""
        # Create a tracer with default parameters
        tracer = AxonTracer()
        
        # Check that the parameters are set correctly
        self.assertEqual(tracer.chain_level, 1.05)
        self.assertEqual(tracer.line_length_min, 32)
        self.assertEqual(tracer.debug, False)
        
        # Create a tracer with custom parameters
        tracer = AxonTracer(
            chain_level=1.1,
            total_node=64,
            node_r=8,
            line_length_min=16,
            debug=True
        )
        
        # Check that the parameters are set correctly
        self.assertEqual(tracer.chain_level, 1.1)
        self.assertEqual(tracer.total_node, 64)
        self.assertEqual(tracer.node_r, 8)
        self.assertEqual(tracer.line_length_min, 16)
        self.assertEqual(tracer.debug, True)

if __name__ == "__main__":
    unittest.main()
