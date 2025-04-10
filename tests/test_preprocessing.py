"""
Tests for the image preprocessing module.
"""

import unittest
import numpy as np
from hmm_tracer.core.preprocessing import ImagePreprocessor

class TestImagePreprocessor(unittest.TestCase):
    """Test cases for the ImagePreprocessor class."""
    
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
    
    def test_normalize(self):
        """Test the normalize method."""
        # Normalize the image
        normalized = ImagePreprocessor.normalize(self.test_image)
        
        # Check that the values are in the expected range
        self.assertLessEqual(normalized.max(), 100)
        self.assertGreaterEqual(normalized.min(), 0)
        
        # Check that the shape is preserved
        self.assertEqual(normalized.shape, self.test_image.shape)
    
    def test_boundary_masking_canny(self):
        """Test the boundary_masking_canny method."""
        # Apply Canny edge detection
        edge_map = ImagePreprocessor.boundary_masking_canny(self.test_image)
        
        # Check that the output is binary
        self.assertTrue(np.all(np.isin(edge_map, [0, 1])))
        
        # Check that the shape is preserved
        self.assertEqual(edge_map.shape, self.test_image.shape)
        
        # Check that the boundaries are masked
        self.assertEqual(edge_map[0, 0], 0)
        self.assertEqual(edge_map[-1, -1], 0)
    
    def test_boundary_masking_blob(self):
        """Test the boundary_masking_blob method."""
        # Apply blob detection
        edge_map = ImagePreprocessor.boundary_masking_blob(
            self.test_image,
            min_sigma=1,
            max_sigma=2,
            threshold=0.1
        )
        
        # Check that the output is binary
        self.assertTrue(np.all(np.isin(edge_map, [0, 1])))
        
        # Check that the shape is preserved
        self.assertEqual(edge_map.shape, self.test_image.shape)
    
    def test_apply_edge_detection(self):
        """Test the apply_edge_detection method."""
        # Test with different methods
        methods = ["canny", "threshold", "blob"]
        
        for method in methods:
            edge_map = ImagePreprocessor.apply_edge_detection(
                self.test_image,
                method=method
            )
            
            # Check that the output is binary
            self.assertTrue(np.all(np.isin(edge_map, [0, 1])))
            
            # Check that the shape is preserved
            self.assertEqual(edge_map.shape, self.test_image.shape)
        
        # Test with invalid method
        with self.assertRaises(ValueError):
            ImagePreprocessor.apply_edge_detection(
                self.test_image,
                method="invalid"
            )

if __name__ == "__main__":
    unittest.main()
