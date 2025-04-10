"""
Image preprocessing module for HMM Tracer.

This module provides functions for preprocessing microscopy images
for axon tracing, including normalization and edge detection.
"""

from typing import Optional, Callable, Union, Tuple
import numpy as np
from skimage.feature import canny
from skimage.feature import blob_dog as local_max
from skimage.filters import median, threshold_li
from skimage.morphology import skeletonize
from skimage.io import imsave

class ImagePreprocessor:
    """
    Class for preprocessing microscopy images for axon tracing.
    
    This class provides methods for normalizing images, detecting edges,
    and generating masks for seed point generation.
    """
    
    @staticmethod
    def normalize(
        image: np.ndarray, 
        percentile: float = 99.9
    ) -> np.ndarray:
        """
        Normalize image intensity based on percentile value.
        
        Args:
            image: Input image as numpy array
            percentile: Percentile value for normalization (default: 99.9)
            
        Returns:
            Normalized image
        """
        percentile_value = np.percentile(image, percentile)
        normalized = image / percentile_value
        normalized = np.clip(normalized, 0, 100)
        return normalized
    
    @staticmethod
    def apply_edge_detection(
        image: np.ndarray,
        method: str = "blob",
        **kwargs
    ) -> np.ndarray:
        """
        Apply edge detection to an image using various methods.
        
        Args:
            image: Input image as numpy array
            method: Edge detection method ('canny', 'threshold', or 'blob')
            **kwargs: Additional parameters for the specific method
            
        Returns:
            Binary edge mask
        """
        if method == "canny":
            return ImagePreprocessor.boundary_masking_canny(image)
        elif method == "threshold":
            threshold_func = kwargs.get("threshold_func", threshold_li)
            min_size = kwargs.get("min_size", 2)
            return ImagePreprocessor.boundary_masking_threshold(
                image, threshold_func, min_size
            )
        elif method == "blob":
            min_sigma = kwargs.get("min_sigma", 1)
            max_sigma = kwargs.get("max_sigma", 2)
            threshold = kwargs.get("threshold", 0.02)
            return ImagePreprocessor.boundary_masking_blob(
                image, min_sigma, max_sigma, threshold
            )
        else:
            raise ValueError(f"Unknown edge detection method: {method}")
    
    @staticmethod
    def boundary_masking_canny(image: np.ndarray) -> np.ndarray:
        """
        Apply Canny edge detection with boundary masking.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Binary edge mask
        """
        bool_im_axon_edit = canny(image)
        # Mask boundaries
        bool_im_axon_edit[:,:2] = False
        bool_im_axon_edit[:,-2:] = False
        bool_im_axon_edit[:2,:] = False
        bool_im_axon_edit[-2:,:] = False
        return np.array(bool_im_axon_edit, dtype=np.int64)
    
    @staticmethod
    def boundary_masking_threshold(
        image: np.ndarray,
        threshold_func: Callable = threshold_li,
        min_size: int = 2
    ) -> np.ndarray:
        """
        Apply threshold-based edge detection with boundary masking.
        
        Args:
            image: Input image as numpy array
            threshold_func: Thresholding function to use
            min_size: Minimum size of objects to keep
            
        Returns:
            Binary edge mask
        """
        threshed = threshold_func(image)
        bool_image = image > threshed
        # Mask boundaries
        bool_image[:,:2] = False
        bool_image[:,-2:] = False
        bool_image[:2,:] = False
        bool_image[-2:,:] = False
        cleaned_bool_im_axon_edit = skeletonize(bool_image)
        return np.array(bool_image, dtype=np.int64)
    
    @staticmethod
    def boundary_masking_blob(
        image: np.ndarray,
        min_sigma: float = 1,
        max_sigma: float = 2,
        threshold: float = 0.02
    ) -> np.ndarray:
        """
        Apply blob detection with boundary masking.
        
        Args:
            image: Input image as numpy array
            min_sigma: Minimum sigma for blob detection
            max_sigma: Maximum sigma for blob detection
            threshold: Threshold for blob detection
            
        Returns:
            Binary edge mask
        """
        # Apply default values if None
        min_sigma = 1 if min_sigma is None else min_sigma
        max_sigma = 2 if max_sigma is None else max_sigma
        threshold = 0.02 if threshold is None else threshold
        
        # Apply median filter
        image_median = median(image)
        
        # Detect blobs
        galaxy = local_max(
            image_median, 
            min_sigma=min_sigma, 
            max_sigma=max_sigma, 
            threshold=threshold
        )
        
        # Create mask from blob positions
        yy = np.int64(galaxy[:, 0])
        xx = np.int64(galaxy[:, 1])
        boundary_mask = np.zeros_like(image)
        boundary_mask[yy, xx] = 1
        
        return boundary_mask
    
    @staticmethod
    def save_image(
        image_path: str,
        image: np.ndarray,
        plugin: str = 'pil',
        format_str: str = 'png'
    ) -> None:
        """
        Save an image to disk.
        
        Args:
            image_path: Path to save the image
            image: Image to save
            plugin: Plugin to use for saving
            format_str: Format string for saving
        """
        imsave(image_path, image, plugin=plugin, format_str=format_str)
