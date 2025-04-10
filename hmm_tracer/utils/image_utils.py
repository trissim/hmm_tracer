"""
Image utility functions for HMM Tracer.

This module provides utility functions for working with images.
"""

import os
import numpy as np
from typing import List, Tuple, Optional
from skimage.io import imread, imsave
from skimage.transform import resize

def load_image(
    image_path: str,
    normalize: bool = True,
    percentile: float = 99.9
) -> np.ndarray:
    """
    Load an image from disk and optionally normalize it.
    
    Args:
        image_path: Path to the image file
        normalize: Whether to normalize the image
        percentile: Percentile value for normalization
        
    Returns:
        Loaded image as numpy array
    """
    # Check if the file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load the image
    image = imread(image_path)
    
    # Normalize if requested
    if normalize:
        percentile_value = np.percentile(image, percentile)
        image = image / percentile_value
        image = np.clip(image, 0, 100)
    
    return image

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
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(image_path)), exist_ok=True)
    
    # Save the image
    imsave(image_path, image, plugin=plugin, format_str=format_str)

def resize_image(
    image: np.ndarray,
    scale: float = 1.0,
    output_shape: Optional[Tuple[int, int]] = None,
    preserve_range: bool = True
) -> np.ndarray:
    """
    Resize an image.
    
    Args:
        image: Image to resize
        scale: Scale factor (ignored if output_shape is provided)
        output_shape: Output shape (height, width)
        preserve_range: Whether to preserve the value range
        
    Returns:
        Resized image
    """
    if output_shape is not None:
        return resize(image, output_shape, preserve_range=preserve_range)
    else:
        new_shape = (int(image.shape[0] * scale), int(image.shape[1] * scale))
        return resize(image, new_shape, preserve_range=preserve_range)

def crop_image(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Crop an image to a bounding box.
    
    Args:
        image: Image to crop
        bbox: Bounding box (y_min, x_min, y_max, x_max)
        
    Returns:
        Cropped image
    """
    y_min, x_min, y_max, x_max = bbox
    return image[y_min:y_max, x_min:x_max]

def pad_image(
    image: np.ndarray,
    pad_width: int,
    mode: str = 'constant',
    constant_values: float = 0
) -> np.ndarray:
    """
    Pad an image.
    
    Args:
        image: Image to pad
        pad_width: Width of padding
        mode: Padding mode
        constant_values: Value to use for constant padding
        
    Returns:
        Padded image
    """
    return np.pad(
        image,
        pad_width=((pad_width, pad_width), (pad_width, pad_width)),
        mode=mode,
        constant_values=constant_values
    )
