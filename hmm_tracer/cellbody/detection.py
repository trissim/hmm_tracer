"""
Cell body detection module for HMM Tracer.

This module provides algorithms for detecting cell bodies in microscopy images.
"""

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
    
    def visualize(
        self,
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
        from skimage.color import gray2rgb
        from skimage.draw import circle_perimeter
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
        
        # Draw cell body outlines
        for cell in cell_bodies:
            y, x = cell["centroid"]
            
            # Calculate radius
            if "radius" in cell:
                radius = int(cell["radius"])
            else:
                # Estimate radius from area
                radius = int(np.sqrt(cell["area"] / np.pi))
            
            # Draw circle
            rr, cc = circle_perimeter(int(y), int(x), radius, shape=vis_image.shape)
            vis_image[rr, cc] = color
        
        # Save visualization
        imsave(output_path, vis_image)
