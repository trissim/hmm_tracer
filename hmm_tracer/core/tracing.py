"""
Axon tracing module for HMM Tracer.

This module provides functions for tracing axons in microscopy images
using Hidden Markov Models from the alva_machinery package.
"""

from typing import Tuple, List, Optional, Dict, Any, Union
import numpy as np
import alva_machinery.markov.aChain as alva_MCMC
import alva_machinery.branching.aWay as alva_branch
from skimage.filters import threshold_otsu
from skimage.morphology import closing, disk
from skimage.measure import label, regionprops
from skimage.io import imsave

class AxonTracer:
    """
    Class for tracing axons in microscopy images using Hidden Markov Models.

    This class provides methods for generating seed points and tracing axons
    using the alva_machinery HMM implementation.
    """

    def __init__(
        self,
        chain_level: float = 1.05,
        total_node: Optional[int] = None,
        node_r: Optional[int] = None,
        line_length_min: int = 32,
        debug: bool = False
    ):
        """
        Initialize the AxonTracer with tracing parameters.

        Args:
            chain_level: Parameter controlling the HMM chain level
            total_node: Number of nodes in the HMM chain
            node_r: Node radius parameter
            line_length_min: Minimum line length to consider
            debug: Whether to save debug images
        """
        self.chain_level = chain_level
        self.total_node = total_node
        self.node_r = node_r
        self.line_length_min = line_length_min
        self.debug = debug

    @staticmethod
    def generate_random_seeds(
        edge_map: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate random seed points from an edge map.

        Args:
            edge_map: Binary edge map

        Returns:
            Tuple of (seed_xx, seed_yy) arrays
        """
        yy, xx = edge_map.nonzero()
        seed_index = np.random.choice(len(xx), len(xx))
        seed_xx = xx[seed_index]
        seed_yy = yy[seed_index]
        return seed_xx, seed_yy

    @staticmethod
    def detect_growth_cones(
        image: np.ndarray,
        save_mask: bool = False,
        mask_path: str = "./mask.png"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect growth cones in an image for seed point generation.

        Args:
            image: Input image as numpy array
            save_mask: Whether to save the mask image
            mask_path: Path to save the mask image

        Returns:
            Tuple of (seed_xx, seed_yy) arrays
        """
        # Create binary mask
        mask = image > threshold_otsu(image)
        mask = closing(mask, disk(3))

        # Label connected components
        labeled = label(mask)

        # Save mask if requested
        if save_mask:
            imsave(mask_path, mask)

        # Extract region properties
        props = regionprops(labeled)

        # Extract centroids as seed points
        seed_xx = []
        seed_yy = []
        for prop in props:
            # Note: In the original code, x and y were swapped
            # Original: seed_xx.append(prop.centroid[0])
            # Original: seed_yy.append(prop.centroid[1])
            # We're keeping the same convention as in generate_random_seeds
            seed_yy.append(prop.centroid[0])  # y coordinate is first in centroid
            seed_xx.append(prop.centroid[1])  # x coordinate is second in centroid

        print(f"{len(seed_xx)} growth cones detected")

        return np.array(seed_xx), np.array(seed_yy)

    def trace_from_seeds(
        self,
        image: np.ndarray,
        seed_xx: np.ndarray,
        seed_yy: np.ndarray
    ) -> Tuple[List, List, List, List]:
        """
        Trace axons from seed points using HMM.

        Args:
            image: Input image as numpy array
            seed_xx: X coordinates of seed points
            seed_yy: Y coordinates of seed points

        Returns:
            Tuple of (root_tree_yy, root_tree_xx, root_tip_yy, root_tip_xx)
        """
        # Create a copy of the image
        im_copy = np.copy(image)

        # Initialize AlvaHmm
        alva_hmm = alva_MCMC.AlvaHmm(
            im_copy,
            total_node=self.total_node,
            total_path=None,
            node_r=self.node_r,
            node_angle_max=None,
        )

        # Run HMM chain
        chain_hmm_1st, pair_chain_hmm, pair_seed_xx, pair_seed_yy = alva_hmm.pair_HMM_chain(
            seed_xx=seed_xx,
            seed_yy=seed_yy,
            chain_level=self.chain_level,
        )

        # Process chain results (this loop in the original code doesn't seem to do anything)
        for chain_i in [0, 1]:
            chain_hmm = [chain_hmm_1st, pair_chain_hmm][chain_i]
            real_chain_ii, real_chain_aa, real_chain_xx, real_chain_yy = chain_hmm[0:4]
            seed_node_xx, seed_node_yy = chain_hmm[4:6]

        # Generate chain image
        chain_im_fine = alva_hmm.chain_image(chain_hmm_1st, pair_chain_hmm)

        # Connect paths
        return alva_branch.connect_way(
            chain_im_fine,
            line_length_min=self.line_length_min,
            free_zone_from_y0=None,
        )

    def trace_image(
        self,
        image: np.ndarray,
        edge_map: np.ndarray,
        seed_method: str = "random"
    ) -> Tuple[List, List, List, List]:
        """
        Trace axons in an image using the specified seed method.

        Args:
            image: Input image as numpy array
            edge_map: Binary edge map for seed generation
            seed_method: Method for generating seeds ('random' or 'growth_cones')

        Returns:
            Tuple of (root_tree_yy, root_tree_xx, root_tip_yy, root_tip_xx)
        """
        # Generate seed points
        if seed_method == "random":
            seed_xx, seed_yy = self.generate_random_seeds(edge_map)
        elif seed_method == "growth_cones":
            seed_xx, seed_yy = self.detect_growth_cones(
                image,
                save_mask=self.debug
            )
        else:
            raise ValueError(f"Unknown seed method: {seed_method}")

        # Trace from seeds
        return self.trace_from_seeds(image, seed_xx, seed_yy)
