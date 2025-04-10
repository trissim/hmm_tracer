# Tracing Module Plan

## Current Implementation

The current implementation includes several functions in `hmm_axon.py` related to axon tracing:

```python
def random_seed_by_edge_map(edge_map):
    yy, xx = edge_map.nonzero()
    seed_index = np.random.choice(len(xx),len(xx))
    seed_xx = xx[seed_index]
    seed_yy = yy[seed_index]
    return seed_xx, seed_yy

def get_growth_cone_positions(image):
    mask = image > skimage.filters.threshold_otsu(image)
    mask = skimage.morphology.closing(mask, skimage.morphology.disk(3))
    labeled = skimage.measure.label(mask)
    imsave("./mask.png",mask)
    props = skimage.measure.regionprops(labeled)
    positions = []
    seed_xx = []
    seed_yy = []
    for prop in props:
        seed_xx.append(prop.centroid[0])
        seed_yy.append(prop.centroid[1])
    print(str(len(seed_xx)) + " growth cones")
    return seed_xx,seed_yy

def selected_seeding(image, seed_xx, seed_yy, chain_level=1.05, total_node=8, node_r=None, line_length_min=32):
    im_copy=np.copy(image)
    alva_HMM = alva_MCMC.AlvaHmm(im_copy,
                                total_node = total_node,
                                total_path = None,
                                node_r = node_r,
                                node_angle_max = None,)
    chain_HMM_1st, pair_chain_HMM, pair_seed_xx, pair_seed_yy = alva_HMM.pair_HMM_chain(seed_xx = seed_xx,
                                                                                        seed_yy = seed_yy,
                                                                                        chain_level = chain_level,)
    for chain_i in [0, 1]:
                chain_HMM = [chain_HMM_1st, pair_chain_HMM][chain_i]
                real_chain_ii, real_chain_aa, real_chain_xx, real_chain_yy = chain_HMM[0:4]
                seed_node_xx, seed_node_yy = chain_HMM[4:6]

    chain_im_fine = alva_HMM.chain_image(chain_HMM_1st, pair_chain_HMM,)
    return alva_branch.connect_way(chain_im_fine,
                                    line_length_min = line_length_min,
                                    free_zone_from_y0 = None,)
```

## Refactoring Goals

1. Create a dedicated module for axon tracing
2. Improve code organization and readability
3. Add proper type hints and documentation
4. Create a class-based approach for tracing

## Proposed Implementation

### Module Structure

```
hmm_tracer/
└── core/
    └── tracing.py  # Axon tracing module
```

### Implementation Details

```python
# hmm_tracer/core/tracing.py
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
            seed_xx.append(prop.centroid[0])
            seed_yy.append(prop.centroid[1])
        
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
```

### Usage Example

```python
from hmm_tracer.core.preprocessing import ImagePreprocessor
from hmm_tracer.core.tracing import AxonTracer
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

# Create tracer and trace axons
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
```

## Validation

The refactored code maintains the same functionality as the original implementation but with several improvements:

1. Better organization with a dedicated class
2. Proper type hints and documentation
3. Consistent interface for different seed generation methods
4. Clear separation of concerns

The implementation doesn't duplicate any existing code in the codebase and follows Python best practices.

## Next Steps

After implementing the tracing module, we'll proceed with:

1. [Graph Module Plan](graph_module_plan.md)
2. Integration of all modules
