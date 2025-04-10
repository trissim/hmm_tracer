# Cell Body Detection Implementation Plan

## Overview

This plan outlines the implementation of cell body detection and connection to traced neurites in the HMM Tracer package. This feature allows the detection of neuronal cell bodies in microscopy images and connects them to the traced neurites.

## Current Status

The initial implementation of the cell body detection and connection modules has been completed:

1. Created the `hmm_tracer/cellbody/` directory with:
   - `__init__.py` - Module initialization
   - `detection.py` - Cell body detection algorithms
   - `connection.py` - Algorithms for connecting cell bodies to neurites

2. Added an example script:
   - `examples/cell_body_detection.py` - Example of using the cell body detection and connection modules

3. Added tests:
   - `tests/test_cellbody.py` - Tests for the cell body detection and connection modules

## Goals

1. Refine and optimize the cell body detection algorithms
2. Improve the connection algorithms for better accuracy
3. Add more visualization options
4. Integrate with the existing CLI for batch processing

## Implementation Approach

### 1. Refine Cell Body Detection Algorithms

The current implementation includes three detection methods:
- Threshold-based detection
- Local threshold-based detection
- Blob detection

Improvements to consider:
- Add more preprocessing options (e.g., contrast enhancement, denoising)
- Implement machine learning-based detection (e.g., using scikit-learn or a simple CNN)
- Add more parameters for fine-tuning detection

```python
# Example of adding preprocessing options
def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
    """
    Preprocess the image for cell body detection.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Preprocessed image
    """
    # Apply denoising
    if self.denoise:
        from skimage.restoration import denoise_bilateral
        image = denoise_bilateral(image, sigma_color=0.1, sigma_spatial=1)
    
    # Apply contrast enhancement
    if self.enhance_contrast:
        from skimage.exposure import equalize_adapthist
        image = equalize_adapthist(image)
    
    return image
```

### 2. Improve Connection Algorithms

The current implementation includes two connection methods:
- Connect to nearest neurite point
- Connect to all neurite points within a distance

Improvements to consider:
- Implement more sophisticated connection methods (e.g., based on neurite direction)
- Add options for filtering connections (e.g., by length, angle)
- Improve the analysis of connections

```python
# Example of direction-based connection
def _connect_direction_based(
    self,
    cell_bodies: List[Dict[str, Any]],
    neurite_graph: NeuriteGraph
) -> nx.DiGraph:
    """
    Connect each cell body to neurites based on direction.
    
    Args:
        cell_bodies: List of cell body dictionaries
        neurite_graph: NeuriteGraph instance
        
    Returns:
        Directed graph with cell bodies and neurites
    """
    # Implementation details...
    pass
```

### 3. Add More Visualization Options

The current implementation includes basic visualization for:
- Cell bodies
- Connections between cell bodies and neurites

Improvements to consider:
- Add more customization options (e.g., colors, line styles)
- Implement interactive visualization (e.g., using matplotlib or plotly)
- Add options for saving visualizations in different formats

```python
# Example of interactive visualization
def visualize_interactive(
    self,
    image: np.ndarray,
    connected_graph: nx.DiGraph,
    output_path: Optional[str] = None
) -> None:
    """
    Create an interactive visualization of cell bodies and neurites.
    
    Args:
        image: Input image
        connected_graph: Directed graph with cell bodies and neurites
        output_path: Optional path to save the visualization
    """
    # Implementation details...
    pass
```

### 4. Integrate with CLI

The current CLI doesn't include options for cell body detection. We need to update it to include:
- Options for enabling cell body detection
- Parameters for controlling detection and connection
- Output options for cell body analysis

```python
# Example of CLI integration
def main():
    """
    Main function for the command-line interface.
    """
    parser = argparse.ArgumentParser(description='Trace axons in microscopy images.')
    
    # Existing arguments...
    
    # Add cell body detection arguments
    parser.add_argument('--detect-cell-bodies', action='store_true',
                        help='Detect cell bodies in the images')
    parser.add_argument('--cell-method', choices=['threshold', 'local_threshold', 'blob'],
                        default='threshold', help='Cell body detection method')
    parser.add_argument('--cell-min-size', type=int, default=100,
                        help='Minimum size of cell bodies to detect')
    parser.add_argument('--cell-max-size', type=int, default=None,
                        help='Maximum size of cell bodies to detect')
    parser.add_argument('--connect-method', choices=['nearest', 'all_within_distance'],
                        default='nearest', help='Method for connecting cell bodies to neurites')
    parser.add_argument('--max-distance', type=float, default=50.0,
                        help='Maximum distance for connecting cell bodies to neurites')
    
    # Process arguments and run...
```

## Testing Approach

1. **Unit Tests**:
   - Test each detection method with different parameters
   - Test connection methods with different graph structures
   - Test visualization functions

2. **Integration Tests**:
   - Test the full pipeline from image loading to visualization
   - Test CLI integration

3. **Performance Testing**:
   - Measure execution time for different image sizes
   - Optimize performance bottlenecks

## Validation

To validate the cell body detection:

1. Compare detected cell bodies with manual annotations
2. Calculate precision, recall, and F1 score
3. Evaluate the quality of connections between cell bodies and neurites
4. Verify that the analysis results match expectations

## Next Steps

1. Implement the refinements outlined in this plan
2. Add more tests for the new functionality
3. Update documentation to include the cell body detection feature
4. Create more example scripts demonstrating different use cases
