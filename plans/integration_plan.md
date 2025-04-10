# Integration Plan

## Overview

Now that we have detailed plans for each module, we need to integrate them into a cohesive package. This plan outlines the steps to implement all modules, ensure they work together correctly, and validate the refactored code against the original implementation.

## Implementation Order

We'll implement the modules in the following order:

1. Project structure setup
2. Image preprocessing module
3. Tracing module
4. Graph module
5. CLI module
6. Integration testing

## Implementation Steps

### 1. Project Structure Setup

Follow the steps in [Project Structure Plan](project_structure_plan.md) to:
- Initialize Git repository
- Create basic package structure
- Create setup files
- Create README.md

### 2. Image Preprocessing Module

Implement the `ImagePreprocessor` class in `hmm_tracer/core/preprocessing.py` as outlined in [Image Processing Module Plan](image_processing_module_plan.md).

Key functions to implement:
- `normalize()`
- `apply_edge_detection()`
- `boundary_masking_canny()`
- `boundary_masking_threshold()`
- `boundary_masking_blob()`

### 3. Tracing Module

Implement the `AxonTracer` class in `hmm_tracer/core/tracing.py` as outlined in [Tracing Module Plan](tracing_module_plan.md).

Key functions to implement:
- `generate_random_seeds()`
- `detect_growth_cones()`
- `trace_from_seeds()`
- `trace_image()`

### 4. Graph Module

Implement the `NeuriteGraph` class in `hmm_tracer/core/graph.py` as outlined in [Graph Module Plan](graph_module_plan.md).

Key functions to implement:
- `euclidean_distance()`
- `from_paths()`
- `get_total_length()`
- `save_as_image()`
- `get_graph_statistics()`

### 5. CLI Module

Implement the `BatchProcessor` class and `main()` function in `hmm_tracer/cli.py` as outlined in [CLI Module Plan](cli_module_plan.md).

Key functions to implement:
- `process_file()`
- `process_all()`
- `main()`

### 6. Integration Testing

Create a simple test script to validate that all modules work together correctly:

```python
# test_integration.py
import os
from skimage.io import imread
from hmm_tracer.core.preprocessing import ImagePreprocessor
from hmm_tracer.core.tracing import AxonTracer
from hmm_tracer.core.graph import NeuriteGraph

def test_single_image(image_path, output_path):
    """Test the full pipeline on a single image."""
    # Load and preprocess image
    image = imread(image_path)
    normalized = ImagePreprocessor.normalize(image)
    
    # Generate edge map
    edge_map = ImagePreprocessor.apply_edge_detection(
        normalized,
        method="blob",
        min_sigma=1,
        max_sigma=64,
        threshold=0.015
    )
    
    # Trace axons
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
    
    # Create graph
    neurite_graph = NeuriteGraph.from_paths(root_tree_yy, root_tree_xx)
    distance = neurite_graph.get_total_length()
    
    # Save output image
    neurite_graph.save_as_image(output_path, normalized)
    
    print(f"Processed {image_path}")
    print(f"Total axon length: {distance}")
    print(f"Output saved to {output_path}")
    
    return distance

if __name__ == "__main__":
    # Test on a sample image
    image_path = "path/to/sample/image.tif"
    output_path = "output.png"
    
    if os.path.exists(image_path):
        test_single_image(image_path, output_path)
    else:
        print(f"Sample image not found: {image_path}")
        print("Please provide a valid image path.")
```

## Validation Strategy

To ensure the refactored code maintains the same functionality as the original implementation, we'll:

1. **Compare outputs**: Process the same image with both the original and refactored code and compare the results
2. **Visual inspection**: Visually inspect the traced images to ensure they look similar
3. **Measure performance**: Compare processing time between the original and refactored code

## Handling the alva_machinery Dependency

For now, we'll continue using the existing alva_machinery package. In the future, we can fork it and make improvements. To ensure it's properly installed:

1. Add it to the `setup.py` dependencies:
   ```python
   setup(
       # ... other setup parameters ...
       install_requires=[
           # ... other dependencies ...
           "alva_machinery==0.0.6",
       ],
   )
   ```

2. Include the package in the repository or provide installation instructions in the README.

## Documentation

Update the README.md with:
- Installation instructions
- Usage examples
- Parameter descriptions
- Example outputs

## Next Steps After Integration

1. **Testing**: Create more comprehensive tests
2. **Documentation**: Add more detailed documentation
3. **Examples**: Create example scripts for common use cases
4. **Future features**: Prepare for cell body detection and other planned features
