# HMM Tracer Refactoring Summary Plan

## Project Overview

The HMM Tracer project is focused on axon tracing in microscopy images using Hidden Markov Models. The current implementation is a monolithic script that needs to be refactored into a proper Python package with modular components.

## Refactoring Goals

1. Create a proper Python package structure
2. Separate concerns into modular components
3. Set up version control
4. Maintain all existing functionality
5. Improve code organization and readability
6. Prepare for future extensions (like cell body detection)

## Detailed Plans

We've created several detailed plan files for different aspects of the refactoring:

1. [Project Analysis Plan](project_analysis_plan.md) - High-level analysis of the project
2. [Project Structure Plan](project_structure_plan.md) - Plan for the new project structure
3. [Image Processing Module Plan](image_processing_module_plan.md) - Plan for the image preprocessing module
4. [Tracing Module Plan](tracing_module_plan.md) - Plan for the axon tracing module
5. [Graph Module Plan](graph_module_plan.md) - Plan for the graph operations module
6. [CLI Module Plan](cli_module_plan.md) - Plan for the command-line interface
7. [Integration Plan](integration_plan.md) - Plan for integrating all modules

## Implementation Roadmap

### Phase 1: Project Setup

1. Initialize Git repository
2. Create basic package structure
3. Create setup files
4. Create README.md

### Phase 2: Core Modules Implementation

1. Implement `ImagePreprocessor` class in `hmm_tracer/core/preprocessing.py`
2. Implement `AxonTracer` class in `hmm_tracer/core/tracing.py`
3. Implement `NeuriteGraph` class in `hmm_tracer/core/graph.py`

### Phase 3: CLI and Integration

1. Implement `BatchProcessor` class and `main()` function in `hmm_tracer/cli.py`
2. Create integration tests
3. Validate against original implementation

### Phase 4: Documentation and Finalization

1. Update documentation
2. Create example scripts
3. Final testing and validation

## New Package Structure

```
hmm_tracer/
├── .git/                      # Git repository
├── .gitignore                 # Git ignore file
├── README.md                  # Project documentation
├── setup.py                   # Package installation
├── requirements.txt           # Dependencies
├── hmm_tracer/                # Main package
│   ├── __init__.py            # Package initialization
│   ├── core/                  # Core functionality
│   │   ├── __init__.py
│   │   ├── preprocessing.py   # Image preprocessing
│   │   ├── tracing.py         # Axon tracing algorithms
│   │   └── graph.py           # Graph extraction and analysis
│   ├── utils/                 # Utility functions
│   │   ├── __init__.py
│   │   └── image_utils.py     # Image utility functions
│   └── cli.py                 # Command-line interface
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_tracing.py
│   └── test_graph.py
└── examples/                  # Usage examples
    └── batch_processing.py
```

## Key Classes and Functions

### ImagePreprocessor

```python
class ImagePreprocessor:
    @staticmethod
    def normalize(image, percentile=99.9)
    
    @staticmethod
    def apply_edge_detection(image, method="blob", **kwargs)
    
    @staticmethod
    def boundary_masking_canny(image)
    
    @staticmethod
    def boundary_masking_threshold(image, threshold_func=threshold_li, min_size=2)
    
    @staticmethod
    def boundary_masking_blob(image, min_sigma=1, max_sigma=2, threshold=0.02)
```

### AxonTracer

```python
class AxonTracer:
    def __init__(self, chain_level=1.05, total_node=None, node_r=None, line_length_min=32, debug=False)
    
    @staticmethod
    def generate_random_seeds(edge_map)
    
    @staticmethod
    def detect_growth_cones(image, save_mask=False, mask_path="./mask.png")
    
    def trace_from_seeds(self, image, seed_xx, seed_yy)
    
    def trace_image(self, image, edge_map, seed_method="random")
```

### NeuriteGraph

```python
class NeuriteGraph:
    def __init__(self, graph=None)
    
    @staticmethod
    def euclidean_distance(x1, y1, x2, y2)
    
    @classmethod
    def from_paths(cls, root_tree_xx, root_tree_yy)
    
    def get_total_length(self)
    
    def save_as_image(self, image_path, original_image, value=1.0)
    
    def get_graph_statistics(self)
```

### BatchProcessor

```python
class BatchProcessor:
    def __init__(self, input_folder, output_folder, chain_level=1.1, total_node=None, node_r=4, line_length_min=32, min_sigma=1, max_sigma=64, threshold=0.015, debug=True, num_cores=None)
    
    def process_file(self, filename)
    
    def process_all(self)
```

## Handling alva_machinery

For now, we'll continue using the existing alva_machinery package. In the future, we can fork it and make improvements.

## Future Extensions

After completing the refactoring, we'll be well-positioned to add new features:

1. Cell body detection
2. Connection of axons to cell bodies
3. More sophisticated graph analysis
4. Improved visualization

## Conclusion

This refactoring will transform the HMM Tracer project from a monolithic script into a well-structured Python package with modular components. The new structure will make it easier to maintain, extend, and use the code, while preserving all existing functionality.
