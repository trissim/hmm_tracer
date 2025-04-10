# HMM Tracer Project Analysis

## High-Level Overview

The HMM Tracer project is focused on axon tracing in microscopy images using Hidden Markov Models. The project currently consists of:

1. A main script (`hmm_axon.py`) that contains all the functionality
2. A dependency on the `alva_machinery` package (version 0.0.6) for HMM-based tracing algorithms
3. No proper package structure or version control

The main functionality includes:
- Image preprocessing and normalization
- Edge detection and seed point generation
- Axon tracing using HMM algorithms
- Graph extraction from traced paths
- Measurement of axon length
- Batch processing with multiprocessing

## Current Architecture

The current architecture is monolithic, with all functionality in a single script:

```
hmm_tracer/
├── hmm_axon.py                # Main script with all functionality
├── alva_machinery-0.0.6.tar.gz # Dependency package archive
├── alva_machinery-0.0.6/      # Extracted dependency package
├── graph.png                  # Output file
├── mask.png                   # Output file
└── results.csv                # Output file
```

The `hmm_axon.py` script contains several functions that can be categorized as:

1. **Image preprocessing**:
   - `normalize()`: Normalizes image intensity
   - `boundary_masking_canny()`: Edge detection using Canny
   - `boundary_masking_threshold()`: Edge detection using thresholding
   - `boundary_masking_blob()`: Edge detection using blob detection

2. **Seed generation**:
   - `random_seed_by_edge_map()`: Generates random seed points from edge map
   - `get_growth_cone_positions()`: Detects growth cones for seeding

3. **Tracing**:
   - `selected_seeding()`: Performs HMM-based tracing from seed points

4. **Graph operations**:
   - `extract_graph()`: Converts traced paths to a networkx graph
   - `graph_to_length()`: Calculates total length of the graph
   - `graph_to_image()`: Converts graph to an image

5. **Processing pipeline**:
   - `process_file()`: Processes a single image file
   - `main()`: Orchestrates batch processing with multiprocessing

## Dependencies

The project relies on several external libraries:
- `scikit-image`: For image processing and feature detection
- `numpy`: For numerical operations
- `networkx`: For graph representation and analysis
- `pandas`: For data management and CSV output
- `alva_machinery`: For HMM-based tracing algorithms

## Refactoring Goals

The primary goals for refactoring are:

1. Create a proper Python package structure
2. Separate concerns into modular components
3. Set up version control
4. Maintain all existing functionality
5. Improve code organization and readability
6. Prepare for future extensions (like cell body detection)

## Refactoring Strategy

The refactoring will be approached in the following phases:

1. **Project Structure Setup**:
   - Initialize Git repository
   - Create basic package structure
   - Set up configuration files

2. **Code Modularization**:
   - Separate functionality into logical modules
   - Create proper class hierarchies
   - Maintain existing API where possible

3. **Testing and Validation**:
   - Ensure refactored code maintains the same behavior
   - Add basic tests for core functionality

4. **Documentation**:
   - Add proper docstrings
   - Create README and usage examples

## Integration with Forked alvahmm Package

We will integrate with the forked alvahmm package at https://github.com/trissim/alvahmm:

1. Include the forked alvahmm as a Git submodule
2. Make changes to alvahmm while developing hmm_tracer
3. Commit and push changes to the alvahmm fork
4. Configure hmm_tracer to use the local development version of alvahmm

## Next Steps

The next step is to break down each phase into specific tasks and create detailed plan files for each component to be refactored.

See the following plan files for detailed implementation:
- [Project Structure Plan](project_structure_plan.md)
- [Image Processing Module Plan](image_processing_module_plan.md)
- [Tracing Module Plan](tracing_module_plan.md)
- [Graph Module Plan](graph_module_plan.md)
- [CLI Module Plan](cli_module_plan.md)

## Note on Thought File Completion Tracking

This file is marked as complete with the `_complete` suffix, indicating that all points in this thought have been extensively addressed. However, complete thoughts can still be revisited and have more information added to them if necessary.

If significant additions are made to a complete thought, the `_complete` suffix may be removed until the new content is fully addressed. Once the additional content is completed, the `_complete` suffix can be added back.

This system helps track the progress of thought development while allowing for iterative refinement.
