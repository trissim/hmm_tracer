# HMM Tracer

A package for tracing axons in microscopy images using Hidden Markov Models. This package builds upon the [alvahmm](https://github.com/trissim/alvahmm) package for HMM-based tracing algorithms.

## Features

- Axon tracing using HMM algorithms
- Graph-based representation of neurites
- Measurement of neurite length
- Batch processing with multiprocessing
- Cell body detection (coming soon)

## Installation

```bash
# Clone the repository with submodules
git clone --recurse-submodules https://github.com/trissim/hmm_tracer.git
cd hmm_tracer

# Install both packages in development mode
./install_dev.sh

# Or install manually
cd alvahmm
pip install -e .
cd ..
pip install -e .
```

### If you already cloned without submodules

```bash
# Initialize and update the submodule
git submodule update --init --recursive

# Install both packages in development mode
./install_dev.sh
```

## Usage

```python
from hmm_tracer.core.tracing import AxonTracer
from hmm_tracer.core.preprocessing import ImagePreprocessor
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

# Create a tracer with default parameters
tracer = AxonTracer()

# Trace axons in an image
root_tree_yy, root_tree_xx, root_tip_yy, root_tip_xx = tracer.trace_image(
    normalized,
    edge_map,
    seed_method="random"
)

# Create graph and calculate length
from hmm_tracer.core.graph import NeuriteGraph
neurite_graph = NeuriteGraph.from_paths(root_tree_yy, root_tree_xx)
length = neurite_graph.get_total_length()
print(f"Total axon length: {length}")

# Save output image
neurite_graph.save_as_image("output.png", normalized)
```

## Command Line Interface

```bash
# Process a single image
hmm_tracer --input input_image.tif --output output_directory

# Process a directory of images
hmm_tracer --input input_directory --output output_directory

# Process with custom parameters
hmm_tracer --input input_directory --output output_directory \
    --chain-level 1.05 --node-r 8 --line-length-min 16 \
    --min-sigma 1 --max-sigma 32 --threshold 0.02 --debug
```

## Development Workflow

This project consists of two components:
1. hmm_tracer - The main package for axon tracing
2. alvahmm - A forked dependency for HMM algorithms

### Making Changes to hmm_tracer

1. Make your changes to the hmm_tracer code
2. Commit and push as usual:
   ```bash
   git add .
   git commit -m "Your commit message"
   git push
   ```

### Making Changes to alvahmm

1. Navigate to the alvahmm directory:
   ```bash
   cd alvahmm
   ```

2. Make your changes to the alvahmm code

3. Commit and push to your fork:
   ```bash
   git add .
   git commit -m "Your commit message"
   git push
   ```

4. Return to the hmm_tracer directory:
   ```bash
   cd ..
   ```

5. Update the hmm_tracer repository to point to the new alvahmm commit:
   ```bash
   git add alvahmm
   git commit -m "Update alvahmm submodule"
   git push
   ```

## Project Structure

- `hmm_tracer/` - Main package directory
  - `core/` - Core functionality modules
    - `preprocessing.py` - Image preprocessing
    - `tracing.py` - Axon tracing
    - `graph.py` - Graph operations
  - `utils/` - Utility functions
  - `cli.py` - Command-line interface
- `alvahmm/` - Forked alvahmm package (submodule)
- `plans/` - Planning documents
- `tests/` - Unit tests
- `examples/` - Example scripts

## License

[MIT License](LICENSE)
