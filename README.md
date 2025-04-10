# HMM Tracer

A package for tracing axons in microscopy images using Hidden Markov Models.

## Features

- Axon tracing using HMM algorithms
- Graph-based representation of neurites
- Measurement of neurite length
- Batch processing with multiprocessing

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hmm_tracer.git
cd hmm_tracer

# Install in development mode
pip install -e .
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

## License

[MIT License](LICENSE)
