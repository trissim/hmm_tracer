# Project Structure Plan

## Current Structure

The current project structure is minimal:

```
hmm_tracer/
├── hmm_axon.py                # Main script with all functionality
├── alva_machinery-0.0.6.tar.gz # Dependency package archive
├── alva_machinery-0.0.6/      # Extracted dependency package
├── graph.png                  # Output file
├── mask.png                   # Output file
└── results.csv                # Output file
```

## Proposed Structure

The proposed structure follows Python package best practices:

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

## Implementation Steps

### 1. Initialize Git Repository

```bash
# Navigate to project directory
cd /home/ts/code/projects/hmm_tracer

# Initialize git repository
git init

# Create .gitignore file
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE files
.idea/
.vscode/
*.swp
*.swo

# Project specific
results.csv
*.png
*.tif
EOF
```

### 2. Create Basic Package Structure

```bash
# Create directory structure
mkdir -p hmm_tracer/core
mkdir -p hmm_tracer/utils
mkdir -p tests
mkdir -p examples

# Create empty __init__.py files
touch hmm_tracer/__init__.py
touch hmm_tracer/core/__init__.py
touch hmm_tracer/utils/__init__.py
touch tests/__init__.py
```

### 3. Create Setup Files

#### setup.py

```python
from setuptools import setup, find_packages

setup(
    name="hmm_tracer",
    version="0.1.0",
    description="A package for tracing axons using Hidden Markov Models",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-image",
        "networkx",
        "pandas",
        "matplotlib",
        # Note: alva_machinery will be handled separately
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.6",
)
```

#### requirements.txt

```
numpy>=1.19.0
scikit-image>=0.17.2
networkx>=2.5
pandas>=1.1.0
matplotlib>=3.3.0
```

### 4. Create README.md

```markdown
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

# Create a tracer with default parameters
tracer = AxonTracer()

# Trace axons in an image
graph, length = tracer.trace_image("input_image.tif", "output_image.png")
print(f"Total axon length: {length}")
```

## Command Line Interface

```bash
# Process a single image
hmm_tracer --input input_image.tif --output output_directory

# Process a directory of images
hmm_tracer --input input_directory --output output_directory
```
```

### 5. Handle alva_machinery Dependency

For now, we'll continue using the existing alva_machinery package. In the future, we can fork it and make improvements.

### 6. Initial Commit

```bash
# Add all files to git
git add .

# Make initial commit
git commit -m "Initial project structure"
```

## Validation

This structure follows Python package best practices and provides a solid foundation for the refactored code. It separates concerns into logical modules and prepares for future extensions.

## Next Steps

After setting up the project structure, we'll proceed with refactoring the code into the appropriate modules. See the following plan files for details:

- [Image Processing Module Plan](image_processing_module_plan.md)
- [Tracing Module Plan](tracing_module_plan.md)
- [Graph Module Plan](graph_module_plan.md)
- [CLI Module Plan](cli_module_plan.md)
