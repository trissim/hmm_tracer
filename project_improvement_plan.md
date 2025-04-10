# Project Improvement Plan for HMM Tracer

## Description of Problem and Context

Based on my exploration, I understand that:

1. The project is focused on axon tracing using the alvahmm package
2. The main functionality is in `hmm_axon.py`, which uses the `alva_machinery` package for HMM-based tracing
3. The `alva_machinery` package is included as a tarball and extracted directory (version 0.0.6)
4. There's no current version control setup (.git directory)
5. The code needs to be organized better for maintainability and to add new features like cell body detection

## Proposed Solution

Here's my high-level plan to improve the project:

1. **Set up proper version control**
   - Initialize a Git repository
   - Create a proper .gitignore file
   - Create a README.md with project documentation

2. **Restructure the project**
   - Create a proper Python package structure for your code
   - Fork and include alvahmm as a submodule or dependency
   - Separate the main functionality into modules

3. **Improve code quality**
   - Add proper documentation
   - Clean up the code (remove commented code, improve variable names)
   - Add type hints
   - Create proper configuration management

4. **Add testing infrastructure**
   - Set up unit tests
   - Add sample data for testing

5. **Prepare for future features**
   - Create placeholder modules for cell body detection
   - Set up structure for neurite tracing with cell body connections

## Detailed Implementation Steps

### 1. Set up proper version control

```bash
# Initialize git repository
git init

# Create .gitignore file with Python-specific patterns
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

# Create a basic README.md
cat > README.md << EOF
# HMM Tracer

A package for tracing axons and neurons using Hidden Markov Models.

## Features

- Axon tracing using HMM
- Graph-based representation of neurites
- Measurement of neurite length

## Installation

\`\`\`bash
pip install -e .
\`\`\`

## Usage

\`\`\`python
from hmm_tracer import trace_axons

# Example usage
trace_axons(input_folder="path/to/images", output_folder="path/to/output")
\`\`\`
EOF
```

### 2. Restructure the project

Create a proper Python package structure:

```
hmm_tracer/
├── .git/
├── .gitignore
├── README.md
├── setup.py
├── requirements.txt
├── hmm_tracer/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── tracing.py  # Core tracing functionality
│   │   ├── preprocessing.py  # Image preprocessing
│   │   └── graph.py  # Graph extraction and analysis
│   ├── utils/
│   │   ├── __init__.py
│   │   └── image_utils.py  # Image utility functions
│   ├── cellbody/  # Future module for cell body detection
│   │   ├── __init__.py
│   │   └── detection.py
│   └── cli.py  # Command-line interface
├── tests/
│   ├── __init__.py
│   ├── test_tracing.py
│   └── test_graph.py
├── examples/
│   ├── simple_tracing.py
│   └── batch_processing.py
└── data/  # Sample data for examples and tests
    └── sample_image.tif
```

### 3. Improve code quality

Example of refactored code structure for the main tracing module:

```python
# hmm_tracer/core/tracing.py
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import networkx as nx
from skimage.io import imread, imsave
from skimage.filters import median
import alva_machinery.markov.aChain as alva_MCMC
import alva_machinery.branching.aWay as alva_branch

class AxonTracer:
    """
    Class for tracing axons in microscopy images using Hidden Markov Models.
    
    This class provides methods for detecting and tracing axons in microscopy images,
    and representing them as graph structures for further analysis.
    """
    
    def __init__(
        self, 
        chain_level: float = 1.05,
        total_node: Optional[int] = None,
        node_r: Optional[int] = None,
        line_length_min: int = 32,
        min_sigma: float = 1.0,
        max_sigma: float = 2.0,
        threshold: float = 0.02,
        debug: bool = False
    ):
        """
        Initialize the AxonTracer with tracing parameters.
        
        Args:
            chain_level: Parameter controlling the HMM chain level
            total_node: Number of nodes in the HMM chain
            node_r: Node radius parameter
            line_length_min: Minimum line length to consider
            min_sigma: Minimum sigma for blob detection
            max_sigma: Maximum sigma for blob detection
            threshold: Threshold for blob detection
            debug: Whether to save debug images
        """
        self.chain_level = chain_level
        self.total_node = total_node
        self.node_r = node_r
        self.line_length_min = line_length_min
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.threshold = threshold
        self.debug = debug
    
    def trace_image(self, image_path: str, output_path: str) -> Tuple[nx.Graph, float]:
        """
        Trace axons in a single image and save the result.
        
        Args:
            image_path: Path to the input image
            output_path: Path to save the output traced image
            
        Returns:
            Tuple containing the graph representation and the total axon length
        """
        # Implementation here
        pass
```

### 4. Add testing infrastructure

Example test file:

```python
# tests/test_tracing.py
import unittest
import numpy as np
from skimage.io import imread
from hmm_tracer.core.tracing import AxonTracer

class TestAxonTracer(unittest.TestCase):
    def setUp(self):
        self.tracer = AxonTracer()
        self.test_image = np.zeros((100, 100), dtype=np.float32)
        # Create a simple line pattern for testing
        self.test_image[40:60, 40] = 1.0
        self.test_image[40, 40:60] = 1.0
    
    def test_trace_image(self):
        # Test tracing functionality
        graph, length = self.tracer.trace_image_from_array(self.test_image)
        self.assertGreater(length, 0)
        self.assertGreater(len(graph.nodes), 0)
```

### 5. Prepare for future features

Example structure for cell body detection:

```python
# hmm_tracer/cellbody/detection.py
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import networkx as nx
from skimage.measure import regionprops, label
from skimage.filters import threshold_otsu
from skimage.morphology import closing, disk

class CellBodyDetector:
    """
    Class for detecting cell bodies in microscopy images.
    """
    
    def __init__(
        self,
        min_size: int = 100,
        max_size: Optional[int] = None,
        threshold_method: str = "otsu"
    ):
        """
        Initialize the cell body detector.
        
        Args:
            min_size: Minimum size of cell bodies to detect
            max_size: Maximum size of cell bodies to detect
            threshold_method: Method for thresholding the image
        """
        self.min_size = min_size
        self.max_size = max_size
        self.threshold_method = threshold_method
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect cell bodies in an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of dictionaries containing cell body properties
        """
        # Implementation placeholder
        # This will be implemented in the future
        pass
```

## Validation

I've checked the codebase and confirmed that the proposed structure doesn't conflict with existing code. The plan maintains the core functionality while improving organization and preparing for future features.

## Implementation Plan

Here's how we'll implement this plan:

1. First, set up the Git repository and basic project structure
2. Refactor the existing code into the new structure
3. Fork the alvahmm package and include it properly
4. Add tests and documentation
5. Prepare for future features
