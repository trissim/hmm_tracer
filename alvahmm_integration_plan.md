# alvahmm Integration Plan

## Overview

This plan outlines the approach for integrating the forked alvahmm package (https://github.com/trissim/alvahmm) with the hmm_tracer project. We'll set up a development environment that allows simultaneous work on both projects.

## Current Status

1. The hmm_tracer project has been refactored into a proper Python package
2. The alvahmm package is currently included as a static copy (version 0.0.6)
3. You have a fork of alvahmm at https://github.com/trissim/alvahmm

## Integration Goals

1. Replace the static alvahmm package with a Git submodule pointing to your fork
2. Set up the development environment to allow changes to both projects
3. Ensure hmm_tracer uses the local development version of alvahmm
4. Enable easy pushing of alvahmm changes to your GitHub fork

## Implementation Steps

### 1. Remove the Static alvahmm Package

```bash
# Remove the static alvahmm package
git rm -r alva_machinery-0.0.6/
git rm alva_machinery-0.0.6.tar.gz
git commit -m "Remove static alvahmm package"
```

### 2. Add the Forked alvahmm as a Git Submodule

```bash
# Add the forked alvahmm as a Git submodule
git submodule add https://github.com/trissim/alvahmm.git alvahmm
git commit -m "Add forked alvahmm as Git submodule"
```

### 3. Install the Local alvahmm Package in Development Mode

Update the setup.py file to handle the alvahmm dependency correctly:

```python
# setup.py
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
        # alvahmm will be installed separately in development mode
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.6",
    entry_points={
        'console_scripts': [
            'hmm_tracer=hmm_tracer.cli:main',
        ],
    },
)
```

Create an installation script that installs both packages in development mode:

```bash
#!/bin/bash
# install_dev.sh

# Install alvahmm in development mode
cd alvahmm
pip install -e .
cd ..

# Install hmm_tracer in development mode
pip install -e .

echo "Both packages installed in development mode"
```

### 4. Update Import Statements in hmm_tracer

Update the import statements in hmm_tracer to use the new package name if necessary:

```python
# Before
import alva_machinery.markov.aChain as alva_MCMC
import alva_machinery.branching.aWay as alva_branch

# After (if the package name has changed)
import alvahmm.markov.aChain as alva_MCMC
import alvahmm.branching.aWay as alva_branch
```

### 5. Create a Development Workflow Guide

Create a guide for the development workflow:

```markdown
# Development Workflow

This project consists of two components:
1. hmm_tracer - The main package for axon tracing
2. alvahmm - A forked dependency for HMM algorithms

## Setup

1. Clone the repository with submodules:
   ```bash
   git clone --recurse-submodules https://github.com/yourusername/hmm_tracer.git
   cd hmm_tracer
   ```

2. Install both packages in development mode:
   ```bash
   ./install_dev.sh
   ```

## Making Changes to hmm_tracer

1. Make your changes to the hmm_tracer code
2. Commit and push as usual:
   ```bash
   git add .
   git commit -m "Your commit message"
   git push
   ```

## Making Changes to alvahmm

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
```

## Validation

To validate the integration:

1. Clone a fresh copy of the repository with submodules
2. Run the installation script
3. Run the tests to ensure everything works correctly
4. Make a small change to alvahmm and verify that it affects hmm_tracer

## Next Steps

After integrating the forked alvahmm package:

1. Identify improvements to make to alvahmm
2. Implement and test those improvements
3. Push changes to both repositories
4. Continue developing hmm_tracer with the improved alvahmm
