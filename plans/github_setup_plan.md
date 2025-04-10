# GitHub Setup Plan

## Overview

This plan outlines the steps to set up GitHub repositories for both the hmm_tracer project and the forked alvahmm package, ensuring they work together correctly.

## Current Status

1. The hmm_tracer project has been refactored locally
2. You have a fork of alvahmm at https://github.com/trissim/alvahmm
3. We need to create a new GitHub repository for hmm_tracer

## Setup Goals

1. Create a new GitHub repository for hmm_tracer
2. Push the refactored hmm_tracer code to the new repository
3. Set up the alvahmm fork as a submodule
4. Configure both repositories for collaborative development

## Implementation Steps

### 1. Create a New GitHub Repository for hmm_tracer

1. Go to GitHub.com and sign in
2. Click the "+" icon in the top-right corner and select "New repository"
3. Name the repository "hmm_tracer"
4. Add a description: "A package for tracing axons using Hidden Markov Models"
5. Choose "Public" or "Private" visibility as preferred
6. Do not initialize with README, .gitignore, or license (we already have these)
7. Click "Create repository"

### 2. Push the Local hmm_tracer Repository to GitHub

```bash
# Add the GitHub repository as a remote
git remote add origin https://github.com/trissim/hmm_tracer.git

# Push your local repository to GitHub
git push -u origin master
```

### 3. Remove the Static alvahmm Package and Add as Submodule

```bash
# Remove the static alvahmm package
git rm -r alva_machinery-0.0.6/
git rm alva_machinery-0.0.6.tar.gz
git commit -m "Remove static alvahmm package"

# Add the forked alvahmm as a Git submodule
git submodule add https://github.com/trissim/alvahmm.git alvahmm
git commit -m "Add forked alvahmm as Git submodule"

# Push the changes to GitHub
git push
```

### 4. Update the README.md with Submodule Instructions

Update the README.md to include instructions for cloning with submodules:

```markdown
## Installation

### Cloning the Repository

To clone this repository with the alvahmm submodule:

```bash
# Clone the repository with submodules
git clone --recurse-submodules https://github.com/trissim/hmm_tracer.git
cd hmm_tracer

# If you already cloned without --recurse-submodules, run:
git submodule update --init --recursive
```

### Installing for Development

To install both packages in development mode:

```bash
# Install alvahmm in development mode
cd alvahmm
pip install -e .
cd ..

# Install hmm_tracer in development mode
pip install -e .
```
```

### 5. Create Branch Protection Rules (Optional)

For collaborative development, consider setting up branch protection rules:

1. Go to the repository settings on GitHub
2. Click on "Branches" in the left sidebar
3. Under "Branch protection rules", click "Add rule"
4. Enter "master" as the branch name pattern
5. Select appropriate protection options:
   - Require pull request reviews before merging
   - Require status checks to pass before merging
   - Require branches to be up to date before merging
6. Click "Create" to save the rule

### 6. Set Up GitHub Actions for CI/CD (Optional)

Create a GitHub Actions workflow file for continuous integration:

```yaml
# .github/workflows/python-tests.yml
name: Python Tests

on:
  push:
    branches: [ master ]
  pull_request:
    branches: [ master ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.7, 3.8, 3.9]

    steps:
    - uses: actions/checkout@v2
      with:
        submodules: recursive
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        cd alvahmm
        pip install -e .
        cd ..
        pip install -e .
        pip install pytest
    - name: Test with pytest
      run: |
        pytest
```

## Validation

To validate the GitHub setup:

1. Clone the repository from GitHub with submodules
2. Verify that both hmm_tracer and alvahmm are present
3. Install both packages in development mode
4. Run the tests to ensure everything works correctly

## Next Steps

After setting up the GitHub repositories:

1. Start developing new features for both projects
2. Use pull requests for code reviews
3. Keep the submodule up to date when making changes to alvahmm
4. Consider setting up automated testing and documentation
