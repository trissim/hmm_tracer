#!/bin/bash
# install_dev.sh
# Script to install both alvahmm and hmm_tracer in development mode

echo "Installing alvahmm and hmm_tracer in development mode..."

# Install alvahmm in development mode
cd alvahmm
pip install -e .
cd ..

# Install hmm_tracer in development mode
pip install -e .

echo "Both packages installed in development mode"
echo "You can now import alvahmm and hmm_tracer in your Python code"
