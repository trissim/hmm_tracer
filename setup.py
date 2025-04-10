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
    entry_points={
        'console_scripts': [
            'hmm_tracer=hmm_tracer.cli:main',
        ],
    },
)
