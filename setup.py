from setuptools import setup, find_packages

setup(
    name="sliced_spectral_mlp",
    version="0.1.0",
    description="SlicedSpectralMLP: shared-weight MLP on nested Laplacian eigenvector slices",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "torch_geometric>=2.3",
        "scipy>=1.10",
        "numpy>=1.24",
        "matplotlib>=3.7",
        "pyyaml>=6.0",
    ],
)
