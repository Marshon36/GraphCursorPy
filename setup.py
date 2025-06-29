# setup.py
from setuptools import setup, find_packages

setup(
    name="GraphCursorPy",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'cartopy',
        'obspy',
        'torch',
        'torch_geometric',
    ],
)