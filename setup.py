#!/usr/bin/python
"""Build script for the Cython confusion-matrix extension.

Compiles ``addToConfusionMatrix.pyx`` (which embeds
``addToConfusionMatrix_impl.c``) into a shared library that can be imported
by :mod:`evaluate` for fast C++-based confusion-matrix accumulation.

Build
-----
::

    pip install setuptools cython numpy
    python setup.py build_ext --inplace

The resulting ``.so`` / ``.pyd`` file is placed next to the source files so
that ``import addToConfusionMatrix`` works without any path manipulation.

Warning
-------
Only tested on Ubuntu 64-bit with g++/g++ as the compiler.
"""

try:
	from setuptools import setup
	from Cython.Build import cythonize
except ImportError:
	print('Unable to setup. Please install the required packages:')
	print('    pip install setuptools cython')
	raise

import os
import numpy

# Force both the C and C++ compiler to g++ so that the C++ features used in
# addToConfusionMatrix_impl.c (e.g. C99 variable-length for-loop counters)
# are accepted without warnings.
os.environ['CC'] = 'g++'
os.environ['CXX'] = 'g++'

setup(
	ext_modules=cythonize('addToConfusionMatrix.pyx'),
	include_dirs=[numpy.get_include()],
)
