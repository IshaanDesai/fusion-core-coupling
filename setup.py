from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("diffusion_core.pyx"), install_requires=['Cython']
)
