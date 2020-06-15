from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("diffusion_core/diffusion_core.py"), install_requires=['Cython']
)
