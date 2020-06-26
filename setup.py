import os
from setuptools import setup, find_packages
from Cython.Build import cythonize


def find_pyx(path='.'):
    pyx_files = []
    for root, dirs, filenames in os.walk(path):
        for fname in filenames:
            if fname.endswith('.pyx'):
                pyx_files.append(os.path.join(root, fname))
    return pyx_files


setup(
    name='diffusion_core',
    version='0.1',
    ext_modules=cythonize(find_pyx(), language_level=3),
    install_requires=['Cython'],
    packages=find_packages()
)
