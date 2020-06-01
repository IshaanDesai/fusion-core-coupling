from setuptools import setup

# from https://stackoverflow.com/a/9079062
import sys  
if sys.version_info[0] < 3:
    raise Exception("The diffusion_core only supports Python3. Did you run $python setup.py <option>.? Try running $python3 setup.py <option>.")

setup(name='diffusion_core',
      version='0.1',
      description='Diffusion code in polar coodinate system for core region of a plasma fusion reactor ',
      author="Ishaan Desai",
      author_email='ishaan.desai@tum.de',
      packages=['diffusion_core'],
      install_requires=['numpy>=1.13.3'])
