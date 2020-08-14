#!/bin/bash

module purge
module load gcc/9 impi/2019.7 cmake petsc-real anaconda
module list

# adapt path to precice
#PRECICE_ROOT=$HOME/precice

# Move to bindings folder
cd /draco/u/idesai/python-bindings

# this assumes that the .so file is in $PRECICE_ROOT/build
python3 setup.py build_ext --include-dirs=$PRECICE_ROOT/src --library-dirs=$PRECICE_ROOT/build --rpath=$PRECICE_ROOT/build

# correct rpath for shared objects
LIBS=$PWD/build/lib*/*.so
for lib in $LIBS; do
  patchelf --set-rpath /usr/lib64:$(patchelf --print-rpath $lib) $lib
done
python3 setup.py install --user
