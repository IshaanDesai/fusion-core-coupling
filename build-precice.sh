#!/bin/bash

module purge
module load gcc/9 impi/2019.7 cmake petsc-real anaconda
module list

# adapt to eigen path
export Eigen3_ROOT=$HOME/eigen-3.3.7

cd $PRECICE_ROOT
rm -rf build
mkdir -p build && cd build
cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=$PWD/installed -DPRECICE_PETScMapping=OFF \
  -DPRECICE_PythonActions=OFF -DMPI_CXX_COMPILER=mpigcc -DPYTHON_EXECUTABLE=$(which python) \
  -DBoost_NO_BOOST_CMAKE=TRUE -DBoost_NO_SYSTEM_PATHS=TRUE -DBOOST_ROOT:PATHNAME=/u/idesai/boost_install -DBoost_LIBRARY_DIRS:FILEPATH=/u/idesai/boost_install/lib ..
make -j20

# correct RPATH in shared object
LIB=$PWD/libprecice.so
#echo Current RPATH: $(patchelf --print-rpath $LIB)
patchelf --set-rpath /usr/lib64:$(patchelf --print-rpath $LIB) $LIB
#echo RPATH after patching: $(patchelf --print-rpath $LIB)

make test_base
make install
