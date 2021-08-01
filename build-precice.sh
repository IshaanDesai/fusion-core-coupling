#!/bin/bash

module purge
module load gcc/9 impi/2019.7 cmake/3.18 petsc-real boost/1.74
module list

rm -rf build
mkdir -p build && cd build
cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=$PWD/installed -DMPI_CXX_COMPILER=mpigcc \
   -DCMAKE_BUILD_TYPE=Debug -DPRECICE_PythonActions=OFF ..
make -j20

#make test_base
#make install

