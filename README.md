## fusion-core-coupling
This code is used to solve equations in a geomtric configuration suitable for modelling the core region of a tokamak fusion reactor. Flux-aligned coordinate systems are valid in the core region. A basic variant of such coordinates is the polar coordinate system which is implemented in this code. This code has the basic capability to handle diverted geometries as well.
The main purpose of this code is to couple the core region of a fusion reactor with a code which models the edge region. Coupling is done using the library [preCICE](https://github.com/precice/precice).
This code is developed as part of a Master thesis by Ishaan Desai done jointly with the Max Planck Insitute of Plasma Physics and the Chair of Scientific Computing at Technical University of Munich.
Contact: Ishaan Desai (*ishaan.desai@tum.de*)

## Compile Cython code
Clone repository and run:
```
python3 setup.py build_ext --inplace
```
To compile code on a cluster run `./CompileScript`

## Running diffusion problem:
Create directory for VTK output:
```
mkdir output
```

Run serial code on local machine:
```
python3 main.py &
```
A batch script is available if the code is run on a cluster having SLURM system:
```
sbatch run_serial.sh
```
The log of the run can be checked by: `tail -f logfile.log`

## Dependencies for MPCDF Cluster Draco
For running this code on MPCDF clusters most of the Python packages can be loaded by loading `anaconda`:
```
module load anaconda/3/2019.03
```
For VTK output the `pyevtk` module needs to be installed manually:
```
pip install --user pyevtk
```

## Configuration
The setup can be configured via the JSON configuration file: `diffusion-coupling-config.json`

## Output and visualization
* Log output is generated in the file `logfile.log`
* VTK output can be found in the `output/` directory 

