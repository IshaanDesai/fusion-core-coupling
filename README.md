
# fusion-core-coupling

This code is used to solve equations in a geomtric configuration suitable for modelling the core region of a tokamak fusion reactor. Flux-aligned coordinate systems are valid in the core region. A basic variant of such coordinates is the polar coordinate system which is implemented in this code. This code has the basic capability to handle diverted geometries as well.
The main purpose of this code is to couple the core region of a fusion reactor with a code which models the edge region. Coupling is done using the library [preCICE](https://github.com/precice/precice)[1].

This code was initially developed as part of a [master thesis by Ishaan Desai](https://mediatum.ub.tum.de/604993?query=desai&show_id=1580087)[2] done jointly with the Max Planck Insitute of Plasma Physics and the Chair of Scientific Computing in Computer Science at Technical University of Munich. Current development is done as a joint collaboration between the [Tokamak Theory Division](https://www.ipp.mpg.de/ippcms/eng/for/bereiche/tokamak) at the Max Planck Institute of Plasma Physics and the [Department of Usability and Sustainability of Simulation Software](https://www.ipvs.uni-stuttgart.de/departments/us3/) at the University of Stuttgart.

## Compile Cython code

Only the following branches can be run as stand-alone single physics simulations: [master](https://github.com/IshaanDesai/fusion-core-coupling), [diverted-diffusion](https://github.com/IshaanDesai/fusion-core-coupling/tree/diverted_diffusion). The other branches are developed for coupling purposes and require several dependencies to be installed before being able to use in a coupled simulation.

Clone repository and run:

```[bash]
python3 setup.py build_ext --inplace
```

To compile code on a cluster run `./CompileScript`

## Running diffusion problem

Create directory for VTK output:

```[bash]
mkdir output
```

Run serial code on local machine:

```[bash]
python3 main.py &
```

The `&` marker at the end will send the program into the background of the terminal. Log of the running program can be checked by: `tail -f logfile.log`

A batch script is available if the code is run on a cluster having SLURM system:

```[bash]
sbatch run_serial.sh
```

## Dependencies for MPCDF Cluster Draco

For running this code on MPCDF clusters most of the Python packages can be loaded by loading `anaconda`:

```[bash]
module load anaconda/3/2019.03
```

For VTK output the `pyevtk` module needs to be installed manually:

```[bash]
pip install --user pyevtk
```

## Configuration

The setup can be configured via the JSON configuration file: `diffusion-coupling-config.json`

## Output and visualization

* Log output is generated in the file `logfile.log`
* VTK output can be found in the `output/` directory

## References

[1] *H.-J. Bungartz, F. Lindner, B. Gatzhammer, M. Mehl, K. Scheufele, A. Shukaev, and B. Uekermann: preCICE - A Fully Parallel Library for Multi-Physics Surface Coupling. Computers and Fluids, 141, 250–258, 2016.*  
[2] *Ishaan Desai, Geometric aspects of code coupling in magnetic fusion applications. Masterarbeit, Technical University of Munich, Oct 2020.*
