## Polar Coordinate System Code to solve 2D Diffusion Equation 
Contact: Ishaan Desai (*ishaan.desai@tum.de*)

## Install dependencies
For running this code on MPCDF clusters most of the Python packages can be loaded by loading `anaconda`:
```
module load anaconda/3/2019.03
```
For VTK output the `pyevtk` module needs to be installed manually:
```
pip install --user pyevtk
```

## Compile Cython code
Clone repository and run:
```
python3 setup.py build_ext --inplace
```

## Running diffusion problem:
Create directory for VTK output:
```
mkdir output
```

Run diffusion code on local machine:
```
python3 main.py &
```
Run code on cluster:
```
sbatch run_serial.sh

```

## Configuration
The setup can be configured via the JSON configuration file: `diffusion-coupling-config.json`

## Output and visualization
* Log output is generated in the file `logfile.log`
* VTK output can be found in the `output/` directory 

