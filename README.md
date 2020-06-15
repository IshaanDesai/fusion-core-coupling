## Polar Coordinate System 
Contact: Ishaan Desai (*ishaan.desai@tum.de*)

### Compile Cython code
Clone repository and run:
```
python3 setup.py build_ext --inplace
```

### Running diffusion problem:
Run diffusion code:
```
python3 main.py
```
The problem can be configured via the JSON configuration file: `diffusion-coupling-config.json`

### Output and visualization
The magnitude of the field variable is output on the terminal at each output interval.
VTK output can be found in the `output/` directory 

