#!/usr/bin/env bash
#SBATCH -J polar_diff
#SBATCH --partition=general
# number of nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000
#SBATCH --time=01:00:00

# Reloading correct modules for FCI-Core:
module load anaconda/3/2019.03

# Run both coupling jobs
echo "Launching Polar Diffusion Code"
srun python3 /draco/u/idesai/fusioncoupling-polar/main.py > log_polardiffusion
echo "Polar Diffusion Run completed"

