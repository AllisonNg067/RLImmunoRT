#!/bin/bash -l
#SBATCH --job-name=varyIORatio
#SBATCH --nodes=1
#SBATCH --tasks-per-node=10
#SBATCH --cpus-per-task=10
#SBATCH --time=01:00:00
#SBATCH --mail-user=23303755@student.uwa.edu.au
#SBATCH --mail-type=ALL
# load modules
module load rocm
# Unload conflicting modules
module unload python

# Load necessary modules
module load tensorflow/rocm5.6-tf2.12
#python3 -m pip install --user matplotlib numpy pandas scikit-learn
# launch serial python script

python3 test_network.py
