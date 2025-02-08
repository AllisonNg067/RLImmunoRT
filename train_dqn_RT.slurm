#!/bin/bash -l
#SBATCH --job-name=GPUtrainDQN
#SBATCH --partition=gpu-highmem
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --account=pawsey1166-gpu
#SBATCH --time=24:00:00
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
export OMP_NUM_THREADS=1
python3 train_dqn_RT.py
