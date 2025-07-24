#!/bin/bash

#SBATCH --account=def-beltrame
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=48000M
#SBATCH --job-name=portiloop_training
#SBATCH --output=sbatch_out/job_output.log
#SBATCH --error=sbatch_out/job_error.log
#SBATCH --time=1-00:00:00

set -e
echo "Loading environment"
source ~/.bashrc
source .env/bin/activate || { echo "Failed to activate virtualenv"; exit 1; }

echo "Virtual environment activated"
which python
nvidia-smi                       # ✅ Print GPU info on the compute node

echo "Starting training"
.env/bin/python3 portiloopml/portiloop_python/ANN/portiloop_detector_training.py > sbatch_out/training.log 2>&1

echo "End of training"

