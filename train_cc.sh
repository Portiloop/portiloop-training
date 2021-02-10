#!/bin/bash

#SBATCH --account=def-beltrame
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=32000M
#SBATCH --time=00-12:00

#SBATCH --array=0-20
#SBATCH --output=outputs/%A_%a.out

echo $SLURM_ARRAY_TASK_ID

DATETIME=$(date +"%Y%m%d%H%M%S")

EXPERIMENT_NAME="run_v1_"$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID"_"$DATETIME

echo $EXPERIMENT_NAME

cd $SCRATCH/Portiloop
cp Portiloop.tar $SLURM_TMPDIR/.
cd $SLURM_TMPDIR
tar -xvf Portiloop.tar

module load python/3.8
virtualenv venv
source venv/bin/activate

cd $SLURM_TMPDIR/Portiloop/
pip install .
cd portiloop_detector
python portiloop_detector_training.py --experiment_name=$EXPERIMENT_NAME
