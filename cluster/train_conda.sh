#!/bin/bash

#SBATCH --job-name=SunSkelTest
#SBATCH --partition=haicore-gpu4
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=152
#SBATCH --time=20:00:00
#SBATCH --output=/hkfs/work/workspace/scratch/im9193-H1/SunSkelTest.txt

export CUDA_CACHE_DISABLE=1
export OMP_NUM_THREADS=8

group_workspace=/hkfs/work/workspace/scratch/im9193-H1

source /hkfs/work/workspace/scratch/im9193-conda/conda/etc/profile.d/conda.sh
conda activate ${group_workspace}/health_baseline_conda_env
python ${group_workspace}/hackathon_health/scripts/train.py "$@"
