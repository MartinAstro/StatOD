#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=atesting_a100
#SBATCH --output=SlurmFiles/train-network-%j-%a.out

module purge

echo "== Load Anaconda =="

module load anaconda
module load cudnn/8.6
export PATH=$PATH:/curc/sw/cuda/11.8/bin:/curc/sw/cuda/11.8/lib64
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/curc/sw/cuda/11.8/


echo "== Activate Env =="

conda activate research

echo "== Run data generation =="

echo "Running orbit search"

srun python /projects/joma5012/StatOD/Scripts/DataGeneration/Networks/train_pm_model.py
srun python /projects/joma5012/StatOD/Scripts/DataGeneration/Networks/train_poly_model.py


wait # Necessary to wait for all processes to finish
echo "== End of Job =="
exit 0