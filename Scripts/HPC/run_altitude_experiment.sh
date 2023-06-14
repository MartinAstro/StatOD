#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-100
#SBATCH --time=00:20:00
#SBATCH --partition=amilan
#SBATCH --output=SlurmFiles/Altitude/hparam-search-%j-%a.out

module purge

echo "== Load Anaconda =="

module load anaconda

echo "== Activate Env =="

conda activate research

# Installed package in the compile node already
# echo "== Install =="
# cd /projects/joma5012/GravNN/
# conda develop .

echo "== Run data generation =="

# Note the --exclusive flag ensures that only the one script can be run on the node at a time. 
echo "Running hparams"

 #$SLURM_ARRAY_TASK_ID
srun python /projects/joma5012/StatOD/Scripts/HparamsSearch/hparamsSearchArray.py $SLURM_ARRAY_TASK_ID

wait # Necessary to wait for all processes to finish
echo "== End of Job =="
exit 0