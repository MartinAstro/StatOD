
conda activate research

python /projects/joma5012/StatOD/Scripts/DataGeneration/Networks/train_poly_model.py
python /projects/joma5012/StatOD/Scripts/DataGeneration/Networks/train_pm_model.py
python /projects/joma5012/StatOD/Scripts/DataGeneration/Trajectories/generate_asteroid_rotating_traj.py
python /projects/joma5012/StatOD/Scripts/Scenarios/DMC_high_fidelity.py