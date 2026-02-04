#!/bin/bash
#SBATCH --job-name=create_probe_dataset
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --partition=ialab
# #SBATCH --nodelist=llaima,hydra,ventress
#SBATCH --exclude=ahsoka,antuco
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pedro.palma@uc.cl
#SBATCH --chdir=/home/pedropalmav/archive/emergent-planning/experiments/vit_bc
#SBATCH --export=ALL


# Run the training script
uv run create_probe_dataset.py --num_episodes 3000 --name train
uv run create_probe_dataset.py --num_episodes 1000 --name test --env_name "valid-"

duration=$SECONDS
days=$((duration / 86400))
hours=$(((duration % 86400) / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

echo "Tiempo total de ejecución: ${days}d ${hours}h ${minutes}m ${seconds}s"