#!/bin/bash
#SBATCH --job-name=train_conv_probe
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --gpus=1
#SBATCH --partition=ialab
#SBATCH --exclude=ahsoka,antuco
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pedro.palma@uc.cl
#SBATCH --chdir=/home/pedropalmav/archive/emergent-planning/experiments/vit_bc
#SBATCH --export=ALL


# Run the training script
uv run train_conv_probe.py --feature "tracked_box_next_push_onto_with" --kernel 3 --num_epochs 10

duration=$SECONDS
days=$((duration / 86400))
hours=$(((duration % 86400) / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

echo "Tiempo total de ejecución: ${days}d ${hours}h ${minutes}m ${seconds}s"