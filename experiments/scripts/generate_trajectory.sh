#!/bin/bash
#SBATCH --job-name=drc_trajectory
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=ialab
#SBATCH --nodelist=ventress,llaima
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pedro.palma@uc.cl
#SBATCH --chdir=/home/pedropalmav/archive/emergent-planning/experiments/sokoban_experiments
#SBATCH --export=ALL
#SBATCH --array=0-8

# Initialize pyenv
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

LEVELS_PER_TASK=100000
STARTING_LEVEL=$((SLURM_ARRAY_TASK_ID * LEVELS_PER_TASK))

# Run the training script
pyenv activate drc-planning
python sokoban_experiments/create_trajectory_dataset.py --starting_level $STARTING_LEVEL --num_levels $LEVELS_PER_TASK
pyenv deactivate
