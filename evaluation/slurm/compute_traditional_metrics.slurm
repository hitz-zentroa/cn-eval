#!/bin/bash
#SBATCH --job-name=
#SBATCH --cpus-per-task=
#SBATCH --gres=
#SBATCH --mem=
#SBATCH --time=

#SBATCH --output=./evaluation/logs/%x_%j.out
#SBATCH --error=./evaluation/logs/%x_%j.err

source ./.venv/bin/activate

corpus=CONAN # other option MT-CONAN

export DATA_FOLD=./data/
export CORPUS=${corpus}
export GEN_FOLD=./models/${corpus}/generation/
export SAVE_FOLD=./evaluation/automatic_evaluation/

srun python ./evaluation/scripts/traditional_metrics.py