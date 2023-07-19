#!/bin/bash
#SBATCH --job-name="gnn_source_detect"
#SBATCH --gres=gpu:4
#SBATCH --exclude=ac922-[01-02]
#SBATCH --mem=100G
#SBATCH --cpus-per-gpu=2
#SBATCH --account=renard
#SBATCH --partition=sorcery
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/hpi/fs00/home/conrad.halle/source_detection
#SBATCH --mail-type=ALL
#SBATCH --mail-user=conrad.halle@student.hpi.de
#SBATCH --verbose
#SBATCH --output=/hpi/fs00/home/conrad.halle/source_detection/slurm_training/slurm_%j_%N_%x_%A_%a.out
#SBATCH --error=/hpi/fs00/home/conrad.halle/source_detection/slurm_training/slurm_%j_%N_%x_%A_%a.err

source venv/bin/activate
echo "START TRAINING"

# python -m src.training
# python -m src.validation --dataset=synthetic
# python -m src.validation --dataset=karate
# python -m src.validation --dataset=airports
# python -m src.validation --dataset=facebook
# python -m src.validation --dataset=wiki
# python -m src.validation --dataset=actor
# python -m src.training
# python -m src.validation --dataset=synthetic
# python -m src.validation --dataset=karate
# python -m src.validation --dataset=airports
# python -m src.validation --dataset=facebook
# python -m src.validation --dataset=wiki
# python -m src.validation --dataset=actor

dvc repro training
dvc repro validation@synthetic
dvc repro validation