#!/bin/bash
#SBATCH --job-name="gnn_source_detect"
#SBATCH --exclude=ac922-[01-02]
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --cpus-per-gpu=2
#SBATCH --account=renard
#SBATCH --partition=sorcery
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/hpi/fs00/home/conrad.halle/source_detection
#SBATCH --mail-type=ALL
#SBATCH --mail-user=conrad.halle@student.hpi.de
#SBATCH --verbose
#SBATCH --output=/hpi/fs00/home/conrad.halle/source_detection/sbatch_out/slurm_%j_%N_%x_%A_%a.out
#SBATCH --error=/hpi/fs00/home/conrad.halle/source_detection/sbatch_out/slurm_%j_%N_%x_%A_%a.err

conda deactivate

source venv/bin/activate

python update_params.py --idx ${SLURM_ARRAY_TASK_ID}

dvc repro
