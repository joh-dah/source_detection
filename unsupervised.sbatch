#!/bin/bash
#SBATCH --job-name="unsupervised"
#SBATCH --mem=256G
#SBATCH -n 101
#SBATCH --exclude=ic922-[01-02]
#SBATCH --cpus-per-task=1
#SBATCH --account=renard
#SBATCH --partition=magic
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/hpi/fs00/home/conrad.halle/source_detection
#SBATCH --mail-type=ALL
#SBATCH --mail-user=conrad.halle@student.hpi.de
#SBATCH --verbose
#SBATCH --output=/hpi/fs00/home/conrad.halle/source_detection/slurm_unsup/slurm_%j_%N_%x_%A_%a.out
#SBATCH --error=/hpi/fs00/home/conrad.halle/source_detection/slurm_unsup/slurm_%j_%N_%x_%A_%a.err

source venv/bin/activate
dvc repro unsupervised@synthetic