#!/bin/bash
#SBATCH --job-name=data_contamination_job
#SBATCH --output=outputs/slurm/data_contamination_job.out
#SBATCH --error=outputs/slurm/data_contamination_job.err
#SBATCH --partition=gpu_4_h100
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1


srun python model.py