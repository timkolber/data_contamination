#!/bin/bash
#SBATCH --job-name=data_contamination_job
#SBATCH --output=outputs/slurm/data_contamination_job-%j.out
#SBATCH --error=outputs/slurm/data_contamination_job-%j.err
#SBATCH --partition=gpu_h100
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

for ds in "ibragim-bad/arc_challenge" "ibragim-bad/arc_easy" "allenai/openbookqa"

do
    for model in "meta-llama/Meta-Llama-3-8B-Instruct" "mistralai/Mistral-7B-Instruct-v0.3" "microsoft/phi-4"
    do
        srun python main.py \
        --model $model \
        --dataset $ds \
        --mask_amount 2
    done
done