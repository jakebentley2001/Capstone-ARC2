#!/bin/bash
#SBATCH -p setup_env
#SBATCH -A cis250063p
#SBATCH --gpus=A100-40:1     # Request 1 GPU (adjust as needed)
#SBATCH --time=1:00:00     # Request 3 hours of runtime (adjust as needed)
#SBATCH --ntasks=1         # Run a single task
#SBATCH --cpus-per-task=4  # Request 8 CPU cores per task (adjust as needed)
#SBATCH --mem=16G 
#SBATCH --partition=GPU


#Load conda
module load anaconda3

#Create env
conda create -n llm_sft python=3.10 -y
conda activate llm_sft

#Install Pytorch
pip install torch==2.3.1 --extra-index-url https://download.pytorch.org/whl/cu121

# Hugging Face Ecosystem
pip install transformers accelerate datasets peft trl

# Unsloth (builds on GPU node)
pip install unsloth

#optional
pip install wandb