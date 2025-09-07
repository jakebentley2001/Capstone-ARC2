#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --account=cis250063p     # your allocation
#SBATCH --partition=GPU-shared   # shared GPU partition for < full node
#SBATCH --gres=gpu:1             # generic 1 GPU (no type pinning)
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


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