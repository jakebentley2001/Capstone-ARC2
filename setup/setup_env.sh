#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:A100_40GB:1
#SBATCH --partition=general

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