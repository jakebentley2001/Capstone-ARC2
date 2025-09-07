#!/bin/bash
#SBATCH -J setup_env
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -o logs/setup-%j.out
#SBATCH -e logs/setup-%j.err

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