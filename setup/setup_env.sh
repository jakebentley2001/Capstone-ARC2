#!/bin/bash
#SBATCH --job-name=upload_model
#SBATCH --output=./logs/upload_model_%j.out
#SBATCH --error=./logs/upload_model_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
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