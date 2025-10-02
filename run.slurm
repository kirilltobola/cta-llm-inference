#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=79G
#SBATCH --time=4-00:15:00
#SBATCH --job-name=qwen-inference

source .env/bin/activate && pip install -r requirements.txt && python3 inference.py 2> error.log
