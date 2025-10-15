#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80G
#SBATCH --time=7-00:30:00
#SBATCH --job-name=qwen-i

source .env/bin/activate && pip install -r requirements.txt && python3 inference.py 2> error.log
