#!/bin/bash
#SBATCH -A research
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=3-00:00:00
#SBATCH --nodelist gnode041
#SBATCH --mail-type=END
#SBATCH -c 10

module add u18/cuda/10.0
module add u18/cudnn/7.6-cuda-10.0
#module load u18/python/3.7.4
#source ~/spk/env/bin/activate

#source /home2/neha.sherin/miniconda3/bin/activate
#conda activate py37

# python3 train.py -p config/DT/preprocess.yaml -m config/DT/model.yaml -t config/DT/train.yaml

python3 train.py -p config/Blizzard/preprocess.yaml -m config/Blizzard/model.yaml -t config/Blizzard/train.yaml
