#!/bin/bash
#SBATCH -A research
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=3-00:00:00
#SBATCH --nodelist gnode013
#SBATCH --mail-type=END
#SBATCH -c 20

module add u18/cuda/10.0
module add u18/cudnn/7.6-cuda-10.0
#module load u18/python/3.7.4
#source ~/spk/env/bin/activate

#source /home2/neha.sherin/miniconda3/bin/activate
#conda activate py37

python3 train.py -p config/DailyTalk/preprocess.yaml -m config/DailyTalk/model.yaml -t config/DailyTalk/train.yaml --restore_step 100000
