import torch.nn as nn
import torch
from model import FastSpeech2
import argparse
import yaml
from utils.model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument("--restore_step", type=int, default=0)
parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )

args = parser.parse_args()

    # Read Config
preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
configs = (preprocess_config, model_config, train_config)



model, optimizer = get_model(args, configs, device, train=True)

model = nn.DataParallel(model)

for nm, pm in model.named_parameters():
    print(nm, pm.shape, pm.requires_grad)
    #if not 'decoder' in nm:
    #    pm.requires_grad = False
#for nm, pm in model.named_parameters():
#    print(nm, pm.requires_grad)
