import torch.nn as nn
import os
import numpy as np
import argparse
import yaml
import torch
from model.GST import GST #, WordRegulator
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class gst_model(nn.Module):

    def __init__(self, model_config): # , preprocess_config, model_config):
        super(gst_model, self).__init__()
        self.model_config = model_config

        self.gst = GST(model_config)

    def forward(self, mels=None,):

        style_embedding = self.gst(mels) #[:,:int(0.8*mels.shape[1]),:])
        print('style embd shape', style_embedding.shape)
        return style_embedding


model_config = yaml.load(open('config/DT/model.yaml', "r"), Loader=yaml.FullLoader)
model = gst_model(model_config).to(device)

model.load_state_dict(torch.load('/ssd_scratch/cvit/neha/output/ckpt/Bli_DT/800000.pth.tar'), strict=False)

model.eval()
model.requires_grad_ = False

#for nm, pm in model.named_parameters():
#    print(nm, pm.requires_grad)
#    pm.requires_grad = False

parser = argparse.ArgumentParser()
parser.add_argument(
        "-m",
        "--mel",
        type=str,
        default='refs/1-mel-0_1_d2435.npy',
        help="path to mel .npy file",
    )

args = parser.parse_args()
mel = args.mel   # 1-mel-0_1_d2435.npy


path = '/ssd_scratch/cvit/neha/preprocessed_data_gst/gst_emb'
melpath = '/ssd_scratch/cvit/neha/preprocessed_data_gst/mel'

for mel in os.listdir(melpath):
    mel = os.path.join(melpath,mel)
    file_id = (mel.split('/')[-1]).split('-')[-1]
    print(file_id)

    mels = torch.tensor(np.array([[np.load(mel)]])).to(device)
    gst_embd = model(mels)
    print(gst_embd[0][0].shape)
    np.save(os.path.join(path, file_id),gst_embd[0][0].cpu().detach().numpy())
