import torch
import numpy as np
import torch.nn as nn
import os
import json
from utils.tools import get_mask_from_lengths


class ConversationalContextEncoder(nn.Module):
    """ Conversational Context Encoder """

    def __init__(self, preprocess_config, model_config):
        super(ConversationalContextEncoder, self).__init__()
        d_model = model_config["transformer"]["encoder_hidden"]
        d_cont_enc = model_config["history_encoder"]["context_hidden"]
        num_layers = model_config["history_encoder"]["context_layer"]
        dropout = model_config["history_encoder"]["context_dropout"]
        self.text_emb_size = model_config["history_encoder"]["gst_emb_size"]
        self.max_history_len = model_config["history_encoder"]["max_history_len"]

        self.text_emb_linear = nn.Linear(self.text_emb_size, d_cont_enc)
        self.speaker_linear = nn.Linear(d_model, d_cont_enc)
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            ),
            "r",
        ) as f:
            n_speaker = len(json.load(f))
        self.speaker_embedding = nn.Embedding(
            n_speaker,
            model_config["transformer"]["encoder_hidden"],
        )

        self.enc_linear = nn.Sequential(
            nn.Linear(2*d_cont_enc, d_cont_enc),
            nn.ReLU()
        )
        self.gru = nn.GRU(
            input_size=d_cont_enc,
            hidden_size=d_cont_enc,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.gru_linear = nn.Sequential(
            nn.Linear(2*d_cont_enc, d_cont_enc),
            nn.ReLU()
        )

        self.context_linear = nn.Linear(d_cont_enc, d_model)
        self.context_attention = SLA(d_model)

    def forward(self, text_emb, speaker, history_text_emb, history_speaker, history_lens):

        history_masks = get_mask_from_lengths(history_lens, self.max_history_len)

        # Embedding
        history_text_emb = torch.cat([history_text_emb, text_emb.unsqueeze(1)], dim=1)
        history_text_emb = self.text_emb_linear(history_text_emb)
        history_speaker = torch.cat([history_speaker, speaker.unsqueeze(1)], dim=1)
        history_speaker = self.speaker_linear(self.speaker_embedding(history_speaker))

        history_enc = torch.cat([history_text_emb, history_speaker], dim=-1)
        history_enc = self.enc_linear(history_enc)

        # Split
        enc_current, enc_past = torch.split(history_enc, self.max_history_len, dim=1)

        # GRU
        enc_current = self.gru_linear(self.gru(enc_current)[0])
        enc_current = enc_current.masked_fill(history_masks.unsqueeze(-1), 0)

        # Encoding
        context_enc = torch.cat([enc_current, enc_past], dim=1)
        context_enc = self.context_attention(self.context_linear(context_enc)) # [B, d]

        return context_enc


class SLA(nn.Module):
    """ Sequence Level Attention """

    def __init__(self, d_enc):
        super(SLA, self).__init__()
        self.linear = nn.Linear(d_enc, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoding, mask=None):

        attn = self.linear(encoding)
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(-1), -np.inf)
            aux_mask = (attn == -np.inf).all(self.softmax.dim).unsqueeze(self.softmax.dim)
            attn = attn.masked_fill(aux_mask, 0) # Remove all -inf along softmax.dim
        score = self.softmax(attn).transpose(-2, -1) # [B, 1, T]
        fused_rep = torch.matmul(score, encoding).squeeze(1) # [B, d]

        return fused_rep
