import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths
from .GST import GST, WordRegulator
from .ContextEncoder import ConversationalContextEncoder
from .TextContextEncoder import TextConversationalContextEncoder

class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        #self.gst = GST(model_config)
        self.WE = WordRegulator(model_config)

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )

        self.history_type = model_config["history_encoder"]["type"]
        if self.history_type == "gst-text":
            self.context_encoder = ConversationalContextEncoder(preprocess_config, model_config)
            self.textcontext_encoder = TextConversationalContextEncoder(preprocess_config, model_config)

        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()


    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        words_dur=None,
        wc = None,
        history_info=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        dur_diff=None,
        pitch_diff=None,
        energy_diff=None,

    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        # print('----',texts.shape)
        output = self.encoder(texts, src_masks)


        #if mels !=None:
            # cut = torch.FloatTensor(1).uniform_(0.5, 1)
            #if cut>0.75:
            #    style_embedding = self.gst(mels[:,:int(cut*mels.shape[1]),:])
            #    print('style_embedding',cut, style_embedding)
            #else:
            #    style_embedding = self.gst(mels[:,-int(cut*mels.shape[1]):,:])
            #    print('style_embedding',cut, style_embedding)
            #style_embedding = self.gst(mels) #[:,:int(0.8*mels.shape[1]),:])
            #print('style embd shape', style_embedding.shape)
            #print('style_embedding', style_embedding)
            
            
            #output = output+style_embedding.expand(-1, max_src_len, -1)


        # context encoding / adding history 
        context_encoding = None
        textcontext_encoding = None
        if self.history_type == "gst-text":
            (
                gst_embs,
                history_lens,
                history_gst_embs,
                history_speakers,
                text_embs, history_text_embs
            ) = history_info

            context_encoding = self.context_encoder(
                gst_embs,
                speakers,
                history_gst_embs,
                history_speakers,
                history_lens,
                )

            textcontext_encoding = self.textcontext_encoder(
                text_embs,
                speakers,
                history_text_embs,
                history_speakers,
                history_lens,
                )


        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )
        # print('after se',output.shape)
        # we = self.WE(mels, output, words_dur,wc)

        if context_encoding is not None:
            output = output + context_encoding.unsqueeze(1).expand(
                -1, max_src_len, -1
            )
        if textcontext_encoding is not None:
            output = output + textcontext_encoding.unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        # if context_encoding is not None:
        #    x = x + context_encoding.unsqueeze(1).expand(
        #        -1, text.shape[1], -1
        #    )



        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
            dur_diff,pitch_diff,energy_diff
        )


        # output = output+we
        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )
