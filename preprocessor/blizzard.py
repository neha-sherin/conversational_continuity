import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    speaker = "Blizzard"

    for chapter in tqdm(os.listdir(os.path.join(in_dir, "transcripts"))):
        for file_name in os.listdir(os.path.join(in_dir, "transcripts", chapter)):
            wav_path = os.path.join(in_dir,chapter,file_name.replace(".txt",'.wav'))
            text_path = os.path.join(in_dir,'transcripts',chapter,file_name)
            base_name = file_name[:-4]
            if os.path.exists(wav_path):
                with open(text_path) as f:
                    text = f.readline().strip("\n")
                text = _clean_text(text, cleaners)
                os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                wav, _ = librosa.load(wav_path, sampling_rate)
                wav = wav / max(abs(wav)) * max_wav_value
                wavfile.write(
                    os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
                    sampling_rate,
                    wav.astype(np.int16),
                )
                with open(
                        os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
                        "w",
                ) as f1:
                    f1.write(text)
