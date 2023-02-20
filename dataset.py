import json
import math
import os

import numpy as np
from torch.utils.data import Dataset

from text import text_to_sequence
from utils.tools import pad_1D, pad_2D


neglect_files = ['CB-EM-29-201', 'CB-BBH-01-516', 'CB-LCL-12-505', 'CB-JB2-07-587', 'CB-FFM-53-316', 'CB-FFM-53-164', 'CB-JB-06-65', 'CB-JB2-03-17', 'CB-AW-05-12', 'CB-AW-30-55', 'CB-AW-30-17', 'CB-DM-03-04', 'CB-FFM-14-51', 'CB-EM-44-94', 'CB-FFM-53-311', 'CB-LCL-11-43', 'CB-LCL-05-258', 'CB-FFM-52-216', 'CB-JB2-03-48', 'CB-EM-38-265', 'CB-FFM-17-35', 'CB-JE-04-240', 'CB-EM-43-07', 'CB-FFM-53-314', 'CB-JE-08-87', 'CB-DM-01-515', 'CB-EM-44-140', 'CB-FFM-43-70', 'CB-EM-14-72', 'CB-FFM-49-55', 'CB-LCL-10-1356', 'CB-FFM-32-94', 'CB-FFM-53-146', 'CB-FFM-39-70', 'CB-LCL-16-119', 'CB-EM-36-57', 'CB-LCL-10-158', 'CB-JB2-04-301', 'CB-JE-15-173', 'CB-LCL-11-29', 'CB-FFM-32-86', 'CB-JE-22-69', 'CB-LCL-02-03', 'CB-FFM-06-34', 'CB-EM-26-101', 'CB-FFM-53-250', 'CB-JE-25-08', 'CB-FFM-31-24', 'CB-BBH-01-523', 'CB-EM-38-28', 'CB-FFM-53-257', 'CB-FFM-34-207', 'CB-AW-30-06', 'CB-AW-21-60', 'CB-JE-26-203', 'CB-AW-07-151', 'CB-JE-28-10', 'CB-FFM-53-57', 'CB-JB2-03-112', 'CB-AW-33-190', 'CB-JE-12-02', 'CB-FFM-33-94', 'CB-EM-24-195', 'CB-FFM-34-203', 'CB-EM-36-02', 'CB-EM-19-103', 'CB-JB2-03-70', 'CB-FFM-14-47', 'CB-JB2-06-36', 'CB-LCL-10-817', 'CB-JE-22-149', 'CB-JB2-07-299', 'CB-JB-05-147', 'CB-LCL-15-681', 'CB-EM-37-50', 'CB-LCL-10-851', 'CB-JB-07-316', 'CB-FFM-08-212', 'CB-CHE-12-158', 'CB-LCL-05-299', 'CB-AW-17-25', 'CB-EM-28-174', 'CB-FFM-54-08', 'CB-LCL-16-425', 'CB-LCL-18-58', 'CB-JB-04-353', 'CB-FFM-19-125', 'CB-EM-42-04', 'CB-FFM-22-112', 'CB-JB2-06-16', 'CB-FFM-20-09', 'CB-EM-33-23', 'CB-EM-11-18', 'CB-AW-03-68', 'CB-FFM-19-30', 'CB-EM-32-229', 'CB-LCL-10-125', 'CB-BBH-01-141', 'CB-AW-05-67', 'CB-EF-01-66', 'CB-FFM-14-54', 'CB-20K1-07-48', 'CB-EM-26-173', 'CB-EM-41-33', 'CB-AW-05-81', 'CB-FFM-32-147', 'CB-LCL-09-138', 'CB-CHE-11-201', 'CB-FFM-15-72', 'CB-DM-02-172', 'CB-LCL-10-733', 'CB-LCL-10-1147', 'CB-EM-52-48', 'CB-JE-11-394', 'CB-JB-04-115', 'CB-LCL-17-435', 'CB-EM-04-63', 'CB-JE-04-03', 'CB-ALP-15-56', 'CB-EM-33-119', 'CB-DM-02-394', 'CB-JB2-06-250', 'CB-EM-05-106', 'CB-JE-13-288', 'CB-20K1-10-289', 'CB-LCL-09-154', 'CB-CHE-02-285', 'CB-FFM-32-47', 'CB-EM-28-129', 'CB-LCL-19-162', 'CB-EM-26-59', 'CB-FFM-38-77', 'CB-CHE-11-605', 'CB-EM-28-128', 'CB-EM-30-48', 'CB-AW-20-53', 'CB-CHE-11-719', 'CB-JB-06-64', 'CB-FFM-53-82', 'CB-FFM-53-203', 'CB-LCL-14-667', 'CB-JE-22-54', 'CB-JE-18-11', 'CB-LCL-16-257', 'CB-EM-45-50', 'CB-FFM-19-23', 'CB-EM-36-51', 'CB-EM-03-14', 'CB-FFM-08-96', 'CB-EM-04-29', 'CB-EM-27-69', 'CB-JE-13-128', 'CB-AW-05-46', 'CB-LCL-10-399', 'CB-JB-04-151', 'CB-LCL-17-505', 'CB-LCL-16-240', 'CB-FFM-31-11', 'CB-FFM-22-71', 'CB-EM-15-44', 'CB-LCL-10-813', 'CB-JE-13-62', 'CB-FFM-50-221', 'CB-FFM-22-108', 'CB-EM-42-174', 'CB-BBH-01-135', 'CB-LCL-07-497', 'CB-JB2-06-136', 'CB-EM-02-53', 'CB-FFM-49-09', 'CB-FFM-49-18', 'CB-JB2-06-77', 'CB-LCL-13-475', 'CB-JB-05-261', 'CB-FRA-01-145', 'CB-FFM-23-47', 'CB-CHE-11-696', 'CB-DM-03-07', 'CB-AW-25-41', 'CB-JE-24-14', 'CB-AW-03-29', 'CB-CHE-11-393', 'CB-FFM-34-17', 'CB-EM-48-104', 'CB-EM-54-171', 'CB-LCL-07-470', 'CB-LCL-11-219', 'CB-JB2-05-288', 'CB-EM-34-14', 'CB-JB-05-371', 'CB-EM-34-199', 'CB-FFM-17-25', 'CB-EM-42-221', 'CB-JB2-06-78', 'CB-LCL-09-39', 'CB-EM-53-148', 'CB-JB-02-421', 'CB-JE-04-120', 'CB-EM-04-56', 'CB-AW-04-29', 'CB-EM-03-27', 'CB-FFM-13-82', 'CB-FFM-23-73', 'CB-EM-34-116', 'CB-JE-21-345', 'CB-ALP-06-235', 'CB-FFM-19-45', 'CB-EM-04-19', 'CB-AW-37-30', 'CB-FFM-17-28', 'CB-LCL-10-1010', 'CB-LCL-11-530', 'CB-EM-15-07', 'CB-LCL-04-139', 'CB-EM-38-114', 'CB-AW-03-25', 'CB-AW-12-49', 'CB-EM-48-48', 'CB-AW-05-70', 'CB-LCL-11-79', 'CB-JB2-07-284', 'CB-CHE-02-264', 'CB-JE-24-518', 'CB-AW-17-62', 'CB-AW-03-73', 'CB-JE-17-02', 'CB-AW-17-16', 'CB-EM-06-82', 'CB-BBH-01-581', 'CB-EM-38-66', 'CB-EM-42-216', 'CB-FRA-05-54', 'CB-ALP-07-535', 'CB-EM-34-53', 'CB-FFM-14-34', 'CB-LCL-11-278', 'CB-JB2-07-186', 'CB-FFM-09-58', 'CB-20K2-01-50', 'CB-LCL-18-662', 'CB-EM-36-120', 'CB-EM-48-66', 'CB-FFM-14-57', 'CB-FFM-38-71', 'CB-EM-30-40', 'CB-EM-46-131', 'CB-JE-21-306', 'CB-AW-26-74', 'CB-LCL-11-303', 'CB-LCL-10-1033', 'CB-EM-42-281', 'CB-LCL-14-193', 'CB-EM-28-32', 'CB-DM-02-156', 'CB-EM-38-98', 'CB-JB-02-194', 'CB-EM-38-273', 'CB-FFM-52-252', 'CB-DM-02-298', 'CB-JE-28-100', 'CB-JB2-06-423', 'CB-JB-02-202', 'CB-CHE-05-86', 'CB-EM-19-13', 'CB-JB2-05-158', 'CB-EM-26-429', 'CB-LCL-19-47', 'CB-FFM-54-46', 'CB-FFM-15-64', 'CB-LCL-10-741', 'CB-EM-37-81', 'CB-FFM-25-48', 'CB-FFM-22-78', 'CB-EM-51-90', 'CB-JB-04-253', 'CB-DM-01-94', 'CB-EM-38-219', 'CB-EM-05-130', 'CB-LCL-09-284', 'CB-LCL-16-09', 'CB-EM-18-05', 'CB-FFM-35-88', 'CB-EM-36-61', 'CB-AW-07-36', 'CB-FFM-17-12', 'CB-EM-45-66', 'CB-AW-15-64', 'CB-LCL-13-13', 'CB-EM-44-109', 'CB-EM-25-12', 'CB-LCL-12-507', 'CB-FFM-34-283', 'CB-DM-02-58', 'CB-FRA-18-131', 'CB-LCL-14-679', 'CB-JB-07-361', 'CB-FFM-53-63', 'CB-JE-04-223', 'CB-DM-03-504', 'CB-FFM-42-113', 'CB-FFM-36-106', 'CB-DM-01-08', 'CB-FFM-22-171', 'CB-LCL-19-61', 'CB-FFM-32-186', 'CB-FFM-35-12', 'CB-FFM-21-22', 'CB-JE-17-309', 'CB-JE-14-125', 'CB-FFM-53-235', 'CB-FFM-32-114', 'CB-JE-15-260', 'CB-CHE-11-557', 'CB-FFM-19-32', 'CB-FFM-31-198', 'CB-FFM-15-214', 'CB-FFM-50-232', 'CB-FFM-34-112', 'CB-AW-30-81', 'CB-AW-25-12', 'CB-JE-04-221', 'CB-JE-11-404', 'CB-EM-46-138', 'CB-JB-02-258', 'CB-EM-42-153', 'CB-JE-08-129', 'CB-FFM-42-136', 'CB-FFM-53-65', 'CB-AW-11-51', 'CB-FFM-02-03', 'CB-CHE-01-134', 'CB-CHE-02-301', 'CB-LCL-14-663', 'CB-JB2-08-299', 'CB-LCL-19-130', 'CB-EM-02-07', 'CB-FFM-34-30', 'CB-JB2-01-317', 'CB-JB2-07-104', 'CB-JE-17-253', 'CB-EM-03-75', 'CB-EM-55-54', 'CB-FFM-17-07', 'CB-JB2-07-283', 'CB-AW-13-58', 'CB-JE-17-416', 'CB-FFM-34-23', 'CB-EM-14-84', 'CB-AW-04-54', 'CB-EM-26-68', 'CB-FFM-32-66', 'CB-AW-23-62', 'CB-JB-02-80', 'CB-JE-04-16', 'CB-LCL-02-08', 'CB-FFM-34-53', 'CB-FFM-49-19', 'CB-EM-29-89', 'CB-EM-35-146', 'CB-LCL-02-176', 'CB-LCL-05-204', 'CB-JB2-06-80', 'CB-FFM-01-66', 'CB-LCL-13-266', 'CB-DM-02-143', 'CB-LCL-08-419', 'CB-AW-36-60', 'CB-FFM-23-36', 'CB-AW-02-32', 'CB-LCL-10-1020', 'CB-JB-04-176', 'CB-AW-03-15', 'CB-AW-25-37', 'CB-FFM-31-09', 'CB-EM-52-12', 'CB-FFM-53-313', 'CB-FFM-31-42', 'CB-LCL-15-28', 'CB-JB-06-75', 'CB-CHE-05-02', 'CB-LCL-16-178', 'CB-LCL-13-06', 'CB-EM-41-78', 'CB-JE-15-116', 'CB-FFM-57-60', 'CB-LCL-13-420', 'CB-EM-43-80', 'CB-LCL-11-17', 'CB-AW-30-127', 'CB-FFM-21-21', 'CB-FFM-32-19', 'CB-JB-07-391', 'CB-LCL-19-215', 'CB-FFM-49-87', 'CB-DM-02-181', 'CB-DM-03-453', 'CB-EM-27-77', 'CB-EM-46-82', 'CB-JE-04-27', 'CB-EM-29-151', 'CB-JE-10-160', 'CB-EM-19-132', 'CB-EM-28-02', 'CB-LCL-09-31', 'CB-JB-07-208', 'CB-FFM-57-41', 'CB-EM-27-147', 'CB-FFM-22-107', 'CB-LCL-10-762', 'CB-JB2-06-413', 'CB-EM-35-04', 'CB-FFM-36-137', 'CB-FFM-38-30', 'CB-LCL-10-800', 'CB-EM-03-28', 'CB-EM-29-199', 'CB-AW-02-24', 'CB-EM-29-33', 'CB-EM-48-123', 'CB-JE-04-105', 'CB-LCL-18-531', 'CB-EM-42-232', 'CB-FFM-34-43', 'CB-DM-02-69', 'CB-EM-41-41', 'CB-LCL-17-100', 'CB-CHE-10-159', 'CB-FFM-51-31', 'CB-EM-38-181', 'CB-FFM-34-309', 'CB-FFM-55-35', 'CB-FFM-34-239', 'CB-FFM-20-28', 'CB-JE-22-127', 'CB-JE-16-68', 'CB-JE-11-84', 'CB-JB-01-54', 'CB-LCL-13-470', 'CB-FFM-32-122', 'CB-EM-32-69', 'CB-JB2-02-203', 'CB-LCL-19-385', 'CB-EM-29-52', 'CB-LCL-15-369', 'CB-FFM-22-181', 'CB-BBH-01-164', 'CB-FFM-31-46', 'CB-JE-17-518', 'CB-FFM-34-274', 'CB-FFM-15-223', 'CB-FFM-34-272', 'CB-EM-38-298', 'CB-EM-44-126', 'CB-JB-02-551', 'CB-AW-04-41', 'CB-EM-53-162', 'CB-LCL-07-367', 'CB-FFM-52-74', 'CB-JE-16-35', 'CB-LCL-10-388', 'CB-FFM-50-90', 'CB-FFM-23-22', 'CB-FFM-52-111', 'CB-EM-27-82', 'CB-JB2-05-291', 'CB-FRA-11-23', 'CB-JB2-07-110', 'CB-CHE-01-123', 'CB-JB-04-08', 'CB-LCL-19-87', 'CB-DM-02-145', 'CB-FFM-54-07', 'CB-EM-23-168', 'CB-JB2-06-407', 'CB-EM-46-04', 'CB-ALP-18-19', 'CB-FFM-34-49', 'CB-LCL-11-75', 'CB-EM-36-161', 'CB-FFM-52-204', 'CB-JE-02-131', 'CB-LCL-08-352', 'CB-AW-13-108', 'CB-LCL-19-193', 'CB-JB2-06-391', 'CB-AW-25-33', 'CB-DM-01-15', 'CB-JE-33-91', 'CB-FFM-18-37', 'CB-LCL-11-317', 'CB-LCL-09-353', 'CB-JE-11-310', 'CB-EM-26-56', 'CB-JB-07-328', 'CB-FFM-15-184', 'CB-AW-08-112', 'CB-LCL-19-213', 'CB-LCL-17-344', 'CB-JB2-06-01', 'CB-AW-17-68', 'CB-DM-02-338', 'CB-EM-13-156', 'CB-FFM-52-241', 'CB-DM-03-05', 'CB-FFM-34-81', 'CB-JB2-05-146', 'CB-JB-07-485', 'CB-JE-25-174', 'CB-FFM-51-08', 'CB-EM-33-174', 'CB-FFM-17-15', 'CB-EM-46-165', 'CB-CHE-11-553', 'CB-EM-41-74', 'CB-AW-05-48', 'CB-LCL-10-1113', 'CB-JE-27-368', 'CB-FFM-34-255', 'CB-FFM-48-42', 'CB-EM-28-22', 'CB-FFM-23-12', 'CB-20K2-01-51', 'CB-AW-02-17', 'CB-LCL-07-363', 'CB-LCL-13-589', 'CB-JB2-07-149', 'CB-AW-26-01', 'CB-FFM-35-40', 'CB-EM-31-54', 'CB-EM-34-182', 'CB-LCL-09-286', 'CB-FFM-52-202', 'CB-CHE-11-582', 'CB-AW-33-70', 'CB-FFM-41-189', 'CB-JB2-01-102', 'CB-LCL-07-379', 'CB-JB-04-224', 'CB-JB-05-269', 'CB-EM-19-09', 'CB-CHE-11-560', 'CB-FFM-53-227', 'CB-DM-01-277', 'CB-DM-02-316', 'CB-FFM-52-48', 'CB-FFM-53-272', 'CB-EM-15-80', 'CB-FFM-53-67', 'CB-JB-05-268', 'CB-DM-03-185', 'CB-EM-46-174', 'CB-JB2-06-75', 'CB-FFM-32-191', 'CB-EM-42-35', 'CB-CHE-11-715', 'CB-CHE-11-585', 'CB-FFM-53-85', 'CB-FFM-35-77', 'CB-FFM-42-144', 'CB-LCL-05-253', 'CB-FFM-42-143', 'CB-JB2-08-413', 'CB-JB2-07-509', 'CB-FFM-34-36', 'CB-JB2-04-113', 'CB-EM-29-44']



class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, model_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filename
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last
        self.history_type = model_config["history_encoder"]["type"]
        self.gst_emb_size = model_config["history_encoder"]["gst_emb_size"]
        self.text_emb_size = model_config["history_encoder"]["text_emb_size"]
        self.max_history_len = model_config["history_encoder"]["max_history_len"]
        self.basename_to_id = dict((v, k) for k, v in enumerate(self.basename))
        #print('bn2id', self.basename_to_id)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        # print(basename)
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        #print('\nspeaker_id {}, speaker {}'.format(speaker_id, speaker))
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)

        word_path = os.path.join(
            self.preprocessed_path,
            "word",
            "{}-word-{}.npy".format(speaker, basename),
        )
        words = None #np.load(word_path)

        w2p_path = os.path.join(
            self.preprocessed_path,
            "word",
            "{}-word-{}.npy".format(speaker, basename),
        )
        wc = None #np.load(w2p_path)


        # History
        #print('basename {}'.format(basename))
        
        dialog = basename.split("_")[2].strip("d")
        #print('basename {}, dialog {}'.format(basename,dialog))
        
        turn = int(basename.split("_")[0])
        #print('turn', turn)
        
        history_len = min(self.max_history_len, turn)
        history_gst_emb = list()
        history_text_emb = list()
        history_pitch = list()
        history_energy = list()
        history_duration = list()
        history_speaker = list()
        history_mel_len = list()
        history = None
        if self.history_type == "gst-text":
            pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename) )
            history_basenames = sorted([ t[:-9] for t in os.listdir(os.path.join(self.preprocessed_path,"TextGrid/TextGrid")) 
                if t.split('_')[-1][:-9].strip('d') == dialog ], key=lambda x:int(x.split("_")[0]))
            #print('history_basenames', history_basenames)
            #sorted([tg_path.replace(".wav", "") for tg_path in os.listdir(os.path.join(self.raw_path, self.sub_dir_name, f"{dialog}")) if ".wav" in tg_path], key=lambda x:int(x.split("_")[0]))
            history_basenames = history_basenames[:turn][-history_len:]
            #print('history_basenames', history_basenames)

            if self.history_type == "gst-text":
                gst_emb_path = os.path.join(
                    self.preprocessed_path,
                    "gst_emb",
                    "{}.npy".format(basename),
                )
                text_emb_path = os.path.join(
                    self.preprocessed_path,
                    "text_emb",
                    "{}-text_emb-{}.npy".format(speaker, basename)
                )
                #print('gst_emb_path',gst_emb_path)
                gst_emb = np.load(gst_emb_path, allow_pickle=True)
                text_emb = np.load(text_emb_path)
                #print('gst_emb',gst_emb.shape)
                for i, h_basename in enumerate(history_basenames):
                    h_idx = int(self.basename_to_id[h_basename])
                    h_speaker = self.speaker[h_idx]
                    h_speaker_id = self.speaker_map[h_speaker]
                    h_gst_emb_path = os.path.join(
                        self.preprocessed_path,
                        "gst_emb",
                        "{}.npy".format(h_basename),
                    )
                    h_text_emb_path = os.path.join(
                        self.preprocessed_path,
                        "text_emb",
                        "{}-text_emb-{}.npy".format(h_speaker, h_basename)
                    )
                    h_gst_emb = np.load(h_gst_emb_path, allow_pickle=True)
                    #h_gst_emb = np.array(h_gst_emb, dtype=np.float32)
                    h_text_emb = np.load(h_text_emb_path)

                    #h_text_emb = np.array(h_text_emb, dtype=np.float32)

                    history_gst_emb.append(h_gst_emb)
                    history_text_emb.append(h_text_emb)
                    
                    history_speaker.append(h_speaker_id)
            
                    #f = np.array(history_gst_emb)

                    # Padding
                    if i == history_len-1 and history_len < self.max_history_len:
                        self.pad_history(
                            self.max_history_len-history_len,
                            history_gst_emb=history_gst_emb,
                            history_text_emb=history_text_emb,
                            history_speaker=history_speaker,
                        )
                if turn == 0:
                    self.pad_history(
                        self.max_history_len,
                        history_gst_emb=history_gst_emb,
                        history_text_emb=history_text_emb,
                        history_speaker=history_speaker,
                    )
                    #self.pad_history(
                    #    self.max_history_len,
                    #    history_text_emb=history_text_emb,
                    #    history_speaker=history_speaker,
                    #)


                history = {
                    "gst_emb": gst_emb,
                    "history_len": history_len,
                    "history_gst_emb": history_gst_emb,
                    "text_emb": text_emb,
                    "history_text_emb": history_text_emb,
                    "history_speaker": history_speaker,
                }
        # hist end

        #print('***************h_text_emb', history_text_emb.shape, history_gst_emb)
        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "words": words,
            "wc":wc,
            "history":history,
        }

        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                _ = line.strip("\n").split("|")
                n, s, t, r = _[:4] #line.strip("\n").split("|")
                # if n in neglect_files:
                #     continue
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        words = [data[idx]["words"] for idx in idxs]
        wc = [data[idx]["wc"] for idx in idxs]
        
        history_info = None
        if self.history_type != "none":
            if self.history_type == "gst-text":
                gst_embs = [data[idx]["history"]["gst_emb"] for idx in idxs]
                text_embs = [data[idx]["history"]["text_emb"] for idx in idxs]
                history_lens = [data[idx]["history"]["history_len"] for idx in idxs]
                history_gst_embs = [data[idx]["history"]["history_gst_emb"] for idx in idxs]
                history_text_embs = [data[idx]["history"]["history_text_emb"] for idx in idxs]
                history_speakers = [data[idx]["history"]["history_speaker"] for idx in idxs]

                gst_embs = np.array(gst_embs)
                #print('gst_embs',gst_embs)
                text_embs = np.array(text_embs)
                history_lens = np.array(history_lens)
                history_gst_embs = np.array(history_gst_embs, )
                history_text_embs = np.array(history_text_embs, )
                history_speakers = np.array(history_speakers)

                history_info = (
                    gst_embs,
                    history_lens,
                    history_gst_embs,
                    history_speakers,
                    text_embs, history_text_embs
                )
                #print(history_info)

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)
        #words = pad_1D(words)
        #wc = pad_1D(wc)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,
            words,
            wc,
            history_info,
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))
        return output


    def pad_history(self,
            pad_size,
            history_text=None,
            history_gst_emb=None,
            history_text_emb=None,
            history_text_len=None,
            history_pitch=None,
            history_energy=None,
            history_duration=None,
            history_emotion=None,
            history_speaker=None,
            history_mel_len=None,
        ):
        for _ in range(pad_size):
            history_text.append(np.zeros(1, dtype=np.int64)) if history_text is not None else None
            history_gst_emb.append(np.zeros(self.gst_emb_size, dtype=np.float32)) if history_gst_emb is not None else None
            history_text_emb.append(np.zeros(self.text_emb_size, dtype=np.float32)) if history_text_emb is not None else None
            history_text_len.append(0) if history_text_len is not None else None # meaningless zero padding, should be cut out by mask of history_len
            history_pitch.append(np.zeros(1, dtype=np.float64)) if history_pitch is not None else None
            history_energy.append(np.zeros(1, dtype=np.float32)) if history_energy is not None else None
            history_duration.append(np.zeros(1, dtype=np.float64)) if history_duration is not None else None
            history_emotion.append(0) if history_emotion is not None else None # meaningless zero padding, should be cut out by mask of history_len
            history_speaker.append(0) if history_speaker is not None else None # meaningless zero padding, should be cut out by mask of history_len
            history_mel_len.append(0) if history_mel_len is not None else None # meaningless zero padding, should be cut out by mask of history_len



class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config, model_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.history_type = model_config["history_encoder"]["type"]
        self.text_emb_size = model_config["history_encoder"]["text_emb_size"]
        self.gst_emb_size = model_config["history_encoder"]["gst_emb_size"]
        self.max_history_len = model_config["history_encoder"]["max_history_len"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filepath
        )
        self.basename_to_id = dict((v, k) for k, v in enumerate(self.basename))
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))


        # History
        dialog = basename.split("_")[2].strip("d")
        turn = int(basename.split("_")[0])
        history_len = min(self.max_history_len, turn)
        history_text = list()
        history_gst_emb = list()
        history_text_emb = list()
        history_text_len = list()
        history_pitch = list()
        history_energy = list()
        history_duration = list()
        history_emotion = list()
        history_speaker = list()
        history_mel_len = list()
        history = None

        if self.history_type != "none":
            history_basenames = sorted([ t[:-9] for t in os.listdir(os.path.join(self.preprocessed_path,"TextGrid/TextGrid"))
                if t.split('_')[-1][:-9].strip('d') == dialog ], key=lambda x:int(x.split("_")[0]))
            history_basenames = history_basenames[:turn][-history_len:]
            history_basenames = history_basenames[:turn][-history_len:]

            if self.history_type == "gst-text":
                gst_emb_path = os.path.join(
                    self.preprocessed_path,
                    "gst_emb",
                    "{}.npy".format(basename),
                )
                text_emb_path = os.path.join(
                    self.preprocessed_path,
                    "text_emb",
                    "{}-text_emb-{}.npy".format(speaker, basename)
                )
                #print('gst_emb_path',gst_emb_path)
                gst_emb = np.load(gst_emb_path, allow_pickle=True)
                text_emb = np.load(text_emb_path)
                #print('gst_emb',gst_emb.shape)
                for i, h_basename in enumerate(history_basenames):
                    h_idx = int(self.basename_to_id[h_basename])
                    h_speaker = self.speaker[h_idx]
                    h_speaker_id = self.speaker_map[h_speaker]
                    h_gst_emb_path = os.path.join(
                        self.preprocessed_path,
                        "gst_emb",
                        "{}.npy".format(h_basename),
                    )
                    h_text_emb_path = os.path.join(
                        self.preprocessed_path,
                        "text_emb",
                        "{}-text_emb-{}.npy".format(h_speaker, h_basename)
                    )
                    h_gst_emb = np.load(h_gst_emb_path, allow_pickle=True)
                    #h_gst_emb = np.array(h_gst_emb, dtype=np.float32)
                    h_text_emb = np.load(h_text_emb_path)

                    #h_text_emb = np.array(h_text_emb, dtype=np.float32)

                    history_gst_emb.append(h_gst_emb)
                    history_text_emb.append(h_text_emb)

                    history_speaker.append(h_speaker_id)

                    #f = np.array(history_gst_emb)

                    # Padding
                    if i == history_len-1 and history_len < self.max_history_len:
                        self.pad_history(
                            self.max_history_len-history_len,
                            history_gst_emb=history_gst_emb,
                            history_text_emb=history_text_emb,
                            history_speaker=history_speaker,
                        )
                if turn == 0:
                    self.pad_history(
                        self.max_history_len,
                        history_gst_emb=history_gst_emb,
                        history_text_emb=history_text_emb,
                        history_speaker=history_speaker,
                    )
                    #self.pad_history(
                    #    self.max_history_len,
                    #    history_text_emb=history_text_emb,
                    #    history_speaker=history_speaker,
                    #)


                history = {
                    "gst_emb": gst_emb,
                    "history_len": history_len,
                    "history_gst_emb": history_gst_emb,
                    "text_emb": text_emb,
                    "history_text_emb": history_text_emb,
                    "history_speaker": history_speaker,
                }   


        return (basename, speaker_id, phone, raw_text, history)

    def pad_history(self, 
            pad_size,
            history_text=None,
            history_gst_emb=None,
            history_text_emb=None,
            history_text_len=None,
            history_pitch=None,
            history_energy=None,
            history_duration=None,
            history_emotion=None,
            history_speaker=None,
            history_mel_len=None,
        ):
        for _ in range(pad_size):
            history_text.append(np.zeros(1, dtype=np.int64)) if history_text is not None else None
            history_gst_emb.append(np.zeros(self.gst_emb_size, dtype=np.float32)) if history_gst_emb is not None else None
            history_text_emb.append(np.zeros(self.text_emb_size, dtype=np.float32)) if history_text_emb is not None else None
            history_text_len.append(0) if history_text_len is not None else None # meaningless zero padding, should be cut out by mask of history_len
            history_pitch.append(np.zeros(1, dtype=np.float64)) if history_pitch is not None else None
            history_energy.append(np.zeros(1, dtype=np.float32)) if history_energy is not None else None
            history_duration.append(np.zeros(1, dtype=np.float64)) if history_duration is not None else None
            history_emotion.append(0) if history_emotion is not None else None # meaningless zero padding, should be cut out by mask of history_len
            history_speaker.append(0) if history_speaker is not None else None # meaningless zero padding, should be cut out by mask of history_len
            history_mel_len.append(0) if history_mel_len is not None else None # meaningless zero padding, should be cut out by mask of history_len



    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r,_ = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])

        texts = pad_1D(texts)


        if self.history_type != "none":
            if self.history_type == "gst-text":
                gst_embs = [d[4]["gst_emb"] for d in data]
                text_embs = [d[4]["text_emb"] for d in data]
                history_lens = [d[4]["history_len"] for d in data]
                history_gst_embs = [d[4]["history_gst_emb"] for d in data]
                history_text_embs = [d[4]["history_text_emb"] for d in data]
                history_speakers = [d[4]["history_speaker"] for d in data]


                gst_embs = np.array(gst_embs)
                #print('gst_embs',gst_embs)
                text_embs = np.array(text_embs)
                history_lens = np.array(history_lens)
                history_gst_embs = np.array(history_gst_embs, )
                history_text_embs = np.array(history_text_embs, )
                history_speakers = np.array(history_speakers)

                history_info = (
                    gst_embs,
                    history_lens,
                    history_gst_embs,
                    history_speakers,
                    text_embs, history_text_embs
                )


        return ids, raw_texts, speakers, texts, text_lens, max(text_lens), history_info


if __name__ == "__main__":
    # Test
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from utils.utils import to_device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess_config = yaml.load(
        open("./config/LJSpeech/preprocess.yaml", "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("./config/LJSpeech/train.yaml", "r"), Loader=yaml.FullLoader
    )

    train_dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    val_dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["optimizer"]["batch_size"] * 4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["optimizer"]["batch_size"],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    n_batch = 0
    for batchs in train_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Training set  with size {} is composed of {} batches.".format(
            len(train_dataset), n_batch
        )
    )

    n_batch = 0
    for batchs in val_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Validation set  with size {} is composed of {} batches.".format(
            len(val_dataset), n_batch
        )
    )
