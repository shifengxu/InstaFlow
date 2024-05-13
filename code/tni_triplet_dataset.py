import os
import numpy as np
from torch.utils.data import Dataset

from utils import log_info


class TextNoiseImageDataset(Dataset):
    def __init__(self, data_dir, data_limit=0):
        self.data_dir = data_dir
        self.data_limit = data_limit
        if not os.path.exists(data_dir):
            raise ValueError(f"Path not exist: {data_dir}")
        if not os.path.isdir(data_dir):
            raise ValueError(f"Path not dir: {data_dir}")
        img_dir = os.path.join(data_dir, 'image')
        file_list = os.listdir(img_dir)
        file_list = [f[:-4] for f in file_list if f.endswith('.png')]
        file_list.sort()
        # file_list will be like ['00001', '00002', '00003']
        log_info(f"TextNoiseImageDataset()")
        log_info(f"  data_dir   : {data_dir}")
        log_info(f"  file cnt   : {len(file_list)}")
        log_info(f"  file[0]    : {file_list[0]}")
        log_info(f"  file[-1]   : {file_list[-1]}")
        log_info(f"  data_limit : {data_limit}")
        if data_limit > 0:
            file_list = file_list[:data_limit]
        log_info(f"  train      : {len(file_list)}")
        log_info(f"  train[0]   : {file_list[0]}")
        log_info(f"  train[-1]  : {file_list[-1]}")
        self.file_list = file_list


    def __getitem__(self, index):
        fname = self.file_list[index]
        gaussian_noise  = np.load(os.path.join(self.data_dir, "gaussian_noise", f"{fname}.npy"))
        generate_latent = np.load(os.path.join(self.data_dir, "generate_latent", f"{fname}.npy"))
        prompt_pos_emb  = np.load(os.path.join(self.data_dir, "prompt_pos_emb", f"{fname}.npy"))
        prompt_neg_emb  = np.load(os.path.join(self.data_dir, "prompt_neg_emb", f"{fname}.npy"))
        return gaussian_noise, generate_latent, prompt_pos_emb, prompt_neg_emb

    def __len__(self):
        return len(self.file_list)
