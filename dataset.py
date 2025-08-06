# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:27:30 2021

@author: RKorkin
"""
import os
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from scipy.ndimage import gaussian_filter


class LocalizationDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.seq_len = len(self.data['tracks'][0])
        self.seq_num = len(self.data['tracks'])
        self.samp_seq_len = None

    def __len__(self):
        return self.seq_num

    def set_samp_seq_len(self, seq_len):
        self.samp_seq_len = seq_len

    def __getitem__(self, index):
        seq_idx = index % self.seq_num
        env_map = self.data['map'][seq_idx][:, :, :10]
        all_beac = self.data['map'][seq_idx][:, :, 10:].reshape(30, 3)
        beacons = all_beac[:, :2]
        active = all_beac[:, 2:]
        
        traj = self.data['tracks'][seq_idx]
        
        self.map_mean = np.mean(env_map)
        self.map_std = np.std(env_map)
        env_map = (env_map - self.map_mean) / self.map_std
        env_map = torch.FloatTensor(env_map).unsqueeze(0)
        traj = torch.FloatTensor(traj)
        beacons_tensor = torch.FloatTensor(beacons)
        active_tensor = torch.FloatTensor(active)
        walls_tensor = torch.FloatTensor(walls)

        if self.samp_seq_len is not None and self.samp_seq_len != self.seq_len:
            start = np.random.randint(0, self.seq_len - self.samp_seq_len + 1)
            traj = traj[start:start + self.samp_seq_len]

        obs = traj[:, 5:]
        action = traj[:, 3:5]
        gt_pos = traj[:, :3]

        return (beacons_tensor, active_tensor, obs, action, gt_pos)
