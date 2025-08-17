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
        all_info = self.data['map'][seq_idx][:, :, -6:]
        beacons = all_info[:, :, :4].reshape(all_info.shape[0], 20, 2)
        beacons_active = all_info[:, :, 4:].reshape(all_info.shape[0], 20, 1)
        #walls_flatten = np.argwhere(env_map.flatten()==1).flatten()
        #walls = np.vstack((walls_flatten % env_map.shape[0], env_map.shape[0] - 1 - walls_flatten // env_map.shape[0])).T + 0.5

        traj = self.data['tracks'][seq_idx]

        #self.map_mean = np.mean(env_map)
        #self.map_std = np.std(env_map)
        #env_map = (env_map - self.map_mean) / self.map_std
        #env_map = torch.FloatTensor(env_map).unsqueeze(0)
        traj = torch.FloatTensor(traj)
        beacons_tensor = torch.FloatTensor(beacons)
        beacons_active_tensor = torch.FloatTensor(beacons_active)
        #walls_tensor = torch.FloatTensor(walls)

        if self.samp_seq_len is not None and self.samp_seq_len != self.seq_len:
            start = np.random.randint(0, self.seq_len - self.samp_seq_len + 1)
            traj = traj[start:start + self.samp_seq_len]

        obs = traj[:, 5:]
        action = traj[:, 3:5]
        gt_pos = traj[:, :3]

        return (beacons_tensor, beacons_active_tensor, obs, action, gt_pos)