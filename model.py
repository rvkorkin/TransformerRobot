import random
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
from ModelParams import ModelParams
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
import math
import torch.nn.functional as F
from dataset import LocalizationDataset
import torch.optim as optim

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=ModelParams().track_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10*max_len) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #return self.dropout(x + self.pe[:, :x.size(1)])
        return x + self.pe[:, :x.size(1)]

def active2mask(anti_vector):
    return torch.bmm(anti_vector.unsqueeze(2), anti_vector.unsqueeze(1)).bool()

def generate_mask(seq_len):    
    M = torch.full((seq_len, seq_len), True)
    for i in range(seq_len):
        M[i, max(0, i-seq_len+1):i+1] = False
    return M.to(device)

class MapEncoderLayer(nn.Module):
    def __init__(self, num_heads=4):
        super(MapEncoderLayer, self).__init__()
        self.hidden_dim = 256
        self.num_heads = num_heads
        self.att = nn.MultiheadAttention(self.hidden_dim, self.num_heads, batch_first=True).double().to(device)
        self.norm1 = nn.LayerNorm(self.hidden_dim).double().to(device)
        self.norm2 = nn.LayerNorm(self.hidden_dim).double().to(device)
        self.two_layers = nn.Sequential(
            nn.Linear(self.hidden_dim, 4*self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(4*self.hidden_dim, self.hidden_dim),
        ).to(device)

    def forward(self, emb, active):
        batch_size, seq_size = emb.size(0), emb.size(1)
        key_padding_mask = (1 - active)
        #print('batch_size, seq_size: ', batch_size, seq_size)
        #print('key_padding_mask', key_padding_mask.shape)
        beacon_mask = active2mask(key_padding_mask).unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        beacon_mask = beacon_mask.view(batch_size * self.num_heads, seq_size, seq_size)
        y = self.att(emb, emb, emb, key_padding_mask=key_padding_mask.bool(), attn_mask=beacon_mask)[0] + emb
        y = self.norm1(y)
        y = self.two_layers(y) + y
        y = self.norm2(y)
        return y

class MapEncoder(nn.Module):
    def __init__(self, num_heads=4):
        super(MapEncoder, self).__init__()
        self.map_encoder1 = MapEncoderLayer(num_heads)
        self.map_encoder2 = MapEncoderLayer(num_heads)
        self.map_encoder3 = MapEncoderLayer(num_heads)
        self.map_encoder4 = MapEncoderLayer(num_heads)
        self.map_encoder5 = MapEncoderLayer(num_heads)
        self.map_encoder6 = MapEncoderLayer(num_heads)
        self.map_encoder7 = MapEncoderLayer(num_heads)
        self.map_encoder8 = MapEncoderLayer(num_heads)

    def forward(self, emb, v):
        return self.map_encoder8(self.map_encoder7(self.map_encoder6(self.map_encoder5(self.map_encoder4(self.map_encoder3(self.map_encoder2(self.map_encoder1(emb, v), v), v), v), v), v), v), v)

class DecoderLayer(nn.Module):
    def __init__(self, num_heads=4):
        super(DecoderLayer, self).__init__()
        self.hidden_dim = 256
        self.num_heads = num_heads
        self.self_att = nn.MultiheadAttention(self.hidden_dim, self.num_heads, batch_first=True).double().to(device)
        self.cross_att = nn.MultiheadAttention(self.hidden_dim, self.num_heads, batch_first=True).double().to(device)
        self.norm1 = nn.LayerNorm(self.hidden_dim).double().to(device)
        self.norm2 = nn.LayerNorm(self.hidden_dim).double().to(device)
        self.norm3 = nn.LayerNorm(self.hidden_dim).double().to(device)
        self.two_layers = nn.Sequential(
            nn.Linear(self.hidden_dim, 4*self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(4*self.hidden_dim, self.hidden_dim),
        ).to(device)

    def forward(self, h, b, m_h, v):
        h = self.self_att(h, h, h, attn_mask=m_h)[0] + h
        h = self.norm1(h)
        key_padding_mask = 1 - v
        pair = self.cross_att(h, b, b, key_padding_mask=key_padding_mask.bool())
        h = pair[0] + h
        att = pair[1]
        h = self.norm2(h)
        h = self.two_layers(h) + h
        h = self.norm3(h)
        return att, h

class Decoder(nn.Module):
    def __init__(self, num_heads=4):
        super(Decoder, self).__init__()
        self.decoder1 = DecoderLayer(num_heads)
        self.decoder2 = DecoderLayer(num_heads)
        self.decoder3 = DecoderLayer(num_heads)
        self.decoder4 = DecoderLayer(num_heads)
        self.decoder5 = DecoderLayer(num_heads)
        self.decoder6 = DecoderLayer(num_heads)
        self.decoder7 = DecoderLayer(num_heads)
        self.decoder8 = DecoderLayer(num_heads)

    def forward(self, emb, y, m, v):
        x = self.decoder7(self.decoder6(self.decoder5(self.decoder4(self.decoder3(self.decoder2(self.decoder1(emb, y, m, v)[1], y, m, v)[1], y, m, v)[1], y, m, v)[1], y, m, v)[1], y, m, v)[1], y, m, v)[1]
        a, x = self.decoder8(x, y, m, v)
        return a, x

class MainModel(nn.Module):
    def __init__(self, params=ModelParams(), num_heads=4):
        super(MainModel, self).__init__()
        self.hidden_dim = 256
        self.num_heads = num_heads
        self.beac_emb = 256
        self.map_size = 10
        self.num_obs = params.obs_num
        self.num_sensors = params.num_sensors
        self.hidden2label = nn.Sequential(
            nn.Linear(self.hidden_dim, 3),
            nn.Dropout(0.1),
            nn.Sigmoid()
        ).double().to(device)
        self.track_len = params.track_len
        self.h_weight = params.h_weight
        self.actobs_embedding = nn.Linear(6, self.hidden_dim).double().to(device)
        self.beac_emb = nn.Linear(2, self.beac_emb).double().to(device)
        self.pos = PositionalEncoding(d_model=self.hidden_dim).double().to(device)
        self.look_back = params.look_back
        self.map_enc = MapEncoder()
        self.dec = Decoder()

    def forward(self, act_in, obs_in, beac_in, y_gt, active=None):
        if active == None:
            active = torch.ones((beac_in.size(0), beac_in.size(1))).double().to(device)
        batch_size, seq_len = act_in.size(0), act_in.size(1)
        mask = generate_mask(seq_len)
        x0 = self.actobs_embedding(torch.cat((act_in, obs_in), dim=2))
        x = self.pos(x0)
        b = self.beac_emb(beac_in)
        b = self.map_enc(b, active)
        att, h = self.dec(x, b, mask, active)
        return att, self.hidden2label(h)

    def test(self, act_in, obs_in, beac_in, active=None, look_back=None):
        if active == None:
            active = torch.ones((beac_in.size(0), beac_in.size(1))).double().to(device)
        batch_size, seq_len = act_in.size(0), act_in.size(1)
        if look_back==None:
            look_back = seq_len
        mask = generate_mask(seq_len)
        x0 = self.actobs_embedding(torch.cat((act_in, obs_in), dim=2))
        b = self.beac_emb(beac_in)
        b = self.map_enc(b, active)
        hidden_states = torch.zeros(batch_size, 0, self.hidden_dim).double().to(device)
        att_all = torch.zeros(batch_size, 0, beac_in.size(1)).double().to(device)
        for time_step in range(seq_len):
            start = max(0, time_step - look_back)
            x = self.pos(x0[:, start:time_step+1])
            att, h = self.dec(x, b, mask[start:time_step+1, start:time_step+1], active)
            hidden_states = torch.cat((hidden_states, h[:, -1].unsqueeze(1)), dim=1)
            att_all = torch.cat((att_all, att[:, -1].unsqueeze(1)), dim=1)
        return att_all, self.hidden2label(hidden_states)

    def step(self, act_in, obs_in, beacons, gt_pos, bpdecay, active):
        gt_y_normalized = gt_pos[:, :, 1:2] / self.map_size
        gt_theta_normalized = gt_pos[:, :, 2:] / (2 * np.pi)
        gt_normalized = torch.cat((gt_x_normalized, gt_y_normalized, gt_theta_normalized), dim=2)
        att, pred = self.forward(act_in, obs_in, beacons, gt_normalized, active=active)
        seq_len = pred.size(1)
        bpdecay_params = torch.exp(bpdecay * torch.arange(seq_len))
        bpdecay_params = bpdecay_params / bpdecay_params.mean()
        bpdecay_params = 1 # bpdecay_params.unsqueeze(0).unsqueeze(2).double().to(device)
        l2_xy_loss =  torch.sum(torch.nn.functional.mse_loss(pred[:, :-1, :2], gt_normalized[:, 1:, :2], reduction='none') * bpdecay_params)
        l2_h_loss1 = torch.nn.functional.mse_loss(torch.cos(2 * np.pi * pred[:, :-1, 2]), torch.cos(2 * np.pi * gt_normalized[:, 1:, 2])) * bpdecay_params
        l2_h_loss2 = torch.nn.functional.mse_loss(torch.sin(2 * np.pi * pred[:, :,-1 2]), torch.sin(2 * np.pi * gt_normalized[:, 1:, 2])) * bpdecay_params
        l2_h_loss = torch.sum(l2_h_loss1 + l2_h_loss2)
        #print('xy loss: ', l2_xy_loss)
        #print('vxy loss: ', l2_vxy_loss)
        return l2_xy_loss + 0.1 * l2_h_loss, att, pred
