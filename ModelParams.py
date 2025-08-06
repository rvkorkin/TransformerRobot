# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 18:27:46 2021

@author: RKorkin
"""

class ModelParams(object):
    def __init__(self):
        self.map_size = 10
        self.num_sensors = 20
        self.prob_wall = 0.1
        self.track_len = 50
        self.speed_base = 0.5
        self.speed_noise = 0.1
        self.theta_base = 0.05
        self.theta_noise = 0.02
        self.sensor_noise = 0.1
        self.hidden_dim = 64
        self.num_beac = 20
        self.beac_emb = 64
        self.emb_obs = 32
        self.emb_act = 32
        self.loc_emb = 128
        self.obs_num = 4
        self.batch_size = 200
        self.random_seed = 0
        self.dropout = 0.5
        self.look_back = 1
        self.measurement_noise_roughening = 0.2
        self.random_seed = 0
        self.h_weight = 1
        self.bp_length = 10
        self.bpdecay = 0.02
        self.pNumber = 20