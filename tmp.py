# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 09:13:01 2024

@author: RKorkin
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

num0 = 0

data_type = 'eval'
if data_type == 'train':
    numbers = 100
elif data_type=='eval':
    numbers = 100
else:
    numbers = 100
world_data = [0] * numbers
for i in tqdm(range(num0, num0+numbers, 1)):
    z = np.loadtxt('worlds_' + data_type + '/world' + str(i) + '.csv', delimiter=',')
    xy = z[:, -6:-2].reshape(20, 2)
    w = z[:, -2:].reshape(20, 1)
    w[np.argwhere(np.isnan(xy.sum(axis=1)))] = 0
    w.reshape(10, 2)
    world_data[i-num0] = np.hstack((z[:, :-2], w.reshape(10, 2)))

z = [0] * numbers
for i in tqdm(range(num0, num0+numbers, 1)):
    z[i-num0] = np.array(pd.read_csv('tracks_' + data_type + '/trajectories' + str(i) + '.csv', header=None))

z = np.array(z)
world_data = np.array(world_data)
track_data = z.reshape(numbers*50, 9)

np.savetxt('worlds_' + data_type + '.csv', world_data.reshape(numbers, 160))
np.savetxt('tracks_' + data_type + '.csv', track_data)
