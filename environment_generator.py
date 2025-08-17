# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 19:55:13 2024

@author: RKorkin
"""

import random
import numpy as np
import pandas as pd
import torch
from copy import deepcopy
from tqdm import tqdm
from ModelParams import ModelParams
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import matplotlib as mpl
from ModelParams import ModelParams

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class World():
    def __init__(self):
        super(World, self).__init__()
        self.map_size = ModelParams().map_size
        self.num_sensors = ModelParams().num_sensors
        self.prob_wall = ModelParams().prob_wall
        self.sensors = np.zeros((self.num_sensors, 2))
        self.walls = np.zeros((0, 2))   

    def wall_neighbors(self, point):
        A1 = (point[0]-1)*self.map_size + point[1]
        A2 = (point[0]+1)*self.map_size + point[1]
        A3 = point[0]*self.map_size + point[1]-1
        A4 = point[0]*self.map_size + point[1]+1
        A5 = (point[0]-1)*self.map_size + point[1]-1
        A6 = (point[0]+1)*self.map_size + point[1]+1
        A7 = (point[0]-1)*self.map_size + point[1]+1
        A8 = (point[0]+1)*self.map_size + point[1]-1
        neighbors = (A1 in self.walls[:, 0]*self.map_size+self.walls[:, 1]) + (A2 in self.walls[:, 0]*self.map_size+self.walls[:, 1]) + \
            (A3 in self.walls[:, 0]*self.map_size+self.walls[:, 1]) + (A4 in self.walls[:, 0]*self.map_size+self.walls[:, 1]) +\
                (A5 in self.walls[:, 0]*self.map_size+self.walls[:, 1]) + (A6 in self.walls[:, 0]*self.map_size+self.walls[:, 1]) +\
                    (A7 in self.walls[:, 0]*self.map_size+self.walls[:, 1]) + (A8 in self.walls[:, 0]*self.map_size+self.walls[:, 1])
        return neighbors

    def sensor_neighbors(self, point):
        A1 = (point[0]-1)*self.map_size + point[1]
        A2 = (point[0]+1)*self.map_size + point[1]
        A3 = point[0]*self.map_size + point[1]-1
        A4 = point[0]*self.map_size + point[1]+1
        A5 = (point[0]-1)*self.map_size + point[1]-1
        A6 = (point[0]+1)*self.map_size + point[1]+1
        A7 = (point[0]-1)*self.map_size + point[1]+1
        A8 = (point[0]+1)*self.map_size + point[1]-1
        sensor_neighbors = (A1 in self.sensors[:, 0]*self.map_size+self.sensors[:, 1]) + (A2 in self.sensors[:, 0]*self.map_size+self.sensors[:, 1])
        + (A3 in self.sensors[:, 0]*self.map_size+self.sensors[:, 1]) + (A4 in self.sensors[:, 0]*self.map_size+self.sensors[:, 1])
        + (A5 in self.sensors[:, 0]*self.map_size+self.sensors[:, 1]) + (A6 in self.sensors[:, 0]*self.map_size+self.sensors[:, 1])
        + (A7 in self.sensors[:, 0]*self.map_size+self.sensors[:, 1]) + (A8 in self.sensors[:, 0]*self.map_size+self.sensors[:, 1])
        return sensor_neighbors

    def set_walls(self):
        for j in range(400):
            new_wall = np.random.randint(0, self.map_size, size=2)
            N1 = self.wall_neighbors(new_wall)
            N2 = self.sensor_neighbors(new_wall)
            if N1 + N2 > 2:
                local_prob = 0
            else:
                local_prob = (N1//2 + 1) * self.prob_wall
            prob = np.random.uniform(0, 1)
            if prob < local_prob:
                A = new_wall[0]*self.map_size + new_wall[1]
                if A not in self.walls[:, 0]*self.map_size+self.walls[:, 1] and A not in self.sensors[:, 0]*self.map_size+self.sensors[:, 1]:
                    self.walls = np.vstack((self.walls, new_wall[np.newaxis, :]))

    def set_sensors(self):
        sensors_number = np.random.randint(4, self.num_sensors+1)
        idxs = np.random.choice(int(self.map_size*self.map_size), replace=False, size=sensors_number)
        #idxs = np.array([0, 1, 2, 10, 11, 12, 20, 21, 22, 3])
        coord_x = idxs // self.map_size
        coord_y = idxs % self.map_size
        coords = np.hstack((coord_x[:, np.newaxis], coord_y[:, np.newaxis]))
        #sensors_1d = coords[:, 0] * self.map_size + coords[:, 1]
        #walls_1d = self.walls[:, 0] * self.map_size + self.walls[:, 1]
        #idx = np.array([(x in sensors_1d) for x in walls_1d])
        self.sensors = coords
        sig = ((self.sensors - self.sensors.mean(axis=0, keepdims=True))**2).mean()**0.5
        #self.walls = self.walls[~idx]
        return sig

    def set_env(self):
        sig = self.set_sensors()
        self.set_walls()
        return sig

num_tracks = 1

num = 0
for k in tqdm(range(0, 100)):
    env = World()
    sig = env.set_env()

    W = np.zeros((env.map_size, env.map_size))
    for el in env.walls.astype(int):
        W[env.map_size-el[1]-1, el[0]] = 1
    for i, el in enumerate(env.sensors.astype(int)):
        W[env.map_size-el[1]-1, el[0]] = 1
    W_sensors = np.vstack((env.sensors, np.nan*np.ones((20-len(env.sensors), 2))))
    np.savetxt('worlds_test/world' + str(num) + '.csv', np.hstack((W, W_sensors.reshape(10, 4) + 0.5)), delimiter=',')
    num += 1

plt.close('all')
plt.figure()
plt.imshow(W)
print(W.sum())
#plt.scatter(env.sensors[:, 0]+0.5, env.sensors[:, 1]+0.5, color='red')
plt.xlim(-0.5, env.map_size-0.5)
plt.ylim(-0.5, env.map_size-0.5)
