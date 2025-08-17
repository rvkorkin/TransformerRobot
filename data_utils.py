# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:44:44 2021

@author: RKorkin
"""

import random
import numpy as np
import pandas as pd
import torch
from copy import deepcopy
from tqdm import tqdm
from ModelParams import ModelParams
from dataset import LocalizationDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import matplotlib as mpl
import os

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

obs_num = ModelParams().obs_num
track_len = ModelParams().track_len

class Robot():
    def __init__(self, world_data):
        super(Robot, self).__init__()
        self.speed_base = ModelParams().speed_base
        self.speed_noise = ModelParams().speed_noise
        self.theta_base = ModelParams().speed_noise
        self.theta_noise = ModelParams().theta_noise
        self.sensor_noise = ModelParams().sensor_noise
        self.speed = 0
        self.dtheta = 0
        self.world_data = world_data
        self.height = ModelParams().map_size
        self.width = ModelParams().map_size
        self.blocks = []
        self.sensors = []
        self.direction = np.random.normal(2 * np.pi)
        self.x, self.y = self.random_place()
        self.sensors = world_data[:, 10:14].reshape(20, 2)
        for i, line in enumerate(self.world_data):
            nb_y = self.height - i - 1
            for j, block in enumerate(line):
                if block == 1:
                    self.blocks.append((j, nb_y))

    def inside(self, x, y):
        if x < 0 or y < 0 or x > self.width or y > self.height:
            return False
        return True

    def free(self, x, y):
        if not self.inside(x, y):
            return False
        return self.world_data[self.height - int(y) - 1][int(x)] == 0

    def random_place(self):
        while True:
            x, y = np.random.uniform(0, self.width), np.random.uniform(0, self.height)
            if self.free(x, y):
                return x, y

    def distance(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def distance_to_sensors(self, x, y, measurement_num=5):
        distances = []
        for i, c in enumerate(self.sensors):
            d = self.distance(c[0], c[1], x, y)
            distances.append(d)
            idxs = np.argsort(distances)[:measurement_num]
        return idxs, sorted(distances)[:measurement_num]

    def read_sensor(self, measurement_num):
        idxs, measurement = self.distance_to_sensors(self.x, self.y, measurement_num)
        return idxs, [x + self.sensor_noise * np.random.uniform(-1, 1) for x in measurement]

    def move(self):
        direction = deepcopy(self.direction)
        speed = self.speed_base * np.random.uniform(0, 1)
        dtheta = self.theta_base * np.random.normal(0, 2 * np.pi)
        direction += dtheta
        while True:
            dx = np.cos(direction) * speed
            dy = np.sin(direction) * speed
            if self.free(self.x + dx, self.y + dy):
                break
            dtheta = np.random.normal(2 * np.pi)
            direction += dtheta
        self.x += dx
        self.y += dy
        self.speed = speed + self.speed_noise * np.random.uniform(-1, 1)
        self.dtheta = dtheta + self.theta_noise * np.random.normal(0, 2 * np.pi)
        self.direction = direction % (2*np.pi)
        return self.speed, self.dtheta

def gen_track(world_data, track_len=track_len, measurement_num=obs_num):
    robot = Robot(world_data)
    track_ret = []
    inactive = set()
    for _ in range(track_len):
        #step_data = [deepcopy(robot.x), deepcopy(robot.y), deepcopy(robot.direction)]
        robot.speed, robot.dtheta = robot.move()
        motion_data = [deepcopy(robot.speed), deepcopy(robot.dtheta)]
        idxs, sensor_data = robot.read_sensor(measurement_num)
        inactive.update(idxs)
        step_data = [deepcopy(robot.x), deepcopy(robot.y), deepcopy(robot.direction)]
        step_data = step_data + motion_data + sensor_data
        track_ret.append(step_data)
    return inactive, np.array(track_ret), world_data

def gen_data(world_data, num_tracks=1, track_len=track_len, measurement_num=obs_num):
    data_tracks = {'tracks': []}
    for _ in range(num_tracks):
        inactive, track_data, world_data = gen_track(world_data, track_len, measurement_num)
        data_tracks['tracks'].append(track_data)
    data_tracks['map'] = np.array(world_data)
    tracks = np.array(data_tracks['tracks'])
    Matr = np.zeros((0, tracks[0].shape[1]))
    for i in range(num_tracks):
        Matr = np.vstack((Matr, tracks[i]))
    return inactive, Matr

if __name__ == "__main__":
    track_len = ModelParams().track_len
    #np.random.seed(ModelParams().random_seed)
    data_type = 'test'
    num0 = 0
    num = num0
    folder_path = 'tracks_'+data_type
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")
    for num in tqdm(range(num0, num0+100)):
        info = np.loadtxt('worlds_'+data_type+'/world'+str(num)+'.csv', delimiter=',')
        world = info[:, :10]
        beacons = info[:, 10:14].reshape(20, 2)
        active, Matr = gen_data(info, track_len=track_len)
        active = np.array(list(active))
        active0 = deepcopy(active)
        inactive = np.array([el for el in range(len(beacons)) if el not in active])
        #if len(inactive):
        #    inactive2active = np.random.choice(inactive, replace=False, size=np.random.randint(len(inactive)))
        #    active = np.append(active, inactive2active)
        #walls_flatten = np.argwhere(world.flatten()==1).flatten()
        #walls = np.vstack((walls_flatten % world.shape[0], world.shape[0] - 1 - walls_flatten // world.shape[0])).T + 0.5
        #b_flatten = np.argwhere(world.flatten()==2).flatten()
        #b = np.vstack((b_flatten % world.shape[0], world.shape[0] - 1 - b_flatten // world.shape[0])).T + 0.5
        active_sensors = np.zeros(len(beacons))
        active_sensors[active] = 1
        world_new = np.hstack((info[:, :14], active_sensors.reshape(10, 2)))
        np.savetxt('worlds_'+data_type+'/world'+str(num)+'.csv', world_new, delimiter=",")
        np.savetxt('tracks_'+data_type+'/trajectories'+str(num)+'.csv', Matr, delimiter=",")

    def draw(world, location):
        fig, ax = plt.subplots(figsize=(8, 8))
        for i in range(world.shape[0]):
            yy = world.shape[0] - i - 1
            for j in range(world.shape[1]):
                xx = j
                if world[i, j] == 1.0:
                    r = mpl.patches.Rectangle((xx, yy), 1, 1, facecolor='gray', alpha=0.5)
                    ax.add_patch(r)
                if world[i, j] == 2.0:
                    r = mpl.patches.Rectangle((xx, yy), 1, 1, facecolor='black', alpha=0.5)
                    ax.add_patch(r)
                    #el = mpl.patches.Ellipse((xx+0.5, yy+0.5), 0.2, 0.2, facecolor='black')
                    #ax.add_patch(el)
                    ax.scatter(xx, yy, s=1000, color='orange', marker='X')
        for k in active:
            r = mpl.patches.Rectangle((beacons[k, 0]-0.5, beacons[k, 1]-0.5), 1, 1, facecolor='black', alpha=0.5)
            ax.add_patch(r)
            ax.scatter(beacons[k, 0], beacons[k, 1], s=1000, color='orange', marker='X')
        for k in active0:
            #r = mpl.patches.Rectangle((beacons[k, 0]-0.5, beacons[k, 1]-0.5), 1, 1, facecolor='black', alpha=0.5)
            #ax.add_patch(r)
            ax.scatter(beacons[k, 0], beacons[k, 1], s=1000, color='red', marker='X')
        ax.scatter(location[:, 0], location[:, 1], color='red', alpha=0.5, label='true location')
        plt.xlim(0, world.shape[1])
        plt.ylim(0, world.shape[0])
        plt.legend(loc='lower left', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=18)
        #plt.savefig('world_fig_' + str(len(world)) + '.pdf')
    plt.close('all')

    #data_tracks = gen_data(world, num_tracks=1)
    location = Matr[:, :2]
    draw(world, location)
