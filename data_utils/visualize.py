#!/usr/bin/env python
import os
import sys
from plot import PlotNuscenes
from nuscenes_reader import NuscenesReader
import argparse
import time
import numpy as np
import torch
import torch.nn as nn

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print("ROOT_DIR:", ROOT_DIR)
sys.path.append(ROOT_DIR)

DATA_OUT = '/media/idriver/TartisanHardDisk/00-datasets/nuscenes/v1.0-trainval/radar_npy/'
DATA_IN = '/media/idriver/TartisanHardDisk/00-datasets/nuscenes/v1.0-trainval/'

def visualize(args):
    nusc = NuscenesReader(version='v1.0-trainval', dataroot=DATA_IN, verbose=False)
    pln = PlotNuscenes()
    name_idx = 0
    while True:
        start = time.time()
        pc_lidar, color_lidar, pose = nusc.get_pointcloud('LIDAR_TOP')
        pc_radar_f, _, pose         = nusc.get_pointcloud('RADAR_FRONT')
        pc_radar_fl, _, pose        = nusc.get_pointcloud('RADAR_FRONT_LEFT')
        pc_radar_fr, _, pose        = nusc.get_pointcloud('RADAR_FRONT_RIGHT')
        pc_radar_bl, _, pose        = nusc.get_pointcloud('RADAR_BACK_LEFT')
        pc_radar_br, _, pose        = nusc.get_pointcloud('RADAR_BACK_RIGHT')
        boxes = nusc.get_boxes('LIDAR_TOP')
        nusc.next_sample()

        points_lidar = pc_lidar.points
        points_radar = np.c_[(pc_radar_f.points, pc_radar_fl.points, 
                              pc_radar_fr.points, pc_radar_bl.points, 
                              pc_radar_br.points)]
        points_radar_with_anno = nusc.points_with_anno(points_radar, boxes)
        points_radar_filter = np.squeeze(points_radar_with_anno[:, np.where(points_radar_with_anno[-1] == 1)])
        print('==================== sample ======================')
        print('points_radar.shape:', points_radar.shape)
        print('points_radar_with_anno.shape:', points_radar_with_anno.shape)
        # print('radar poins num in boxes:', points_radar_filter.shape[1])
        pln.plot_pointcloud(points_lidar, 'c', 0.3, '.')
        pln.plot_pointcloud(points_radar, 'm', 5, 'D')
        pln.plot_boxes(boxes)
        pln.draw(args.save_fig, name_idx)
        name_idx += 1
        end = time.time()
        if end - start < 1000: 
            time.sleep(1000 - (end-start))
        print("Time per frame: {:1.4f}s".format(time.time() - start))


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--save_fig', action='store_true', default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize(args)
    exit(0)
