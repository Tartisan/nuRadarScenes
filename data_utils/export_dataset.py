#!/usr/bin/env python
import os
import sys
import time
import numpy as np
from nuscenes_reader import NuscenesReader

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print("ROOT_DIR:", ROOT_DIR)
sys.path.append(ROOT_DIR)

DATA_OUT = '/media/idriver/TartisanHardDisk/00-datasets/nuscenes/v1.0-trainval/radar_npy/'
DATA_IN = '/media/idriver/TartisanHardDisk/00-datasets/nuscenes/v1.0-trainval/'

nusc = NuscenesReader(version='v1.0-trainval', dataroot=DATA_IN, verbose=False)
name_idx = 0
while True:
    start = time.time()
    pc_radar_f, _, pose  = nusc.get_pointcloud('RADAR_FRONT')
    pc_radar_fl, _, pose = nusc.get_pointcloud('RADAR_FRONT_LEFT')
    pc_radar_fr, _, pose = nusc.get_pointcloud('RADAR_FRONT_RIGHT')
    pc_radar_bl, _, pose = nusc.get_pointcloud('RADAR_BACK_LEFT')
    pc_radar_br, _, pose = nusc.get_pointcloud('RADAR_BACK_RIGHT')
    boxes = nusc.get_boxes('LIDAR_TOP')
    nusc.next_sample()

    points_radar = np.c_[(pc_radar_f.points, pc_radar_fl.points, 
                          pc_radar_fr.points, pc_radar_bl.points, 
                          pc_radar_br.points)]
    points_radar_with_anno = nusc.points_with_anno(points_radar, boxes)
    points_radar_filter = np.squeeze(points_radar_with_anno[:, np.where(points_radar_with_anno[-1] >= 1)])
    print('==================== sample ======================')
    print('points_radar.shape:', points_radar.shape)
    print('points_radar_with_anno.shape:', points_radar_with_anno.shape)
    # print('radar poins num in boxes:', points_radar_filter.shape[1])
    np.save(DATA_OUT + 'v1.0_trainval_sample_%05d.npy' % name_idx, points_radar_with_anno.T)
    name_idx += 1
    end = time.time()
    # if end - start < 1000: 
    #     time.sleep(1000 - (end-start))
    # print("Time per frame: {:1.4f}s".format(time.time() - start))