#!/usr/bin/env python
import os
import sys
from data_utils.plot import PlotNuscenes
from data_utils.nuscenes_reader import NuscenesReader
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import importlib

DATA_IN = '/media/idriver/TartisanHardDisk/00-datasets/nuscenes/v1.0-mini/'
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
print("ROOT_DIR:", ROOT_DIR)

NUM_CLASSES = 2

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--save_fig', action='store_true', default=False)
    parser.add_argument('--model', type=str, default='fc3', help='model name')
    return parser.parse_args()

def main(args):
    classifier = FC3(15, 9, 4, 2).to(device)
    use_cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda" if use_cuda else "cpu")
    model_dir = os.path.join(ROOT_DIR, 'log/'+args.model)
    '''MODEL LOADING'''
    checkpoint = torch.load(model_dir + '/best_model.pth', map_location=device)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    nusc = NuscenesReader(version='v1.0-mini', dataroot=DATA_IN, verbose=False)
    pln = PlotNuscenes()
    while True: 
        start = time.time()
        print('==================== sample ======================')
        pc_lidar, color_lidar, pose = nusc.get_pointcloud('LIDAR_TOP')
        pc_radar_f, _, pose         = nusc.get_pointcloud('RADAR_FRONT')
        pc_radar_fl, _, pose        = nusc.get_pointcloud('RADAR_FRONT_LEFT')
        pc_radar_fr, _, pose        = nusc.get_pointcloud('RADAR_FRONT_RIGHT')
        pc_radar_bl, _, pose        = nusc.get_pointcloud('RADAR_BACK_LEFT')
        pc_radar_br, _, pose        = nusc.get_pointcloud('RADAR_BACK_RIGHT')
        boxes = nusc.get_boxes('LIDAR_TOP')
        nusc.next_sample()
        points_lidar = pc_lidar.points
        points_radar = np.c_[pc_radar_f.points, pc_radar_fl.points, 
                             pc_radar_fr.points, pc_radar_bl.points, 
                             pc_radar_br.points]
        points_radar_predict = inference(points_radar.T, classifier, device)
        # points_radar_with_anno = nusc.points_with_anno(points_radar, boxes)
        # points_radar_filter = np.squeeze(points_radar_with_anno[:, np.where(points_radar_with_anno[-1] == 1)])
        
        print('points_radar.shape:', points_radar.shape)
        # print('radar points num in boxes:', points_radar_filter.shape[1])
        pln.plot_pointcloud(points_lidar, 'c', 0.3, '.')
        pln.plot_pointcloud(points_radar, 'm', 5, 'D')
        pln.plot_pointcloud(points_radar_predict, 'k', 10, 'o')
        pln.plot_boxes(boxes)
        pln.draw(args.save_fig, ROOT_DIR+'/log/'+args.model+'/inference')
        end = time.time()
        # if end - start < 1000: 
        #     time.sleep(1000 - (end-start))
        # print("Time per frame: {:1.4f}s".format(time.time() - start))

def inference(points_orig, classifier, device):
    # resample to fixed shape: num_point * 16
    points_orig = np.delete(points_orig, [4,8,9], axis=1)
    # forward propagation
    points = torch.from_numpy(points_orig).type(torch.FloatTensor).to(device)
    output = classifier(points)
    pred = output.argmax(dim=1, keepdim=True).numpy()
    pred_val = points[np.where(pred == 1)[0], :]
    return pred_val.T


if __name__ == "__main__":
    args = parse_args()
    main(args)