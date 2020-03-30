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

DATA_IN = '/datasets/nuscenes/v1.0-mini/'
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
print("ROOT_DIR:", ROOT_DIR)

NUM_CLASSES = 2
NUM_POINT = 512

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--save_fig', action='store_true', default=False)
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg', help='model name')
    return parser.parse_args()

def main(args):
    MODEL = importlib.import_module(args.model)
    use_cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda" if use_cuda else "cpu")
    model_dir = os.path.join(ROOT_DIR, 'log/sem_seg/'+args.model+'/checkpoints')
    '''MODEL LOADING'''
    classifier = MODEL.get_model(NUM_CLASSES).to(device)
    criterion = MODEL.get_loss().to(device)
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
        points_radar_predict = inference(points_radar.T, classifier, args.model, device)
        # points_radar_with_anno = nusc.points_with_anno(points_radar, boxes)
        # points_radar_filter = np.squeeze(points_radar_with_anno[:, np.where(points_radar_with_anno[-1] == 1)])
        
        print('points_radar.shape:', points_radar.shape)
        # print('radar points num in boxes:', points_radar_filter.shape[1])
        pln.plot_pointcloud(points_lidar, 'c', 0.3, '.')
        pln.plot_pointcloud(points_radar, 'm', 5, 'D')
        pln.plot_pointcloud(points_radar_predict, 'k', 10, 'o')
        pln.plot_boxes(boxes)
        pln.draw(args.save_fig, ROOT_DIR+'/log/sem_seg/'+args.model+'/inference')
        end = time.time()
        # if end - start < 1000: 
        #     time.sleep(1000 - (end-start))
        # print("Time per frame: {:1.4f}s".format(time.time() - start))

def inference(points_orig, classifier, model, device):
    # resample to fixed shape: num_point * 16
    points_orig = np.delete(points_orig, [4,8,9], axis=1)
    point_idxs = np.array(range(points_orig.shape[0]))
    if points_orig.shape[0] >= NUM_POINT:
        selected_point_idxs = np.random.choice(point_idxs, NUM_POINT, replace=False)
    else:
        selected_point_idxs = np.random.choice(point_idxs, NUM_POINT, replace=True)
    if model == 'pointnet2_sem_seg': 
        points_resample = points_orig[selected_point_idxs, :]
    else: 
        points_resample = points_orig[selected_point_idxs, :]
        points_resample = points_resample[:, [0,1,2,4,5,6]]
    # forward propagation
    points = torch.Tensor(points_resample).float().to(device).view(1, NUM_POINT, -1)
    points = points.transpose(2, 1)
    seg_pred, trans_feat = classifier(points)
    pred_val = seg_pred.contiguous().cpu().data.numpy()
    pred_val = np.argmax(pred_val, 2)
    print('foreground points num:', np.sum(pred_val))
    points_predict = points_resample[np.where(pred_val == 1)[0], :]
    return points_predict.T


if __name__ == "__main__":
    args = parse_args()
    main(args)
