#!/usr/bin/env python
import os
from eval import evaluation
from data_management import data_ingest
import argparse
from model import model, train
import time
import threading
import numpy as np
from radarseg.pytorch.radarnet import Net
import torch
import torch.nn as nn
import pickle


def inference(model, device, data, args):
    if args.use_torch: 
        model.eval()
        dataset = torch.from_numpy(data).type(torch.FloatTensor).to(device)
        output = model(dataset)
        pred = output.argmax(dim=1, keepdim=True).numpy()
        pred_data = data[np.where(pred == 1)[0], :]
        return pred_data.T
    else: 
        with open('./radarseg/tensorflow/params_tf.pkl', 'rb') as f: 
            parameters = pickle.load(f)
        Z1 = np.dot(parameters['W1'], data.T) + parameters['b1']
        A1 = np.maximum(0,Z1)
        Z2 = np.dot(parameters['W2'], A1) + parameters['b2']
        A2 = np.maximum(0,Z2)
        Z3 = np.dot(parameters['W3'], A2) + parameters['b3']
        # print("Z3:", Z3[0, :])
        pred = np.argmin(Z3, axis=0)
        # print("pred:", pred)
        pred_data = data[np.where(pred == 1)[0], :]
        return pred_data.T
    

def write_data(args):
    dr = data_ingest.DataReader(version='v1.0-mini', 
                                dataroot='/datasets/nuscenes/v1.0-mini', 
                                verbose=False)
    eval = evaluation.Evaluation()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(18, 9, 4, 2).to(device)
    model.load_state_dict(torch.load('./radarseg/pytorch/radarseg_v1.0_trainval.pt', map_location=device))

    if args.save_csv: 
        f = open('radar_v1.0_mini.csv','a')
    i = 0
    while True:
        start = time.time()
        pc_lidar, color_lidar, pose = dr.get_pointcloud('LIDAR_TOP')
        pc_radar_f, _, pose = dr.get_pointcloud('RADAR_FRONT')
        pc_radar_fl, _, pose = dr.get_pointcloud('RADAR_FRONT_LEFT')
        pc_radar_fr, _, pose = dr.get_pointcloud('RADAR_FRONT_RIGHT')
        pc_radar_bl, _, pose = dr.get_pointcloud('RADAR_BACK_LEFT')
        pc_radar_br, _, pose = dr.get_pointcloud('RADAR_BACK_RIGHT')
        boxes = dr.get_boxes('LIDAR_TOP')
        # go to next sample
        dr.next_sample()

        # grid = preproc.getRadarGrid(pc, pose)
        # preproc.addGrid2File()
        points_lidar = pc_lidar.points
        points_radar = np.c_[(pc_radar_f.points, pc_radar_fl.points, 
                              pc_radar_fr.points, pc_radar_bl.points, 
                              pc_radar_br.points)]
        points_radar_predict = inference(model, device, points_radar.T, args)
        print("points_radar_predict", points_radar_predict.shape)
        eval.plot_pointcloud(points_lidar, 'c', 0.3, '.')
        eval.plot_pointcloud(points_radar, 'm', 5, 'D')
        eval.plot_pointcloud(points_radar_predict, 'k', 10, 'o')
        eval.plot_boxes(boxes)
        eval.draw(args.save_fig, i)
        i += 1
        
        print(points_radar.shape)
        points_radar_with_anno = dr.points_with_anno(points_radar, boxes)
        if args.save_csv: 
            np.savetxt(f, points_radar_with_anno.T, delimiter=',', fmt='%.2f')
        
        # print(points_radar_with_anno.shape)
        # print(points_radar_with_anno[-1, :])
        points_radar_filter = np.squeeze(points_radar_with_anno[:, np.where(points_radar_with_anno[-1] == 1)])
        print(points_radar_filter.shape)

        end = time.time()
        # if end - start < 0.5: 
        #     time.sleep(0.5 - (end-start))
        print("Time per frame: {:1.4f}s".format(time.time() - start))
    if args.save_csv: 
        f.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--use-tf', action='store_true', default=False)
    parser.add_argument('--use-torch', action='store_true', default=False)
    parser.add_argument('--save-fig', action='store_true', default=False)
    parser.add_argument('--save-csv', action='store_true', default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    write_data(args)
    exit(0)
