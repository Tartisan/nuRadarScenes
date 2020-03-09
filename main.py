#!/usr/bin/env python

from eval import evaluation
from data_management import data_ingest
import argparse
from model import model, train
import time
import threading
import numpy as np
from radarseg import Net

def write_dataset(save_csv=False):
    dr = data_ingest.DataReader(version='v1.0-mini', 
                                dataroot='/media/idriver/TartisanHardDisk/00-datasets/nuscenes/v1.0-mini', 
                                verbose=False)
    eval = evaluation.Evaluation()

    if save_csv: 
        f = open('radar_v1.0_mini.csv','a')
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
        eval.plot_pointcloud(points_lidar, 'c', 0.3, '.')
        eval.plot_pointcloud(points_radar, 'm', 5, 'D')
        eval.plot_boxes(boxes)
        eval.draw()
        
        print(points_radar.shape)
        points_radar_with_anno = dr.points_with_anno(points_radar, boxes)
        # save points with annotation
        if save_csv: 
            np.savetxt(f, points_radar_with_anno.T, delimiter=',', fmt='%.2f')
        
        # print(points_radar_with_anno.shape)
        # print(points_radar_with_anno[-1, :])
        points_radar_filter = np.squeeze(points_radar_with_anno[:, np.where(points_radar_with_anno[-1] == 1)])
        print(points_radar_filter.shape)

        end = time.time()
        # if end - start < 0.5: 
        #     time.sleep(0.5 - (end-start))
        print("Time per frame: {:1.4f}s".format(time.time() - start))
    if save_csv: 
        f.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--write-dataset', action='store_true')
    # parser.add_argument('--train-model', action='store_true')
    parser.add_argument('--save-csv', action='store_true', default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    write_dataset(args.save_csv)

    exit(0)
