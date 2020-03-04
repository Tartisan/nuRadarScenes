#!/usr/bin/env python

from eval import evaluation
from data_management import data_ingest, preprocessing
import argparse
from model import model, train
import tensorflow as tf
import time
import threading
import numpy as np


def writeDataset():
    dr = data_ingest.DataReader(version='v1.0-mini', 
                                dataroot='/datasets/nuscenes', 
                                verbose=False)
    eval = evaluation.Evaluation()
    # preproc = preprocessing.Preprocessing()

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
        # eval.plotGrid(grid)
        # eval.plotTrajectory(pose)
        eval.draw()
        
        print(points_radar.shape)
        points_radar_with_anno = dr.points_with_anno(points_radar, boxes)
        # save points with annotation
        np.savetxt(f, points_radar_with_anno.T, delimiter=',', fmt='%.2f')
        
        # print(points_radar_with_anno.shape)
        # print(points_radar_with_anno[-1, :])
        points_radar_filter = np.squeeze(points_radar_with_anno[:, np.where(points_radar_with_anno[-1] == 1)])
        print(points_radar_filter.shape)

        end = time.time()
        # if end - start < 0.5: 
        #     time.sleep(0.5 - (end-start))
        print("Time per frame: {:1.4f}s".format(time.time() - start))
    f.close()

def trainModel():
    preproc = preprocessing.Preprocessing()
    preproc.readFile()

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--write-dataset', action='store_true')
    parser.add_argument('--train-model', action='store_true')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.write_dataset:
        writeDataset()

    if args.train_model:
        trainModel()

    
    model = model.MyModel()

    # mnist = tf.keras.datasets.mnist

    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train, x_test = x_train / 255.0, x_test / 255.0

    # # Add a channels dimension
    # x_train = x_train[..., tf.newaxis]
    # x_test = x_test[..., tf.newaxis]

    # train_ds = tf.data.Dataset.from_tensor_slices(
    #     (x_train, y_train)).shuffle(10000).batch(32)
    # test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    # train.train(model, train_ds, test_ds)

    exit(0)
