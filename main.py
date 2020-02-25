#!/usr/bin/env python

from eval import evaluation
from data_management import data_ingest, preprocessing
import argparse
from model import model, train
import tensorflow as tf
import time, threading
import matplotlib.pyplot as plt


def writeDataset():
    dr = data_ingest.DataReader(version='v1.0-mini', dataroot='/datasets/nuscenes', verbose=False)
    eval = evaluation.Evaluation()
    # preproc = preprocessing.Preprocessing()

    while True:
        start = time.time()
        pc_lidar, color_lidar, pose = dr.get_pointcloud('LIDAR_TOP')
        pc_radar_f, color_radar_f, pose = dr.get_pointcloud('RADAR_FRONT')
        pc_radar_fl, color_radar_fl, pose = dr.get_pointcloud('RADAR_FRONT_LEFT')
        pc_radar_fr, color_radar_fr, pose = dr.get_pointcloud('RADAR_FRONT_RIGHT')
        pc_radar_bl, color_radar_bl, pose = dr.get_pointcloud('RADAR_BACK_LEFT')
        pc_radar_br, color_radar_br, pose = dr.get_pointcloud('RADAR_BACK_RIGHT')
        boxes = dr.get_boxes('LIDAR_TOP')
        # go to next sample
        dr.next_sample()

        # grid = preproc.getRadarGrid(pc, pose)
        # preproc.addGrid2File()
        eval.plot_pointcloud(pc_lidar, 'c', 0.3, '.')
        eval.plot_pointcloud(pc_radar_f, color_radar_f, 5, ',')
        eval.plot_pointcloud(pc_radar_fl, color_radar_fl, 5, ',')
        eval.plot_pointcloud(pc_radar_fr, color_radar_fr, 5, ',')
        eval.plot_pointcloud(pc_radar_bl, color_radar_bl, 5, ',')
        eval.plot_pointcloud(pc_radar_br, color_radar_br, 5, ',')
        eval.plot_boxes(boxes)
        # eval.plotGrid(grid)
        # eval.plotTrajectory(pose)
        eval.draw()

        end = time.time()
        if end - start < 0.5: 
            time.sleep(0.5 - (end-start))
        print("Time per frame: {:1.4f}s".format(time.time() - start))

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
