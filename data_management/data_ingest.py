#!/usr/bin/env python

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from pyquaternion import Quaternion
import numpy as np
import os
import itertools

class DataReader:
    def __init__(self,
                 version: str = 'v1.0-mini',
                 dataroot: str = '/datasets/nuscenes',
                 verbose: bool = True):
        self.dataroot = dataroot
        self.nusc = NuScenes(
            version=version, dataroot=dataroot, verbose=verbose)
        self.scene_id = 0
        self.scene = self.nusc.scene[self.scene_id]
        self.current_sample = self.nusc.get('sample', self.scene['first_sample_token'])
        # for radar filter
        self.invalid_states = [0, 4, 8, 9, 10, 11, 12, 15, 16, 17]
        self.dynprop_states = [0, 1, 2, 3, 5, 6, 7]
        self.ambig_states = [2, 3, 4]
        print('Data Reader Initialized')

    def get_pointcloud(self,
                        sensor: str = 'LIDAR_TOP'):
        valid_sensors = ['RADAR_FRONT_LEFT', 'RADAR_FRONT', 'RADAR_FRONT_RIGHT',
                         'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT', 'LIDAR_TOP']
        assert sensor in valid_sensors, 'Input sensor {} not valid.'.format(sensor)

        sensor_sample_token = self.current_sample['data'][sensor]
        sensor_calib_token = self.nusc.get('sample_data', sensor_sample_token)['calibrated_sensor_token']
        sensor_sample = self.nusc.get('sample_data', sensor_sample_token)
        sensor_calib = self.nusc.get('calibrated_sensor', sensor_calib_token)

        if sensor == 'LIDAR_TOP': 
            pc = LidarPointCloud.from_file(
                    os.path.normpath(self.dataroot + '/' + sensor_sample['filename']))
            intensities = pc.points[3, :]
            intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
            intensities = intensities ** 0.1
            intensities = np.maximum(0, intensities - 0.5)
            coloring = intensities
        else:
            pc = RadarPointCloud.from_file(
                    os.path.normpath(self.dataroot + '/' + sensor_sample['filename']), 
                    self.invalid_states, self.dynprop_states, self.ambig_states)
            coloring = pc.points[2, :]

        print(sensor, "points.shape:", pc.points.shape)
        # transform the point-cloud to the ego vehicle frame
        pc.rotate(Quaternion(sensor_calib['rotation']).rotation_matrix)
        pc.translate(np.array(sensor_calib['translation']))

        ego_pose = self.nusc.get('ego_pose', sensor_sample['ego_pose_token'])

        return pc, coloring, ego_pose
    
    def get_boxes(self, sensor='LIDAR_TOP'):
        _, boxes, _ = self.nusc.get_sample_data(self.current_sample['data'][sensor], BoxVisibility.ANY,
                                                use_flat_vehicle_coordinates=True)
        return boxes

    def next_sample(self):
        if self.current_sample['next'] == "":
            if self.next_scene() == 0: 
                exit(0)
        else:
            self.current_sample = self.nusc.get('sample', self.current_sample['next'])

    def next_scene(self):
        self.scene_id = self.scene_id + 1
        if self.scene_id >= len(self.nusc.scene):
            self.scene_id = 0
        self.scene = self.nusc.scene[self.scene_id]
        self.current_sample = self.nusc.get('sample', self.scene['first_sample_token'])
        return self.scene_id
