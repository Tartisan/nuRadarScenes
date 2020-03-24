#!/usr/bin/env python

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, BoxVisibility
from pyquaternion import Quaternion
import numpy as np
import os
import itertools

class NuscenesReader:
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
        # self.invalid_states = list(range(18))
        # self.dynprop_states = list(range(8))
        # self.ambig_states = list(range(5))
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
                    os.path.normpath(self.dataroot + sensor_sample['filename']))
            intensities = pc.points[3, :]
            intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
            intensities = intensities ** 0.1
            intensities = np.maximum(0, intensities - 0.5)
            coloring = intensities
        else:
            pc = RadarPointCloud.from_file(
                    os.path.normpath(self.dataroot + sensor_sample['filename']), 
                    self.invalid_states, self.dynprop_states, self.ambig_states)
            coloring = pc.points[2, :]

        # print(sensor, "points.shape:", pc.points.shape)
        ## transform the point-cloud to the ego vehicle frame
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

    def test(self): 
        sensor_sample_token = self.current_sample['data']['RADAR_FRONT']
        sensor_calib_token = self.nusc.get('sample_data', sensor_sample_token)['calibrated_sensor_token']
        sensor_sample = self.nusc.get('sample_data', sensor_sample_token)
        sensor_calib = self.nusc.get('calibrated_sensor', sensor_calib_token)

        pc = RadarPointCloud.from_file(
                os.path.normpath(self.dataroot + sensor_sample['filename']), 
                self.invalid_states, self.dynprop_states, self.ambig_states)
        pc_xyz = pc.points[:3, :]
        print(pc_xyz.shape)

        _, boxes, _ = self.nusc.get_sample_data(self.current_sample['data']['RADAR_FRONT'], BoxVisibility.ANY,
                                                use_flat_vehicle_coordinates=True)
        mask = points_in_box(boxes[5], pc_xyz, wlh_factor=1.0)
        print((mask).shape)
        # pc_xyz = pc_xyz[mask]
        print(pc_xyz.T[mask].shape)
        print(pc_xyz.T[mask])

    def points_with_anno(self, points, boxes):
        points_with_anno = np.r_[points, np.zeros((1, points.shape[1]))]
        for box in boxes:
            # category: 'car', 'truck', 'bus', 'trailer', 'construction_vehicle', 
            #           'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'
            category_name = box.name
            if ('cone' in category_name or 
                'barrier' in category_name):
                continue
            mask = self.points_in_box(points_with_anno[:2, :], box)
            points_with_anno[-1, :] += mask
        return points_with_anno

    def points_in_box(self, points, box):
        """
        Checks whether points are inside the box.

        Picks one corner as reference (p1) and computes the vector to a target point (v).
        Then for each of the 3 axes, project v onto the axis and compare the length.
        Inspired by: https://math.stackexchange.com/a/1552579
        :param box: <Box>.
        :param points: <np.float: 3, n>.
        :param wlh_factor: Inflates or deflates the box.
        :return: <np.bool: n, >.
        """
        corners = view_points(box.corners(), np.eye(3), False)[:2, [7, 3, 2, 6]]
        # print(corners)
        p1 = corners[:, 0]
        p_x = corners[:, 1]
        p_y = corners[:, 3]
        # p_z = corners[:, 3]

        i = p_x - p1
        j = p_y - p1
        # k = p_z - p1

        v = points - p1.reshape((-1, 1))

        iv = np.dot(i, v)
        jv = np.dot(j, v)
        # kv = np.dot(k, v)

        mask_x = np.logical_and(0 <= iv, iv <= np.dot(i, i))
        mask_y = np.logical_and(0 <= jv, jv <= np.dot(j, j))
        # mask_z = np.logical_and(0 <= kv, kv <= np.dot(k, k))
        mask = np.logical_and(mask_x, mask_y)

        return mask

