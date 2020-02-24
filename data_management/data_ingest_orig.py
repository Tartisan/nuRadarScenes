#!/usr/bin/env python

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from pyquaternion import Quaternion
from pypcd import pypcd
import numpy as np
from transforms3d import quaternions, affines
import os
import itertools


class DataReader:
    def __init__(self,
                 version: str = 'v1.0-mini',
                 dataroot: str = '/datasets/nuscenes',
                 verbose: bool = True):
        self.nusc = NuScenes(
            version=version, dataroot=dataroot, verbose=verbose)
        self.sceneID = 0
        self.scene = self.nusc.scene[self.sceneID]
        self.current_sample = self.nusc.get(
            'sample', self.scene['first_sample_token'])
        print('Data Reader Initialized')

    def get_next_pcl(self,
                     sensor: str = 'RADAR_FRONT'):
        radar_front_data = self.nusc.get(
            'sample_data', self.current_sample['data'][sensor])
        radar_front_left_data = self.nusc.get(
            'sample_data', self.current_sample['data'][sensor])
        radar_front_right_data = self.nusc.get(
            'sample_data', self.current_sample['data'][sensor])
        radar_back_left_data = self.nusc.get(
            'sample_data', self.current_sample['data'][sensor])
        radar_back_right_data = self.nusc.get(
            'sample_data', self.current_sample['data'][sensor])

        radar_front_calib = self.nusc.get('calibrated_sensor',
                                          self.nusc.get('sample_data', self.current_sample['data']['RADAR_FRONT'])[
                                              'calibrated_sensor_token'])
        radar_front_left_calib = self.nusc.get('calibrated_sensor',
                                               self.nusc.get('sample_data', self.current_sample['data']['RADAR_FRONT_LEFT'])[
                                                   'calibrated_sensor_token'])
        radar_front_right_calib = self.nusc.get('calibrated_sensor',
                                                self.nusc.get('sample_data', self.current_sample['data']['RADAR_FRONT_RIGHT'])[
                                                    'calibrated_sensor_token'])
        radar_back_left_calib = self.nusc.get('calibrated_sensor',
                                              self.nusc.get('sample_data', self.current_sample['data']['RADAR_BACK_LEFT'])[
                                                  'calibrated_sensor_token'])
        radar_back_right_calib = self.nusc.get('calibrated_sensor',
                                               self.nusc.get('sample_data', self.current_sample['data']['RADAR_BACK_RIGHT'])[
                                                   'calibrated_sensor_token'])
        pc_f = RadarPointCloud.from_file(os.path.normpath(
            '/datasets/nuscenes/' + radar_front_data['filename']))
        pc_fl = RadarPointCloud.from_file(os.path.normpath(
            '/datasets/nuscenes/' + radar_front_left_data['filename']))
        pc_fr = RadarPointCloud.from_file(os.path.normpath(
            '/datasets/nuscenes/' + radar_front_right_data['filename']))
        pc_bl = RadarPointCloud.from_file(os.path.normpath(
            '/datasets/nuscenes/' + radar_back_left_data['filename']))
        pc_br = RadarPointCloud.from_file(os.path.normpath(
            '/datasets/nuscenes/' + radar_back_right_data['filename']))
        print("pc_f.points.shape: ", pc_f.points.shape)
        print("pc_fl.points.shape: ", pc_fl.points.shape)
        print("pc_fr.points.shape: ", pc_fr.points.shape)
        print("pc_bl.points.shape: ", pc_bl.points.shape)
        print("pc_br.points.shape: ", pc_br.points.shape)

        def toVecCoord(*x):
            y = np.asarray(x)
            y[0:3] = np.dot(H, np.append(y[0:3], 1))[0:3]
            return y

        pc_f.rotate(Quaternion(radar_front_calib['rotation']).rotation_matrix)
        pc_f.translate(np.array(radar_front_calib['translation']))
        pc_fl.rotate(Quaternion(
            radar_front_left_calib['rotation']).rotation_matrix)
        pc_fl.translate(np.array(radar_front_left_calib['translation']))
        pc_fr.rotate(Quaternion(
            radar_front_right_calib['rotation']).rotation_matrix)
        pc_fr.translate(np.array(radar_front_right_calib['translation']))
        pc_bl.rotate(Quaternion(
            radar_back_left_calib['rotation']).rotation_matrix)
        pc_bl.translate(np.array(radar_back_left_calib['translation']))
        pc_br.rotate(Quaternion(
            radar_back_right_calib['rotation']).rotation_matrix)
        pc_br.translate(np.array(radar_back_right_calib['translation']))

        pc = np.concatenate(
            (pc_f.points, pc_fl.points, pc_fr.points, pc_bl.points, pc_br.points), axis=1)

        ego_pose = self.nusc.get(
            'ego_pose', radar_front_data['ego_pose_token'])

        if self.current_sample['next'] == "":
            self.current_sample = self.nusc.get(
                'sample', self.scene['first_sample_token'])
            pc = None
        else:
            self.current_sample = self.nusc.get(
                'sample', self.current_sample['next'])

        return pc, ego_pose

    def nextScene(self):
        self.sceneID = self.sceneID + 1
        if self.sceneID >= len(self.nusc.scene):
            self.sceneID = 0
        self.scene = self.nusc.scene[self.sceneID]
        self.current_sample = self.nusc.get(
            'sample', self.scene['first_sample_token'])
        return self.sceneID
