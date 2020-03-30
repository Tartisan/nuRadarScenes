import numpy as np
import math
import torch
from torch.utils.data import Dataset

DATA_IN = '/media/idriver/TartisanHardDisk/00-datasets/nuscenes/v1.0-trainval/radar_csv/radar_v1.0_mini.csv'

class RadarDataset(Dataset):
    def __init__(self, split='train', split_ratio=0.8):
        super().__init__()
        points_radar_with_anno = np.loadtxt(DATA_IN, delimiter=',')
        '''
        radar points channels: 
        x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms label
        '''
        num_train = math.floor(points_radar_with_anno.shape[0] * split_ratio)
        # train data
        X_train_orig = points_radar_with_anno[:num_train, :18]
        # delete channels: id / vx_comp / vy_comp
        X_train_orig = np.delete(X_train_orig, [4,8,9], axis=1)
        Y_train_orig = points_radar_with_anno[:num_train, -1]
        Y_train_orig[Y_train_orig > 1] = 1
        # test data
        X_test_orig = points_radar_with_anno[num_train:, :18]
        X_test_orig = np.delete(X_test_orig, [4,8,9], axis=1)
        Y_test_orig = points_radar_with_anno[num_train:, -1]
        Y_test_orig[Y_test_orig > 1] = 1
        print("Trainset ground ratio: ", 1.-1.*np.sum(Y_train_orig)/Y_train_orig.shape[0])
        print("Testset ground ratio: ", 1.-1.*np.sum(Y_test_orig)/Y_test_orig.shape[0])
        if split == 'train': 
            self.datas = X_train_orig
            self.labels = Y_train_orig
        else: 
            self.datas = X_test_orig
            self.labels = Y_test_orig
        print(split+' datas shape:', self.datas.shape)
        print(split+' labels shape:', self.labels.shape)

        # set weights according to num of each label
        labelweights,_ = np.histogram(self.labels, range(3))
        labelweights = labelweights.astype(np.float32) / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print('self.labelweights:', self.labelweights)

    def __getitem__(self, idx):
        return self.datas[idx], self.labels[idx]

    def __len__(self):
        return self.datas.shape[0]


if __name__ == '__main__':
    radar_data = RadarDataset(split='train', split_ratio=0.8)
    print('point data size:', radar_data.__len__())
    print('point data 0 shape:', radar_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', radar_data.__getitem__(0)[1].shape)
    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(radar_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
