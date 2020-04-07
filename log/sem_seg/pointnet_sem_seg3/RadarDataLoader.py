import os
import numpy as np
from torch.utils.data import Dataset

DATA_ROOT = '/media/idriver/TartisanHardDisk/00-datasets/nuscenes/v1.0-trainval/radar_npy/'

class RadarDataset(Dataset):
    def __init__(self, split='train', data_root='/', num_point=512, model='pointnet2_sem_seg', split_ratio=0.8):
        super().__init__()
        self.num_point = num_point
        # load all file
        all_files = os.listdir(DATA_ROOT)
        self.num_sample = len(all_files)
        data_batches = []
        label_batches = []
        for npy_filename in all_files:
            '''
            radar points channels: 
            x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms label
            '''
            sample = np.load(DATA_ROOT + npy_filename) # sample.shape[0] * 19
            ## normalization
            sample[sample[:, -1] >= 1, -1] = 1
            sample[:, :18] = sample[:, :18] / (np.max(sample[:, :18], axis=0).reshape(1, -1) + 1e-5)
            ## resample 
            point_idxs = np.array(range(sample.shape[0]))
            if sample.shape[0] >= self.num_point:
                selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
            else:
                selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)
            resample = sample[selected_point_idxs, :]
            if model == 'pointnet2_sem_seg':
                data_batches.append(resample[:, 0:18])
            else: 
                resample[:, 2] = np.sqrt(np.square(resample[:, 8]) + np.square(resample[:, 9]))
                resample[:, 2] = resample[:, 2] * np.sign(resample[:, 8])
                data_batches.append(resample[:, [0,1,2,5]])
            label_batches.append(resample[:, -1])
            
        split_num = np.floor(self.num_sample * split_ratio).astype(int)
        if split == 'train': 
            self.datas = data_batches[0:split_num]
            self.labels = label_batches[0:split_num]
        else: 
            self.datas = data_batches[split_num:]
            self.labels = label_batches[split_num:]
        print('self.datas len:', len(self.datas))
        print('self.labels len:', len(self.labels))

        # set weights according to num of each label
        labelweights,_ = np.histogram(self.labels, range(3))
        labelweights = labelweights.astype(np.float32) / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print('self.labelweights:', self.labelweights)

    def __getitem__(self, idx):
        return self.datas[idx], self.labels[idx]

    def __len__(self):
        return len(self.datas)


if __name__ == '__main__':
    num_point, split_ratio, block_size, sample_rate = 512, 0.8, 1.0, 0.01

    radar_data = RadarDataset(split='train', data_root=DATA_ROOT, num_point=num_point, split_ratio=split_ratio)
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
    # for idx in range(4):
    #     end = time.time()
    #     for i, (input, target) in enumerate(train_loader):
    #         print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
    #         end = time.time()