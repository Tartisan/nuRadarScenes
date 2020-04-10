import os
import string
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
file_name = os.path.join(ROOT_DIR, 'log/sem_seg/pointnet_sem_seg/logs/pointnet_sem_seg.txt')

costs = []
fIoUs = []
with open(file_name, 'r') as f:
    for line in f.readlines():
        if 'Training loss' in line: 
            cost = float(line.split(': ')[1])
            costs.append(cost)
        elif 'foreground IoU' in line: 
            fIoU = float(line.split(': ')[1])
            fIoUs.append(fIoU)

plt.figure()
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('epochs')
plt.title('PointNet Sematic Segmantation')

plt.figure()
plt.plot(np.squeeze(fIoUs))
plt.ylabel('foreground IoU')
plt.xlabel('epochs')
plt.title('PointNet Sematic Segmantation')
plt.show()