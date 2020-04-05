import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CHANNELS = {'x':0, 'y':1, 'z':2, 'dyn_prop':3, 'id':4, 'rcs':5, 
            'vx':6, 'vy':7, 'vx_comp':8, 'vy_comp':9, 'is_quality_valid':10, 
            'ambig_state':11, 'x_rms':12, 'y_rms':13, 'invalid_state':14, 
            'pdh0':15, 'vx_rms':16, 'vy_rms':17, 'label':18}

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--channel', type=str, default='all', help='radar points channel')
    return parser.parse_args()

def main(args):
    file_name = os.path.join(ROOT_DIR, 'radar_v1.0_mini.csv')
    points = np.loadtxt(file_name, delimiter=',')

    if args.channel == 'all':
        for key in CHANNELS: 
            if key in ['z', 'id', 'is_quality_valid', 'label']:
                pass
            plot_hist(key, points)
            plt.savefig(ROOT_DIR+'/statistic/'+'%02d-' % CHANNELS[key]+key+'.png')
    else:
        plot_hist(args.channel, points)
        plt.show()

def plot_hist(key, points):
    """
    绘制直方图
    data:必选参数，绘图数据
    bins:直方图的长条形数目，可选项，默认为10
    density:是否将得到的直方图向量归一化，可选项，默认为0，代表不归一化，显示频数。density=1，表示归一化，显示频率。
    facecolor:长条形的颜色
    edgecolor:长条形边框的颜色
    alpha:透明度
    """
    points_channel_neg = points[np.where(points[:, -1] == 0), CHANNELS[key]].T
    points_channel_pos = points[np.where(points[:, -1] >= 1), CHANNELS[key]].T

    plt.figure(figsize=(12, 8))
    plt.suptitle('Channel: ' + key, fontsize=14)
    plt.subplot(1,2,1)
    plt.hist(points_channel_pos, bins=20, density=1, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.title('foreground')
    plt.subplot(1,2,2)
    plt.hist(points_channel_neg, bins=20, density=1, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.title('background')


if __name__ == "__main__":
    args = parse_args()
    main(args)
