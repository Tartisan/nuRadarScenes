import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from models.pointnet import PointNetEncoder, feature_transform_reguliarzer


class get_model(nn.Module):
    def __init__(self, num_class, channel=3, with_rgb=True):
        super(get_model, self).__init__()
        # if with_rgb:
        #     channel = 6
        # else:
        #     channel = 3
        self.k = num_class
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=channel)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat, weight):
        loss = F.nll_loss(pred, target, weight = weight)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        ## foreground iou loss
        pred_val = torch.argmax(pred, dim=1)
        f_I = torch.sum((pred_val == 1) & (target == 1)).type(torch.FloatTensor)
        f_U = torch.sum((pred_val == 1) | (target == 1)).type(torch.FloatTensor) + 1e-6
        fIoU_loss = 1 - f_I / f_U
        ## total_loss
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale + 0.3 * fIoU_loss
        return total_loss


if __name__ == '__main__':
    model = get_model(13, with_rgb=False)
    xyz = torch.rand(12, 3, 2048)
    (model(xyz))