import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from models.pointnet import STN2d, STNkd, feature_transform_reguliarzer

class get_model(nn.Module):
    def __init__(self, n_class, n_channel=4):
        '''v1 3D Instance Segmentation PointNet
        :param n_class: 2
        :param n_channel: 4
        '''
        super(get_model, self).__init__()
        self.k = n_class
        self.cls = ClassficationNet(n_class, n_channel)
        self.conv1 = torch.nn.Conv1d(1088+n_class, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, n_class, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        '''
        :param x: [bs,n_channel,n_pts]
        :param x: [bs,n_pts,n_class]
        '''
        batch_size, n_channel, n_pts = x.size()
        x, trans_feat = self.cls(x)
        x = F.relu(self.bn1(self.conv1(x))) # bs,512,n
        x = F.relu(self.bn2(self.conv2(x))) # bs,256,n
        x = F.relu(self.bn3(self.conv3(x))) # bs,128,n
        x = self.conv4(x) # bs,n_class,n
        x = x.transpose(2,1).contiguous() # bs,n,n_class
        x = F.log_softmax(x.view(-1,self.k), dim=-1) # bs*n,n_class
        x = x.view(batch_size, n_pts, self.k) # bs,n,n_class
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
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale + 0.3 * fIoU_loss # 
        return total_loss

class ClassficationNet(nn.Module):
    def __init__(self, n_class=2, n_channel=4):
        '''v1 3D Instance Segmentation PointNet
        :param n_class: 2
        :param n_channel: 4
        '''
        super(ClassficationNet, self).__init__()
        self.k = n_class
        self.stn = STN2d(n_channel)
        self.fstn = STNkd(64)
        self.conv1 = torch.nn.Conv1d(n_channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_class)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)


    def forward(self, x):
        '''
        :param x: [bs,n_channel,n]
        :param x8: [bs,1088+n_class,n]
        '''
        batch_size, n_channel, n_pts = x.size()
        # T-net for 3*3
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x, feature = x.split(2,dim=2)
        x = torch.bmm(x, trans)
        x = torch.cat([x,feature],dim=2)
        x = x.transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))  # bs,64,n
        x = F.relu(self.bn2(self.conv2(x))) # bs,64,n

        # T-net for 64*64
        trans_feat = self.fstn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans_feat)
        semseg_pts = x.transpose(2, 1)

        x = F.relu(self.bn3(self.conv3(semseg_pts))) # bs,64,n
        x = F.relu(self.bn4(self.conv4(x))) # bs,128,n
        x = F.relu(self.bn5(self.conv5(x))) # bs,1024,n 
        x = torch.max(x, 2, keepdim=True)[0] # bs,1024,1
        global_feat = x.view(batch_size, -1) # bs,1024

        x = F.relu(self.bn6(self.fc1(global_feat))) # bs,512
        x = F.relu(self.bn7(self.fc2(x))) # bs,256
        cls_score = F.log_softmax(self.fc3(x), dim=1) # bs,n_class

        concat_feat = torch.cat([global_feat, cls_score], 1) # bs,1024+n_class
        concat_feat = concat_feat.view(batch_size, -1, 1).repeat(1, 1, n_pts) # bs,1024+n_class,n
        semseg_pts = torch.cat([semseg_pts, concat_feat], 1) # bs,1088+n_class,n
        return semseg_pts, trans_feat