import torch
from torch import nn
import torch.nn.functional as F
from models.focal_loss import focal_loss

class get_model(nn.Module):
    def __init__(self):
        super(get_model, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(5, 16), nn.BatchNorm1d(16),nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(16, 32), nn.BatchNorm1d(32),nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(32, 64), nn.BatchNorm1d(64),nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(64, 128), nn.BatchNorm1d(128),nn.ReLU())
        self.layer5 = nn.Sequential(nn.Linear(128, 256), nn.BatchNorm1d(256),nn.ReLU())
        self.layer6 = nn.Sequential(nn.Linear(256, 512), nn.BatchNorm1d(512),nn.ReLU())
        self.layer7 = nn.Sequential(nn.Linear(512, 1024), nn.BatchNorm1d(1024),nn.ReLU())
        self.layer8 = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512),nn.ReLU())
        self.layer9 = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256),nn.ReLU())
        self.layer10 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128),nn.ReLU())
        self.layer11 = nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64),nn.ReLU())
        self.layer12 = nn.Sequential(nn.Linear(64, 32), nn.BatchNorm1d(32),nn.ReLU())
        self.layer13 = nn.Sequential(nn.Linear(32, 16), nn.BatchNorm1d(16),nn.ReLU())
        self.layer14 = nn.Sequential(nn.Linear(16, 8), nn.BatchNorm1d(8),nn.ReLU())
        self.layer15 = nn.Sequential(nn.Linear(8, 4), nn.BatchNorm1d(4),nn.ReLU())
        self.layer16 = nn.Sequential(nn.Linear(4, 2))
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        return x

class get_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=2):
        super(get_loss, self).__init__()
        self.loss_fn = focal_loss(alpha, gamma, num_classes)

    def forward(self, pred, target):
        # total_loss = F.cross_entropy(pred, target, weight=weight)
        total_loss = self.loss_fn(pred, target)
        return total_loss