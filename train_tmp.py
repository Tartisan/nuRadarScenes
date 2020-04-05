import os
import sys
sys.path.append(os.getcwd())
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import math
import matplotlib.pyplot as plt
from models.focal_loss import focal_loss
from models.fc10 import get_model
from data_utils.utils import load_torch_data

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_IN = '/media/idriver/TartisanHardDisk/00-datasets/nuscenes/v1.0-trainval/radar_csv'

def test(args, model, device, data_loader):
    # 模型评估
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    print('Test Accuracy: {}/{} = {:.4f}'.format(
        correct, len(data_loader.dataset), 1. * correct / len(data_loader.dataset)))
    

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='for Saving the current Model')
    return parser.parse_args()


def main(args):
    print(args)
    print("batch_size: ", args.batch_size)
    print("epochs: ", args.epochs)
    print("learing_rate: ", args.lr)

    torch.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    # Loading the dataset
    X_train, Y_train, X_test, Y_test, weights = load_torch_data(DATA_IN+'/radar_v1.0_trainval.csv')
    # to DataLoader
    trainset = Data.TensorDataset(X_train, Y_train)
    testset = Data.TensorDataset(X_test, Y_test)
    train_loader = Data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = Data.DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=True, **kwargs)

    classifier = get_model().to(device)
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0)
    classifier.apply(init_weights)
    costs = []
    #### train(args, classifier, device, train_loader, costs)
    # optimizer = optim.SGD(classifier.parameters(), lr=args.lr)
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=20, gamma=args.gamma)
    classifier.train()
    loss_fn = focal_loss(alpha=0.05, gamma=2, num_classes=2)
    for epoch in range(0, args.epochs+1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = classifier(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch % args.log_interval == 0:
            print('Cost after Epoch {}: {:.6f}'.format(epoch, loss.item()))
        if epoch % 1 == 0: 
            costs.append(loss.item())
        
        test(args, classifier, device, train_loader)
        test(args, classifier, device, test_loader)

    # # plot the cost
    # plt.plot(np.squeeze(costs))
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per tens)')
    # plt.title("Learning rate =" + str(args.lr))
    # plt.show()

    if args.save_model:
        torch.save(model.state_dict(), ROOT_DIR+'/radarseg_v1.0_fc10_trainval.pth')


if __name__ == '__main__':
    args = parse_args()
    main(args)