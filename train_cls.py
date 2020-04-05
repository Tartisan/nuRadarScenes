import os
import sys
import argparse
import logging
import datetime
from pathlib import Path
import importlib
import shutil
import numpy as np
import math
import matplotlib.pyplot as plt
from data_utils.utils import load_torch_data
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
DATA_IN = '/media/idriver/TartisanHardDisk/00-datasets/nuscenes/v1.0-trainval/radar_csv'


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch NuScenes Radar Tutorial')
    parser.add_argument('--model', type=str, default='fc8', help='model name [default: fc8]')
    parser.add_argument('--batch_size', type=int, default=512, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=512, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=5, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)    
    
    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir #.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir #.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, 'log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    torch.manual_seed(args.seed)

    NUM_CLASSES = 2
    # Loading the dataset
    X_train, Y_train, X_test, Y_test, weights = load_torch_data(DATA_IN+'/radar_v1.0_trainval.csv')
    # to DataLoader
    TRAIN_DATASET = Data.TensorDataset(X_train, Y_train)
    TEST_DATASET = Data.TensorDataset(X_test, Y_test)
    train_loader = Data.DataLoader(dataset=TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = Data.DataLoader(dataset=TEST_DATASET, batch_size=args.batch_size, shuffle=True, **kwargs)
    weights = torch.Tensor(weights).to(device)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('data_utils/utils.py', str(experiment_dir))
    classifier = MODEL.get_model().to(device)
    alpha_focal_loss = 0.01
    gamma_focal_loss = 2
    criterion = MODEL.get_loss(alpha_focal_loss, gamma_focal_loss, num_classes=2).to(device)
    log_string("focal_loss parameters: alpha=%f, gamma=%d" % (alpha_focal_loss, gamma_focal_loss))

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0)
    try:
        checkpoint = torch.load(str(experiment_dir) + '/best_model.pth')
        start_epoch = checkpoint['epoch']
        best_iou = checkpoint['foreground_iou']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        best_iou = 0
        classifier = classifier.apply(init_weights)

    costs = []
    global_epoch = 0
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=20, gamma=args.gamma)
    for epoch in range(start_epoch, args.epochs + 1):
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epochs))
        classifier.train()
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
            points, target = data
            points, target = points.to(device), target.to(device)
            optimizer.zero_grad()
            output = classifier(points)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        scheduler.step()
        log_string('Training loss: %f' % loss.item())
        if epoch % args.log_interval == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')
        if epoch % 1 == 0: 
            costs.append(loss.item())

        '''Evaluate on chopped scenes'''
        classifier.eval()
        loss = 0
        correct = 0
        with torch.no_grad():
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            for points, target in test_loader:
                points, target = points.to(device), target.to(device)
                output = classifier(points)
                loss += criterion(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                pred_val = pred.cpu().data.numpy()
                batch_label = target.cpu().data.numpy()
                for l in range(NUM_CLASSES):
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l) )
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)) )
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
            fIoU = np.mean(np.array(total_correct_class[1]) / (np.array(total_iou_deno_class[1], dtype=np.float) + 1e-6))
            log_string('Test point avg class IoU: %f' % (mIoU))
            log_string('Test point foreground IoU: %f' % (fIoU))
            log_string('Test Accuracy: {}/{} = {:.4f}'.format(
                correct, len(test_loader.dataset), 1. * correct / len(test_loader.dataset)))
            if fIoU >= best_iou:
                best_iou = fIoU
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'foreground_iou': fIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best fIoU: %f' % best_iou)
        global_epoch += 1



    # # plot the cost
    # plt.plot(np.squeeze(costs))
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per tens)')
    # plt.title("Learning rate =" + str(args.lr))
    # plt.show()


def train(args, model, device, train_loader, costs):
    # optimizer = optim.SGD(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=20, gamma=args.gamma)
    model.train()
    loss_fn = focal_loss(alpha=0.05, gamma=2, num_classes=2)
    for epoch in range(0, args.epochs+1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch % args.log_interval == 0:
            print('Cost after Epoch {}: {:.6f}'.format(epoch, loss.item()))
        if epoch % 1 == 0: 
            costs.append(loss.item())

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
            print(pred)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('Test Accuracy: {}/{} = {:.4f}'.format(
        correct, len(data_loader.dataset), 1. * correct / len(data_loader.dataset)))


if __name__ == '__main__':
    args = parse_args()
    main(args)