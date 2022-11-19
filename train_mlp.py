import argparse
import os

import torch
import torch.nn as nn
import torchmetrics as tm
import torchvision
import nni
from tensorboardX import SummaryWriter
from torch import nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import trange
from nni.retiarii.evaluator import FunctionalEvaluator
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
import nni.retiarii.strategy as strategy

from utils import AverageMeter

from model_nas import NasNet, MLP

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='dehazeformer-s', type=str, help='model name')
parser.add_argument('--num_workers', default=0, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='RESIDE-IN', type=str, help='dataset name')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0', type=str, help='GPUs used for training')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def train(train_loader, network, criterion, optimizer, scaler):
    losses = AverageMeter()

    torch.cuda.empty_cache()

    network.train()

    for data, label in train_loader:
        data = data.cuda().reshape(-1, 28 * 28)
        label = label.cuda()

        with autocast(args.no_autocast):
            output = network(data)
            loss = criterion(output, label)

        losses.update(loss.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return losses.avg


def valid(val_loader, network):
    avg_acc = tm.Accuracy().cuda()

    torch.cuda.empty_cache()

    network.eval()

    for data, label in val_loader:
        data = data.cuda().reshape(-1, 28 * 28)
        label = label.cuda()

        with torch.no_grad():  # torch.no_grad() may cause warning
            output = network(data)

        avg_acc.update(output, label)

    return avg_acc.compute()


if __name__ == '__main__':
    epochs = 100
    batch_size = 4096
    lr = 0.8
    eval_freq = 1
    save_dir = './saved_models'
    num_workers = 10
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True,
                                               num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # model = NasNet(784, 10).cuda()
    model = MLP(784, 10).cuda()
    # model = NasNet().cuda()
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 1e-2)
    scaler = GradScaler()
    # _no_scheduler
    writer = SummaryWriter(log_dir=f'logs/nni/sgd_bs_{batch_size}_lr_{lr}')
    best_acc = 0.
    for epoch in trange(epochs):
        loss = train(train_loader, model, criterion, optimizer, scaler)

        writer.add_scalar('train_loss', loss, epoch)

        scheduler.step()

        if epoch % eval_freq == 0:
            avg_acc = valid(test_loader, model)

            writer.add_scalar('valid_acc', avg_acc, epoch)

            nni.report_intermediate_result(float(avg_acc.cpu().numpy()))
            if avg_acc > best_acc:
                best_acc = avg_acc
                torch.save({'state_dict': model.state_dict()},
                           os.path.join(save_dir, 'model.pth'))

            writer.add_scalar('best_acc', best_acc, epoch)
    nni.report_final_result(float(best_acc.cpu().numpy()))
    d = 3
