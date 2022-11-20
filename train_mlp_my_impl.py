import pygraphviz as pgv

import gol

gol.init()
TENSOR_GRAPH_ATTR = dict(fontname="Times-Roman", fontsize=14, shape="polygon",
                         style="rounded", color="black",
                         fixedsize=False)
PARA_GRAPH_ATTR = dict(fontname="Times-Roman", fontsize=14, shape="polygon",
                       style="filled", fillcolor="#B4E7B7",
                       fixedsize=False)

OP_GRAPH_ATTR = dict(fontname="Times-Roman", fontsize=14, shape="circle",
                     style="filled", fillcolor="#BBE4FF",
                     fixedsize=False)

G = pgv.AGraph(directed=True, rankdir="RL", overlap=False, normlized=True,
               encoding='UTF-8')
TENSOR_DICT = {}
PARA_DICT = {}
OP_DICT = {}
# GRAPH_FLAG = True

gol.set_value('TENSOR_GRAPH_ATTR', TENSOR_GRAPH_ATTR)
gol.set_value('PARA_GRAPH_ATTR', PARA_GRAPH_ATTR)
gol.set_value('OP_GRAPH_ATTR', OP_GRAPH_ATTR)
gol.set_value('G', G)
gol.set_value('TENSOR_DICT', TENSOR_DICT)
gol.set_value('PARA_DICT', PARA_DICT)
gol.set_value('OP_DICT', OP_DICT)
gol.set_value('GRAPH_FLAG', True)

import argparse
import os

import numpy as np
import nni
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import trange
import np_autograd as anp
from dataloader import Mnist

from utils import AverageMeter

from model_mlp_my_impl import MLP

import sys

sys.setrecursionlimit(3000)


def train(train_loader: DataLoader, network: anp.Layer, criterion: anp.Layer, optimizer: anp.Optimizer) -> float:
    # 损失函数平均器
    losses = AverageMeter()

    # 模型设置为训练模式
    network.train()
    # 迭代训练集
    for data, label in train_loader:
        # 将数据转换为张量
        data = anp.Tensor(data.reshape(-1, 28 * 28).numpy(), requires_grad=True, name='data')
        label = anp.Tensor(label.numpy(), requires_grad=True, name='label')

        # 前向传播
        output = network(data)
        loss = criterion(output, label)
        # 更新损失函数平均器
        losses.update(loss.data.mean())
        if gol.get_value('GRAPH_FLAG'):
            G.layout()
            G.draw('graph.png')

        # 反向传播
        loss.backward(np.ones_like(loss.data))

        # for k, v in TENSOR_DICT.items():
        #     if v.grad is not None:
        #         G.add_node(f'{v.id}_grad', label=f'grad: {v.grad}', **TENSOR_GRAPH_ATTR)
        #         G.add_edge(v.id, f'{v.id}_grad', label=f'grad', **TENSOR_GRAPH_ATTR)
        #         # G.add_edge(v.id, f'{v.id}_grad')

        # 调用优化器，更新模型参数
        optimizer.step()
        # 清空梯度
        optimizer.zero_grad()

        gol.set_value('GRAPH_FLAG', False)

    # print(f'losses: {losses.avg}')
    # 返回损失函数平均值
    return losses.avg


def valid(val_loader:DataLoader, network: anp.Layer) -> float:
    avg_acc = AverageMeter()

    network.eval()

    for data, label in val_loader:
        data = anp.Tensor(data.reshape(-1, 28 * 28).numpy())
        label = anp.Tensor(label.numpy())

        output = network(data)

        acc = np.mean(np.argmax(output, axis=-1) == label)

        avg_acc.update(acc)

    return avg_acc.avg


if __name__ == '__main__':
    # 定义超参数
    epochs = 100
    batch_size = 1024
    train_size = 1024
    test_size = 256
    lr = 0.001
    # momentum = 0.2
    eval_freq = 1
    # save_dir = './saved_models'
    num_workers = 0

    # 定义数据集
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # transform = transforms.Compose([transforms.ToTensor()])
    # train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_set = Mnist(root='./data', train=True, download=True, transform=transform, size=train_size)
    test_set = Mnist(root='./data', train=False, download=True, transform=transform, size=test_size)

    # 定义数据加载器
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=test_size, shuffle=True, num_workers=num_workers)

    # 定义模型
    model = MLP(784, 10)
    print(model.__repr__())

    # 定义损失函数
    criterion = anp.CrossEntropyLoss()
    # optimizer = anp.SGD(list(model.parameters()), lr=lr, momentum=momentum)

    # 定义优化器
    optimizer = anp.Adam(list(model.parameters()), lr=lr)

    # 定义训练过程
    writer = SummaryWriter(log_dir=f'logs/my_impl/sgd_bs_{batch_size}_lr_{lr}')
    best_acc = 0.
    for epoch in trange(epochs):
        # 训练
        loss = train(train_loader, model, criterion, optimizer)

        writer.add_scalar('train_loss', loss, epoch)

        # 评估
        if epoch % eval_freq == 0:
            avg_acc = valid(test_loader, model)

            writer.add_scalar('valid_acc', avg_acc, epoch)

            nni.report_intermediate_result(float(avg_acc))
            # 更新最优精度
            if avg_acc > best_acc:
                best_acc = avg_acc

            writer.add_scalar('best_acc', best_acc, epoch)
    nni.report_final_result(float(best_acc))
