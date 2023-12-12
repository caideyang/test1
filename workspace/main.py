from __future__ import print_function

import argparse
import os

from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
"""
本机的环境变量
MASTER_PORT: rank0上机器的一个空闲端口
MASTER_ADDR: rank0机器的地址
WORLD_SIZE: 这里可以指定，在init函数中也可以指定
RANK: 本机的rank，也可以在init函数中指定
BRANCH: 代码提交的分支名，任务生成的模型文件在data/$BRANCH/路径下
"""

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch, writer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            niter = epoch * len(train_loader) + batch_idx
            writer.add_scalar('loss', loss.item(), niter)


def test(args, model, device, test_loader, writer, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\naccuracy={:.4f}\n'.format(float(correct) / len(test_loader.dataset)))
    writer.add_scalar('accuracy', float(correct) / len(test_loader.dataset), epoch)


def should_distribute():
    return dist.is_available() and WORLD_SIZE > 1


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--dir', default='logs', metavar='L',
                        help='directory where summary logs are stored')
    if dist.is_available():
        parser.add_argument('--backend', type=str, help='Distributed backend',
                            choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
                            default=dist.Backend.GLOO)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print('Using CUDA')

    writer = SummaryWriter(args.dir)

    torch.manual_seed(args.seed)

    # 在代码中指定在哪个设备上执行张量和模型操作(cuda则为使用GPU）
    device = torch.device("cuda" if use_cuda else "cpu")

    if should_distribute():
        print('Using distributed PyTorch with {} backend'.format(args.backend))
        dist.init_process_group(backend=args.backend)

    """
    init_process_group(backend=args.backend, init_method="env://")
    默认情况下使用的都是环境变量来进行分布式通信，也就是指定init_method="env://"，这个进程会自动从本机的环境变量中读取
    """
    print("is_distributed--------1>", is_distributed())
    print("use_cuda--------------1>", use_cuda)
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    print("is_distributed--------2>", is_distributed())
    print("use_cuda--------------2>", use_cuda)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = Net().to(device)

    print("is_distributed--------3>", is_distributed())
    print("use_cuda--------------3>", use_cuda)
    # if is_distributed():
    #     Distributor = nn.parallel.DistributedDataParallel if use_cuda \
    #         else nn.parallel.DistributedDataParallelCPU
    #     model = Distributor(model)
    if is_distributed():
        if use_cuda:
            Distributor = nn.parallel.DistributedDataParallel
            print("---->CUDA is used !<-----")
        else:
            Distributor = nn.parallel.DistributedDataParallelCPU
        model = Distributor(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, writer)
        test(args, model, device, test_loader, writer, epoch)

    if args.save_model:
        model_save_path = "../data/%s_mnist_cnn.pt" % os.environ.get('BRANCH', "")
        torch.save(model.state_dict(), model_save_path)
        print("模型保存成功，路径为%s" % model_save_path)


if __name__ == '__main__':
    main()
