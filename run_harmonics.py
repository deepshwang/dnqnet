"""
    Training code for harmonic Wide Residual Networks.

    Licensed under the BSD License [see LICENSE for details].

    Written by Matej Ulicny, based on implementation by Sergey Zagoruyko:
    https://github.com/szagoruyko/wide-residual-networks/tree/master/pytorch
"""

import argparse
import os
import json
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import SGD, Adam
import torch.utils.data
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import torchnet as tnt
from torchnet.engine import Engine
from utils.harmonics_utils import cast, data_parallel, print_tensor_dict
from torch.backends import cudnn
from models.harm_wrn import resnet
from params import *
from dataloader.dataloader import MNISTDataloader, CIFAR10Dataloader
from dataloader.transforms import *
import pdb

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Wide Residual Networks')
# Model options
parser.add_argument('--model', default='resnet', type=str)
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--width', default=1, type=float)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--level', default=None, type=int)
parser.add_argument('--dataset', default='CIFAR10', type=str)
parser.add_argument('--dataroot', default='../data/cifar/', type=str)
parser.add_argument('--dtype', default='float', type=str)
parser.add_argument('--groups', default=1, type=int)
parser.add_argument('--nthread', default=1, type=int)
parser.add_argument('--seed', default=1, type=int)

# Training options
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.1, type=float) # 0.1 for SGD by original author
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--weight_decay', default=0.0005, type=float)
parser.add_argument('--nesterov', action='store_true', default=False)
parser.add_argument('--epoch_step', default='[60,120,160]', type=str,
                    help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--note', default='', type=str)
parser.add_argument('--subset_size', default=-1, type=int, 
                    help='size of the subset, should be 100, 500, 1000 or -1 for full data')
parser.add_argument('--subset_id', default=-1, type=int, 
                    help='1-5 or -1 for full data')

# Device options
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--save', default='./data/saved_models/harmonic', type=str,
                    help='save parameters and logs in this folder')
parser.add_argument('--ngpu', default=1, type=int,
                    help='number of GPUs to use for training')
parser.add_argument('--gpu_id', default='1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')




def main():
    opt = parser.parse_args()
    args = argument_parser()
    print('parsed options:', vars(opt))

    if opt.dataset == 'CIFAR10':
        train_loader = CIFAR10Dataloader(args, 'train', T_CIFAR10)
        test_loader = CIFAR10Dataloader(args, 'test', T_CIFAR10)
        test_loader_rot = CIFAR10Dataloader(args, 'test', T_CIFAR10_ROT)
    elif opt.dataset == 'MNIST':
        train_loader = MNISTDataloader(args, 'train', T_MNIST)
        test_loader = MNISTDataloader(args, 'test', T_MNIST)
        test_loader_rot = MNISTDataloader(args, 'test', T_MNIST_ROT)



    epoch_step = json.loads(opt.epoch_step)
    num_classes = 10
    log_step = 1
    assert opt.subset_size in [100, 500, 1000, -1], 'subset size should be 100, 500, 1000 or -1'
    assert opt.subset_id in [1, 2, 3, 4, 5, -1], 'subset ide should be 1-5 or -1'
    if opt.subset_size in [100, 500, 1000]:
        log_step = 10000 // opt.subset_size

    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    kwargs = {}
    if not opt.level is None:
        kwargs.update({'level': opt.level})
    f, params = resnet(opt.depth, opt.width, num_classes, opt.dropout, **kwargs)
    def create_optimizer(opt, lr):
        print('creating optimizer with lr = ', lr)
        return SGD([v for v in params.values() if v.requires_grad], lr, momentum=0.9, weight_decay=opt.weight_decay, nesterov=opt.nesterov)
        # return Adam([v for v in params.values() if v.requires_grad], lr)


    optimizer = create_optimizer(opt, opt.lr)

    epoch = 0
    if opt.resume != '':
        state_dict = torch.load(opt.resume)
        epoch = state_dict['epoch']
        params_tensors = state_dict['params']
        for k, v in params.items():
            if k in params_tensors:
                v.data.copy_(params_tensors[k])
        optimizer.load_state_dict(state_dict['optimizer'])

    print('\nParameters:')
    print_tensor_dict(params)

    n_parameters = sum(p.numel() for p in params.values() if p.requires_grad)
    print('\nTotal number of parameters:', n_parameters)

    meter_loss = tnt.meter.AverageValueMeter()
    classacc = tnt.meter.ClassErrorMeter(accuracy=True)
    timer_train = tnt.meter.TimeMeter('s')
    timer_test = tnt.meter.TimeMeter('s')

    if not os.path.exists(opt.save):
        os.mkdir(opt.save)

    def h(sample):
        inputs = cast(sample[0], opt.dtype)
        targets = cast(sample[1], 'long')
        y = data_parallel(f, inputs, params, sample[2], list(range(opt.ngpu))).float()
        return F.cross_entropy(y, targets), y

    def log(t, state):
        torch.save(dict(params={k: v for k, v in params.items() if k.find('dct') == -1}, epoch=t['epoch'], 
                   optimizer=state['optimizer'].state_dict()), os.path.join(opt.save, 'model.pt7'))
        z = vars(opt).copy()
        z.update(t)
        with open(os.path.join(opt.save, 'log.txt'), 'a') as flog:
            flog.write('json_stats: ' + json.dumps(z) + '\n')
        print(z)

    def on_sample(state):
        state['sample'].append(state['train'])

    def on_forward(state):
        loss = float(state['loss'])
        classacc.add(state['output'].data, state['sample'][1])
        meter_loss.add(loss)
        if state['train']:
            state['iterator'].set_postfix(loss=loss)

    def on_start(state):
        state['epoch'] = epoch

    def on_start_epoch(state):
        classacc.reset()
        meter_loss.reset()
        timer_train.reset()
        state['iterator'] = tqdm(train_loader, dynamic_ncols=True)
        epoch = state['epoch'] + 1
        if epoch in epoch_step:
            lr = state['optimizer'].param_groups[0]['lr']
            state['optimizer'] = create_optimizer(opt, lr * opt.lr_decay_ratio)

    def on_end_epoch(state):
        if state['epoch'] % log_step == 0:
            train_loss = meter_loss.value()
            train_acc = classacc.value()
            train_time = timer_train.value()
            meter_loss.reset()
            classacc.reset()
            timer_test.reset()

            with torch.no_grad():
                engine.test(h, test_loader)

            test_acc = classacc.value()[0]

            with torch.no_grad():
                engine.test(h, test_loader_rot)

            test_acc_rot = classacc.value()[0]
            print(log({
                "train_loss": train_loss[0],
                "train_acc": train_acc[0],
                "test_loss": meter_loss.value()[0],
                "test_acc": test_acc,
                "epoch": state['epoch'],
                "num_classes": num_classes,
                "n_parameters": n_parameters,
                "train_time": train_time,
                "test_time": timer_test.value(),
            }, state))
            print('==> id: %s (%d/%d), test_acc: \33[91m%.2f\033[0m' %
                  (opt.save, state['epoch'], opt.epochs, test_acc))

            print('==>[ROT] id: %s (%d/%d), test_acc: \33[91m%.2f\033[0m' %
                  (opt.save, state['epoch'], opt.epochs, test_acc_rot))

    #
    engine = Engine()
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_start'] = on_start
    engine.train(h, train_loader, opt.epochs, optimizer)

    correct = 0
    for i, datum in enumerate(test_loader, 0):
        inputs, labels = datum
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')
        output = f(inputs, params, mode=False)
        pred = output.max(1, keepdim=True)[1]  # get the index of the max

        # Consider 6 and 9 as the same class
        if opt.dataset == 'MNIST':
            pred[pred == 6] = 10
            pred[pred == 9] = 10
            labels[labels == 6] = 10
            labels[labels == 9] = 10

        correct += pred.eq(labels.view_as(pred)).sum().item()

    print('\nTest set:  Accuracy: {}/{} ({:.0f}%)\n'.
          format(correct, len(test_loader.dataset),
                 100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    main()
