import os
import gc
gc.enable()
import sys
import math
import json
import time
import random
from glob import glob
from datetime import datetime
import argparse
import shutil


# from collections import OrderedDict
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import multiprocessing
# from sklearn.preprocessing import LabelEncoder

# import torch
# import torchvision
# from torch import Tensor
# from torchvision import transforms
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# from torch.nn.parameter import Parameter
# from torch.optim import lr_scheduler, Adam, SGD
# from torch.utils.data import DataLoader, Dataset
# from torch.utils.data.sampler import SequentialSampler
# from tqdm import tqdm
# import torchvision.models as models

# import sklearn

import warnings
warnings.filterwarnings("ignore")


# import metrics
from utils import load_data, ImageDataset
# from loss_functions import AngularPenaltySMLoss


parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128),only used for train')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

best_gap = 0
args = parser.parse_args()
use_gpu = torch.cuda.is_available()
MIN_SAMPLES_PER_CLASS = 120
BATCH_SIZE = args.batch_size
NUM_WORKERS = multiprocessing.cpu_count()
MAX_STEPS_PER_EPOCH = 15000
NUM_EPOCHS = args.epochs
LOG_FREQ = args.print_freq
NUM_TOP_PREDICTS = 20

def main():
    global_start_time = time.time()
    global args, best_gap

    # Data loading and preprocessing
    # train = pd.read_csv('../../input/gld2020/train1.csv')
    # test = pd.read_csv('../../input/gld2020/sample_submission.csv')
    # train_dir = '../../input/gld2020/train1'
    # test_dir = '../../input/gld2020/test'
    # val = pd.read_csv('../../input/gld2020/val.csv')
    # val_dir = '../../input/gld2020/val'

    train_loader, val_loader, test_loader, label_encoder, num_classes = load_data(train, val, test, train_dir, val_dir, test_dir)

    # Model building
    print('=> Building model...')
    use_gpu = True
    if use_gpu:
        num_classes = 81313
        #loading resneSt 50
        # model = models.resnet50(pretrained=False)
        model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest269')
        model.fc = nn.Linear(model.fc.in_features, num_classes)  
        
        # mkdir a new folder to store the checkpoint and best model
        if not os.path.exists('result'):
            os.makedirs('result',exist_ok = True)
        fdir = 'result/resneSt_50_1'
        if not os.path.exists(fdir):
            os.makedirs(fdir, exist_ok = True)
        
        # #loading resnet50_places365 model 
        # new_state_dict = OrderedDict()
        # model_file = 'resnet50_places365.pth.tar' 
        # checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        # state_dict = checkpoint['state_dict']

        # for k, v in state_dict.items():
        #     name = k[7:] # remove `module.`
        #     new_state_dict[name] = v
        # # load params
        # model.load_state_dict(new_state_dict)
        # print('num_classes',num_classes)
        # model.avg_pool = nn.AdaptiveAvgPool2d(1)
        # model.fc = nn.Linear(model.fc.in_features, 81313)
        model.cuda()

        weights = num_classes * [1/math.log(num_classes)]
        class_weights = torch.FloatTensor(weights).cuda()


        criterion = nn.CrossEntropyLoss(weight = class_weights).cuda()
        # optimizer = adam(model.parameters(), lr=1e-3, betas=(0.9,0.999), eps=1e-3, weight_decay=1e-4)
        optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9, dampening=0, weight_decay=1e-5)

        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*NUM_EPOCHS, eta_min=1e-6)
        

        in_features = 2048
        out_features = num_classes # Number of classes # change krna h

        cosface = AngularPenaltySMLoss(in_features, out_features, loss_type='cosface') # loss_type in ['arcface', 'sphereface', 'cosface']

        adacos = metrics.AdaCos(num_features=in_features, num_classes=out_features, m=0)

    else:
        print('Cuda is not available!')
        return

    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint "{}"'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec = checkpoint['best_gap']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    
    if args.evaluate:
        inference(testloader, model)
        return

    for epoch in range(args.start_epoch, args.epochs):

        # training model
        train_step(train_loader, model, criterion, optimizer, epoch, scheduler)
        PATH = './model.pth'
        torch.save(model, PATH)

        # evaluate on val set
        gap = validate(val_loader, model)
        
        # evaluate on test set
        predicts_gpu, confs_gpu, _ , _ = inference(test_loader, model)

        # remember best GAP and save checkpoint
        is_best = gap > best_gap
        best_gap = max(gap,best_gap)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_gap': best_gap,
            'optimizer': optimizer.state_dict(),
        }, is_best, fdir)

    print('inference mode')
    #create submission file
    generate_submission(test_loader, model, label_encoder)
    print('Done')
    
    

#defining optimizer    
def adam(parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
    if isinstance(betas, str):
        betas = eval(betas)
    return Adam(parameters,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay)

                      
class AverageMeter:
    ''' Computes and stores the average and current value '''
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# training function
def train_step(train_loader, 
          model, 
          criterion, 
          optimizer,
          epoch, 
          lr_scheduler):
    print(f'epoch {epoch}')
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_score = AverageMeter()

    model.train()
    num_steps = min(len(train_loader), MAX_STEPS_PER_EPOCH)

    print(f'total batches: {num_steps}')

    end = time.time()
    lr = None

    for i, data in enumerate(train_loader):
        input_ = data['image']
        target = data['target']
        batch_size, _, _, _ = input_.shape
        
        # compute output
        output = model(input_.cuda())
        loss = criterion(output, target.cuda())
        output = cosface(output,target.cuda(),s=adacos(output))


        # measure GAP and record loss
        confs, predicts = torch.max(output.detach(), dim=1)
        avg_score.update(GAP(predicts, confs, target))
        losses.update(loss.data.item(), input_.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % LOG_FREQ == 0:
            print(f'Train {epoch} [{i}/{num_steps}]\t'
                    f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                    f'GAP {avg_score.val:.4f} ({avg_score.avg:.4f})'
                    + str(lr))

    print(f' * average GAP on train {avg_score.avg:.4f}')

def validate(data_loader, model):
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_score = AverageMeter()

    criterion = nn.CrossEntropyLoss().cuda()
    model.eval()
    num_steps = min(len(data_loader), MAX_STEPS_PER_EPOCH)

    print(f'total batches: {num_steps}')

    end = time.time()
    lr = None

    for i, data in enumerate(data_loader):
        input_ = data['image']
        target = data['target']
        batch_size, _, _, _ = input_.shape
        
        # compute output
        output = model(input_.cuda())
        loss = criterion(output, target.cuda())

        # measure GAP and record loss
        confs, predicts = torch.max(output.detach(), dim=1)
        avg_score.update(GAP(predicts, confs, target))
        losses.update(loss.data.item(), input_.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % LOG_FREQ == 0:
            print(f' Val [{i}/{num_steps}]\t'
                    f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                    f'GAP {avg_score.val:.4f} ({avg_score.avg:.4f})'
                    )

    print(f' * average GAP on val {avg_score.avg:.4f}')
    
    return avg_score.avg

def inference(data_loader, model):
    model.eval()

    activation = nn.Softmax(dim=1)
    all_predicts, all_confs, all_targets = [], [], []

    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader, disable=IN_KERNEL)):
            if data_loader.dataset.mode != 'test':
                input_, target = data
            else:
                input_, target = data, None
            
            input_ = data['image']
            output = model(input_.cuda())
            output = activation(output)

            confs, predicts = torch.topk(output, NUM_TOP_PREDICTS)
            all_confs.append(confs)
            all_predicts.append(predicts)

            if target is not None:
                all_targets.append(target)

    predicts = torch.cat(all_predicts)
    confs = torch.cat(all_confs)
    targets = torch.cat(all_targets) if len(all_targets) else None

    return predicts, confs, targets

def generate_submission(test_loader, model, label_encoder):
    sample_sub = pd.read_csv('../../input/gld2020/sample_submission.csv')

    predicts_gpu, confs_gpu, _ , _ = inference(test_loader, model)
    predicts, confs = predicts_gpu.cpu().numpy(), confs_gpu.cpu().numpy()

    labels = [label_encoder.inverse_transform(pred) for pred in predicts]
    print('labels')
    print(np.array(labels))
    print('confs')
    print(np.array(confs))

    sub = test_loader.dataset.df
    def concat(label: np.ndarray, conf: np.ndarray) -> str:
        return ' '.join([f'{L} {c}' for L, c in zip(label, conf)])
    sub['landmarks'] = [concat(label, conf) for label, conf in zip(labels, confs)] 

    sample_sub = sample_sub.set_index('id')
    sub = sub.set_index('id')
    sample_sub.update(sub)

    sample_sub.to_csv('submission.csv')   


def save_checkpoint(state, is_best, fdir):
    filepath = os.path.join(fdir, 'checkpoint.pth')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(fdir, 'model_best.pth.tar'))

def GAP(predicts: torch.Tensor, confs: torch.Tensor, targets: torch.Tensor) -> float:
    ''' Simplified GAP@1 metric: only one prediction per sample is supported '''
    assert len(predicts.shape) == 1
    assert len(confs.shape) == 1
    assert len(targets.shape) == 1
    assert predicts.shape == confs.shape and confs.shape == targets.shape

    _, indices = torch.sort(confs, descending=True)

    confs = confs.cpu().numpy()
    predicts = predicts[indices].cpu().numpy()
    targets = targets[indices].cpu().numpy()

    x = pd.DataFrame({'pred': predicts, 'conf': confs, 'true': targets})
    x.sort_values('conf', ascending=False, inplace=True, na_position='last')
    x['correct'] = (x.true == x.pred).astype(int)
    x['prec_k'] = x.correct.cumsum() / (np.arange(len(x)) + 1)
    x['term'] = x.prec_k * x.correct
    gap = x.term.sum() / x.true.count()

    return gap

if __name__=='__main__':
    main()