from __future__ import division
import warnings
import torch.nn as nn
from torch.distributions import Bernoulli
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import dataset
import math
from utils import save_checkpoint, setup_seed
import torch
import os
import logging
import nni
import numpy as np
from nni.utils import merge_parameter
from config import return_args, args
from Networks.TransCrowd import base_patch16_384_gap_decoder
from Networks.DSFormer import *
from image import load_data

warnings.filterwarnings('ignore')
import time

setup_seed(args.seed)

logger = logging.getLogger('mnist_AutoML')

writer = SummaryWriter('logs_total/logs/x')
# The weights below are calculated according to the count range distribution statistics of datasets.
wts_ucf_10 = [0.0664, 0.0908, 0.0990, 0.1034, 0.1060, 0.1073, 0.1084, 0.1090, 0.1092,0.1006]
wts_ucf =  [0.0092, 0.1055, 0.1091, 0.1105, 0.1107, 0.1109, 0.1110, 0.1111, 0.1111, 0.1111]
wts_SHHA = [0.0341, 0.0913, 0.1036, 0.1080, 0.1098, 0.1104, 0.1105, 0.1104, 0.1109, 0.1109]
wts_SHHB = [0.0237, 0.0973, 0.1054, 0.1084, 0.1104, 0.1108, 0.1110, 0.1110, 0.1110, 0.1111]
wts_SHHB_20 = [0.0855, 0.0914, 0.1065, 0.1046, 0.1073, 0.1090, 0.1079, 0.1094, 0.1085,0.0700]
def main(args):
    if args['dataset'] == 'ShanghaiA':
        train_file = './npydata/ShanghaiA_train.npy'
        test_file = './npydata/ShanghaiA_test.npy'
    elif args['dataset'] == 'ShanghaiB':
        train_file = './npydata/ShanghaiB_train.npy'
        test_file = './npydata/ShanghaiB_test.npy'
    elif args['dataset'] == 'UCF_QNRF':
        train_file = './npydata/ucf_qnrf_train.npy'
        test_file = './npydata/ucf_qnrf_test.npy'
    elif args['dataset'] == 'JHU':
        train_file = './npydata/jhu_train.npy'
        test_file = './npydata/jhu_val.npy'
    elif args['dataset'] == 'NWPU':
        train_file = './npydata/nwpu_train.npy'
        test_file = './npydata/nwpu_val.npy'

    with open(train_file, 'rb') as outfile:
        train_list = np.load(outfile).tolist()
    with open(test_file, 'rb') as outfile:
        val_list = np.load(outfile).tolist()

    print(len(train_list), len(val_list))

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_id']

    model = model_DSFormer(pretrained=True)
    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()
    log = open('logs_total/logs/x.txt', mode="w", encoding="utf-8")
    print(model, file=log)
    log.close()
    criterion = nn.SmoothL1Loss(size_average=False).cuda()
    density_classify_loss = nn.BCELoss(weight=torch.tensor(wts_SHHA).cuda()).cuda()
    optimizer = torch.optim.Adam(
        [  #
            {'params': model.parameters(), 'lr': args['lr']},
        ], lr=args['lr'], weight_decay=args['weight_decay'])

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80], gamma=0.1, last_epoch=-1)

    print(args['pre'])

    # args['save_path'] = args['save_path'] + str(args['rdt'])
    print(args['save_path'])
    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])

    if args['pre']:
        if os.path.isfile(args['pre']):
            print("=> loading checkpoint '{}'".format(args['pre']))
            checkpoint = torch.load(args['pre'])
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            args['start_epoch'] = checkpoint['epoch']
            args['best_pred'] = checkpoint['best_prec1']
        else:
            print("=> no checkpoint found at '{}'".format(args['pre']))

    torch.set_num_threads(args['workers'])

    print(args['best_pred'], args['start_epoch'])
    train_data = pre_data(train_list, args, train=True)
    test_data = pre_data(val_list, args, train=False)

    for epoch in range(args['start_epoch'], args['epochs']):
        start = time.time()
        train(train_data, model, criterion, density_classify_loss, optimizer, epoch, args, scheduler)
        writer.add_scalar('LR', scalar_value=args['lr'], global_step=epoch)
        end1 = time.time()

        if epoch % 5 == 0 and epoch >= 10:
            prec1 = validate(test_data, model, args)
            end2 = time.time()
            is_best = prec1 < args['best_pred']
            args['best_pred'] = min(prec1, args['best_pred'])
            writer.add_scalar('MAE', scalar_value=prec1, global_step=epoch)
            print(' * best MAE {mae:.3f} '.format(mae=args['best_pred']), args['save_path'], end1 - start, end2 - end1)

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args['pre'],
                'state_dict': model.state_dict(),
                'best_prec1': args['best_pred'],
                'optimizer': optimizer.state_dict(),
            }, is_best, args['save_path'])
    writer.close()


def pre_data(train_list, args, train):
    print("Pre_load dataset ......")
    data_keys = {}
    count = 0
    for j in range(len(train_list)):
        Img_path = train_list[j]
        fname = os.path.basename(Img_path)
        img, gt_count = load_data(Img_path, args, train)

        blob = {}
        blob['img'] = img
        blob['gt_count'] = gt_count
        blob['fname'] = fname
        data_keys[count] = blob
        count += 1

        '''for debug'''
        # if j> 10:
        #     break
    return data_keys


def train(Pre_data, model, criterion, density_classify_loss, optimizer, epoch, args, scheduler):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args['save_path'],
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),

                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                            ]),
                            train=True,
                            batch_size=args['batch_size'],
                            num_workers=args['workers'],
                            args=args),
        batch_size=args['batch_size'], drop_last=False)
    args['lr'] = optimizer.param_groups[0]['lr']
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args['lr']))

    model.train()
    end = time.time()

    for i, (fname, img, gt_count) in enumerate(train_loader):

        data_time.update(time.time() - end)
        img = img.cuda()
        out1, out2 = model(img)  # out1 人数 out2 密度
        gt_count = gt_count.type(torch.FloatTensor).cuda().unsqueeze(1)
        gt_density_level = np.array(density_level_classify(gt_count.cpu())).astype(float)
        gt_density_level = torch.from_numpy(gt_density_level).type(torch.FloatTensor).cuda()
        # print(out1.shape, kpoint.shape)

        loss1 = criterion(out1, gt_count)
        out2 = torch.sigmoid(out2)  # sigmoid softmax
        loss2 = density_classify_loss(out2, gt_density_level.type(torch.FloatTensor).cuda())
        loss2_lamda = 0.001
        loss = loss1 + loss2_lamda * loss2
        # out1 = model(img)
        # gt_count = gt_count.type(torch.FloatTensor).cuda().unsqueeze(1)
        # loss = criterion(out1, gt_count)
        losses.update(loss.item(), img.size(0))

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args['print_freq'] == 0:
            print('4_Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
    writer.add_scalar('Loss1 Variation', scalar_value=loss1.item(), global_step=epoch)
    writer.add_scalar('Loss2 Variation', scalar_value=loss2.item(), global_step=epoch)
    writer.add_scalar('Loss Variation', scalar_value=losses.avg, global_step=epoch)
    scheduler.step()


def validate(Pre_data, model, args):
    print('begin test')
    batch_size = 1
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args['save_path'],
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),

                            ]),
                            args=args, train=False),
        batch_size=1)

    model.eval()

    mae = 0.0
    mse = 0.0
    for i, (fname, img, gt_count) in enumerate(test_loader):

        img = img.cuda()
        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        with torch.no_grad():

            out1,out2 = model(img)
            count = torch.sum(out1).item()

        gt_count = torch.sum(gt_count).item()
        mae += abs(gt_count - count)
        mse += abs(gt_count - count) * abs(gt_count - count)

        if i % 15 == 0:
            print('{fname} Gt {gt:.2f} Pred {pred}'.format(fname=fname[0], gt=gt_count, pred=count))

    mae = mae * 1.0 / (len(test_loader) * batch_size)
    mse = math.sqrt(mse / (len(test_loader)) * batch_size)

    nni.report_intermediate_result(mae)

    print(' \n* MAE {mae:.3f}\n'.format(mae=mae), '* MSE {mse:.3f}'.format(mse=mse))

    return mae


def density_level_classify(count):
    total_res = []
    for i in range(len(count)):
        res = []
        img_area = 384 * 384
        max_count = 985 / img_area / 10 # This number is calculated by the maximum count of cropped images of corresponding datasets. Related Codes are shown in plotlabels.py.
        t = count[i] / img_area / max_count
        t = int(min(t, 9))
        for j in range(10):
            if t == j:
                res.append(1)
            else:
                res.append(0)
        total_res.append(res)
    return total_res


def get_classifier_weights(self):
    wts = self.count_class_hist
    wts = 1 - wts / (sum(wts))
    wts = wts / sum(wts)
    return wts


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()
    logger.debug(tuner_params)
    params = vars(merge_parameter(return_args, tuner_params))
    print(params)

    main(params)
