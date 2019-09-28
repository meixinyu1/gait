import os
import sys
import argparse
import time
import datetime
import numpy as np

import os.path as osp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import *
from torch.optim import lr_scheduler

from losses import CrossEntropyLabelSmooth, TripletLoss
import models
from models import resnet
import data_manager
from img_loader import ImageDataset
from samplers import RandomIdentitySampler
from utils import AverageMeter, save_checkpoint, Logger
from eval_metrics import evaluate
from models import PNnet
parser = argparse.ArgumentParser(description="Using resnet50 train gait model with hard-triplet-loss")
# Dataset
parser.add_argument("-j", "--workers", default=4, type=int, help="number of data loading workers(default: 4)")
parser.add_argument('--stepsize', default=30, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--mark', default=True, type=int,
                    help="if this is True, mark the area of bag")

# optimization options
parser.add_argument('--max-epoch', default=800, type=int,
                    help="maximum epochs to run")
parser.add_argument("--epoch", default=30, type=int, help="epochs to run")
parser.add_argument("--train_batch", default=32, type=int)
parser.add_argument("--test_batch", default=1, type=int, help="has to be 1")
parser.add_argument("--lr", '--learning-rate', default=0.0001, type=float)
parser.add_argument("--weight-decay", default=5e-4, type=float)
parser.add_argument("--margin", type=float, default=0.3, help="margin for triplet loss")
parser.add_argument('--htri-only', action='store_true', default=False,
                    help="if this is True, only htri loss is used in training")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50')
parser.add_argument('--pool', type=str, default='avg', choices=['avg', 'max'])

# Miscs
parser.add_argument('--save-dir', type=str, default='log_N')
parser.add_argument('--print-freq', type=int, default=200, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--evaluate', type=int, default=False, help="evaluation only")
parser.add_argument('--eval-step', type=int, default=2,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--pretrained-model', type=str,default='/home/zhangsx/gait/attention_bag/log_N/best_model.pth.tar')
parser.add_argument('--user', type=str, default='meixinyu')
args = parser.parse_args()

def main():
    print(args)

    torch.manual_seed(args.seed)

    # GPU / CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Initializing dataset")
    dataset = data_manager.init_dataset(args.mark, args.user)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    model = models.PNnet.resnet50(pretrained=True, num_classes=dataset.num_train_pids, user = args.user).to(device=device)
    model = nn.DataParallel(model).cuda()
    # ss
    trainLoader = DataLoader(
        ImageDataset(dataset.train, sample='random', transform=transform),
        sampler=RandomIdentitySampler(dataset.train, num_instances=2),
        batch_size=args.train_batch, num_workers=args.workers
    )

    # test/val queryLoader
    # test/val galleryLoader
    test_probeLoader = DataLoader(
        ImageDataset(dataset.test_probe, sample='dense', transform=transform),
        shuffle=False, batch_size=args.test_batch,drop_last=False
    )

    test_galleryLoader = DataLoader(
        ImageDataset(dataset.test_gallery, sample='dense', transform=transform),
        shuffle=False, batch_size=args.test_batch,drop_last=False
    )

    print("Initializing model: {}".format(args.arch))

    if args.evaluate is True:
        checkpoint = torch.load(args.pretrained_model)
        print(checkpoint['rank1'])
        model.load_state_dict(checkpoint['state_dict'])

    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
    criterion_xent = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, device=device)
    criterion_htri = TripletLoss(margin=args.margin)
    criterion_bag = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr = 0.0005, momentum=0.9, weight_decay=5e-4, nesterov=True)
    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    if args.evaluate is True:
        print("Evaluate only")
        test(model, test_probeLoader, test_galleryLoader, device)
        return

    start_time = time.time()
    best_rank1 = -np.inf
    for epoch in range(0, args.max_epoch):
        # test(model, test_probeLoader, test_galleryLoader, args.pool, device)
        print("==> {}/{}".format(epoch+1, args.max_epoch))
        train(model, criterion_xent, criterion_htri, criterion_bag, optimizer, trainLoader, device)

        if args.stepsize > 0:
            scheduler.step()

        if args.eval_step > 0 and (epoch + 1)%args.eval_step == 0 or (epoch+1) == args.max_epoch:
            if epoch + 1 >= 20:
                args.eval_step = 1
            print("==> Test")
            rank1 = test(model, test_probeLoader, test_galleryLoader, device)
            # continue
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                if best_rank1 > 0.88:
                    state_dict = model.state_dict()
                    save_checkpoint({
                        'state_dict': state_dict,
                        'rank1': rank1,
                        'epoch': epoch,
                    }, is_best, osp.join(args.save_dir, 'type2_layer4_checkpoint_ep' + str(epoch+1) + '.pth.tar'))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def train(model, criterion_xent, criterion_htri, criterion_bag, optimizer, trainLoader, device):
    losses = AverageMeter()

    for batch_idx, (imgs, pids, tags) in enumerate(trainLoader):
        imgs, pids = imgs.to(device=device, dtype=torch.float), pids.to(device=device, dtype=torch.long)
        tags = tags.to(device=device, dtype=torch.float)
        outputs, features, n_loss = model(imgs, tags)

        if args.htri_only:
            loss = criterion_htri(features, pids)
        else:
            # combine hard triplet loss with cross entropy loss
            xent_loss = criterion_xent(outputs, pids)
            htri_loss = criterion_htri(features, pids)
            n,p = n_loss
            # tags = tags.sum(1)
            n = n.view(n.size(0), -1).sum(1)
            n = n / (6-tags.sum(1))
            p = p.view(p.size(0), -1).sum(1)
            if tags.sum() > 0:
                tmp = torch.clamp(n - p + 1, min=0)
                tmp = tmp * tags.sum(1)
                p = tmp.sum() / tags.sum()
            else:
                p = 0

            loss = xent_loss + htri_loss + p

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), pids.size(0))
        if (batch_idx+1) % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx+1, len(trainLoader), losses.val, losses.avg))

def test(model, queryLoader, galleryLoader, device, ranks=[1,5,10,20]):
    with torch.no_grad():
        model.eval()
        qf, q_pids, q_bags = [], [], []
        for batch_idx, (img, pid, bag) in enumerate(queryLoader):
            img = img.to(device=device, dtype=torch.float)
            _, features = model(img)
            features = features.squeeze(0)
            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pid)
            q_bags.extend(bag)
        qf = torch.stack(qf)
        q_pids = np.asarray(q_pids)
        # q_bags = np.asarray(q_bags)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_bags = [], [], []
        for batch_idx, (img, pid, bag) in enumerate(galleryLoader):
            img = img.to(device=device, dtype=torch.float)
            _,features = model(img)
            features = features.squeeze(0)
            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pid)
            g_bags.extend(bag)
        gf = torch.stack(gf)
        g_pids = np.asarray(g_pids)
        # g_bags = np.asarray(g_bags)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
        print("Computing distance matrix")

        m, n = qf.size(0), gf.size(0)

        distmat = torch.pow(qf,2).sum(dim=1, keepdim=True).expand(m,n) + \
                  torch.pow(gf,2).sum(dim=1, keepdim=True).expand(n,m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.numpy()

        cmc = evaluate(distmat, q_pids, g_pids)
        print("Results ----------")
        # print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        print("------------------")

        return cmc[0]

if __name__ == '__main__':
    main()


class Solution(object):
    def solve(self, array):
        tmp = []
        ans = 0
        for i in range(len(array)):
            if len(tmp) == 0:
                tmp.append([i, array[i]])
            else:
                while len(tmp) > 0 and tmp[len(tmp)-1][1] >= array[i]:
                    l = 0
                    if len(tmp) > 1:
                        l = tmp[len(tmp)-2][0]
                    ans = max(ans,tmp[len(tmp)-1][1]*(i-1-l))
                    tmp.pop()
                tmp.append([i, array[i]])
        while len(tmp) > 0:
            l = 0
            if len(tmp) > 1:
                l = tmp[len(tmp)-2][0]
            ans = max(ans, tmp[len(tmp)-1][1] * (len(array)-1 - l))
            tmp.pop()
        return ans
    def maximalRectangle(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        ans = 0
        if len(matrix) == 0:
            return ans
        tmp = list(np.zeros(len(matrix[0])))
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] == 1:
                    tmp[j] = tmp[j] + 1
                else:
                    tmp[j] = 0
            ans = max(ans, self.solve(tmp))
        return ans






















