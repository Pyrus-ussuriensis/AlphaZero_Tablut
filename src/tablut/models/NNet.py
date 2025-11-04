import os
import sys
import time
import random

import numpy as np
from tqdm import tqdm

#sys.path.append('../../')
from tablut.utils.utils import *

from tablut.father_class.NeuralNet import NeuralNet
from tablut.models.RandomData import RandomSymDataset

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
import torch.nn.functional as F

from tablut.models.MAZM import AlphaZeroNet_Tablut as onnet
from tablut.utils.smaller_actionspace import *




# 训练超参/网络规模
args = dotdict({
    'lr': 1e-3,          # Adam常见默认；OpenSpiel等小盘面复现多取0.001。:contentReference[oaicite:8]{index=8}
    'dropout': 0.10,     # 0.1–0.3均可；为减少随机性先取0.1（工程折中；部分研究用到0.3）。:contentReference[oaicite:9]{index=9}
    'cuda': torch.cuda.is_available(),
    'num_channels': 128, # 轻量化以提高自博弈吞吐（工程建议）
    "policy_rank": 64,   # 双线性from×to头的嵌入维；32在精度/显存间折中（工程建议）
})


def _round_lr_lambda(i: int) -> float:
    r = i # epoch从0计，轮次从1计
    if r == 0: return 0.5
    if 1 <= r <= 7: return 1.0
    if 8 <= r <= 15: return 0.3
    if 16 <= r <= 21: return 0.1
    return 0.03 # r >= 25

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        if args.cuda:
            self.nnet.cuda()
        self.n = game.n
        small2big, big2small = build_maps(game.n)
        self.small2big, self.big2small = torch.from_numpy(small2big), torch.from_numpy(big2small)

        decay, no_decay = [], []
        for i,p in self.nnet.named_parameters():
            (no_decay if p.ndim==1 or 'bn' in i.lower() else decay).append(p)
        self.optimizer = torch.optim.AdamW(
            [{'params': decay, 'weight_decay':1e-4},
            {'params': no_decay, 'weight_decay':0.0}],
            lr=args.lr, betas=(0.9,0.999)
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=_round_lr_lambda)

    def train(self, examples, batch_size, steps, i):

        self.scheduler.step(i-1)
        # 推断棋盘边长 n（最后两维）
        b0 = examples[0][0]
        B0 = np.asarray(b0, np.float32)
        n = B0.shape[-1]
        perms = action_perms(n)


        ds = RandomSymDataset(examples, n, perms)
        total_samples = steps * batch_size
        sampler = RandomSampler(ds, replacement=True, num_samples=total_samples)
        dl = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                        drop_last=False, pin_memory=True)

        self.nnet.train()
        pi_losses, v_losses = AverageMeter(), AverageMeter()
        t = tqdm(dl, desc='Training Net')
        for boards, target_pis, target_vs in t:
            target_pis = compress_big_to_small_pi(target_pis, self.big2small, self.n)
            boards = boards.float()
            target_pis = target_pis.float()
            target_vs = target_vs.float()

            if args.cuda:
                boards = boards.contiguous().cuda(non_blocking=True)
                target_pis = target_pis.contiguous().cuda(non_blocking=True)
                target_vs = target_vs.contiguous().cuda(non_blocking=True)

            out_pi, out_v = self.nnet(boards)
            l_pi = self.loss_pi(target_pis, out_pi)
            l_v  = self.loss_v (target_vs,  out_v)
            loss = l_pi + l_v

            pi_losses.update(l_pi.item(), boards.size(0))
            v_losses.update(l_v.item(),  boards.size(0))
            t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

            self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        img2d = np.array(board.getImage(), dtype=np.int16)
        img = getNNImage(img2d, board.size, board.time)

        x = torch.from_numpy(img).unsqueeze(0)  # (1,C,H,W)
        if args.cuda:
            x = x.contiguous().cuda()

        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(x)
        
        pi = torch.exp(pi)
        p_big = expand_small_to_big_probs(pi, self.small2big, self.n)
        return p_big[0].cpu().numpy(), v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
