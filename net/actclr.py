import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class
import copy
import random

trunk_ori_index = [4, 3, 21, 2, 1]
left_hand_ori_index = [9, 10, 11, 12, 24, 25]
right_hand_ori_index = [5, 6, 7, 8, 22, 23]
left_leg_ori_index = [17, 18, 19, 20]
right_leg_ori_index = [13, 14, 15, 16]

trunk = [i - 1 for i in trunk_ori_index]
left_hand = [i - 1 for i in left_hand_ori_index]
right_hand = [i - 1 for i in right_hand_ori_index]
left_leg = [i - 1 for i in left_leg_ori_index]
right_leg = [i - 1 for i in right_leg_ori_index]
body_parts = [trunk, left_hand, right_hand, left_leg, right_leg]

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def kld(inputs, targets):
        inputs = F.log_softmax(inputs / 0.1, dim=1)
        targets = F.softmax(targets / 0.04, dim=1)
        return F.kl_div(inputs, targets, reduction='batchmean')

class actclr(nn.Module):
    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, queue_size=32768,
                 momentum=0.999, Temperature=0.07, mlp=True, in_channels=3, hidden_channels=64,
                 hidden_dim=256, num_class=60, dropout=0.5, mode='joint',
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, **kwargs):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__()
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain

        if not self.pretrain:
            self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
        else:
            self.K = queue_size
            self.m = momentum
            self.T = Temperature
            
            self.mode = mode

            self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.encoder_k = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)

            if mlp:  # hack: brute-force replacement
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(nn.Conv2d(dim_mlp, dim_mlp, kernel_size=1),
                                    nn.ReLU(),
                                    self.encoder_q.fc)
                self.encoder_k.fc = nn.Sequential(nn.Conv2d(dim_mlp, dim_mlp, kernel_size=1),
                                                  nn.ReLU(),
                                                  self.encoder_k.fc)

            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

            # create the queue
            self.register_buffer("queue", torch.randn(feature_dim, queue_size))
            self.queue = F.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0  # for simplicity
        self.queue_ptr[0] = (self.queue_ptr[0] + batch_size) % self.K


    @torch.no_grad()
    def ske_swap(self, x, epoch=None):
        '''
        swap a batch skeleton
        T   64 --> 32 --> 16    # 8n
        S   25 --> 25 --> 25 (5 parts)
        '''
        if epoch is None:
            spa_l, spa_u, tem_l, tem_u = 2, 3, 3, 5
        else:
            if epoch < 200: 
                spa_l, spa_u, tem_l, tem_u = 2, 3, 3, 5
            else:
                spa_l, spa_u, tem_l, tem_u = 1, 2, 1, 3
        
        N, C, T, V, M = x.size()
        tem_downsample_ratio = 4

        # generate swap swap idx
        idx = torch.arange(N)
        n = torch.randint(1, N - 1, (1,))
        randidx = (idx + n) % N

        # ------ Spatial ------ #
        Cs = random.randint(spa_l, spa_u)
        # sample the parts index
        parts_idx = random.sample(body_parts, Cs)
        # generate spa_idx
        spa_idx = []
        for part_idx in parts_idx:
            spa_idx += part_idx
        spa_idx.sort()

        # ------ Temporal ------ #
        Ct = random.randint(tem_l, tem_u)
        tem_idx = random.randint(0, T // tem_downsample_ratio - Ct)
        rt = Ct * tem_downsample_ratio

        xst = x.clone()

        p = random.random()

        if p > 0.5:
            # begin swap
            xst[:, :, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :] = \
                xst[randidx][:, :, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :]
            # generate mask
            mask = torch.zeros((T + tem_downsample_ratio - 1) // tem_downsample_ratio, V).cuda()
            mask[tem_idx:tem_idx + Ct, spa_idx] = 1
        elif p <= 0.5 and p > 0.25:
            N, C, T, V, M = xst.size()
            xst_temp = xst[:, :, : 2 * rt]
            xst_temp = xst_temp.permute(0, 4, 3, 1, 2).contiguous()
            xst_temp = xst_temp.view(N * M, V * C, -1)
            xst_temp = torch.nn.functional.interpolate(xst_temp, size=rt)
            xst_temp = xst_temp.view(N, M, V, C, rt)
            xst_temp = xst_temp.permute(0, 3, 4, 2, 1).contiguous()
            xst[:, :, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :] = \
                    xst_temp[randidx][:, :, :, spa_idx, :]
            mask_c = torch.zeros((T + tem_downsample_ratio - 1) // tem_downsample_ratio, V).cuda()
            mask_c[tem_idx:tem_idx + Ct, spa_idx] = 1
            mask_p = torch.zeros((T + tem_downsample_ratio - 1) // tem_downsample_ratio, V).cuda()
            mask_p[:, spa_idx] = 1
            mask = (mask_p, mask_c)
        elif p <= 0.25:
            lamb = random.random()
            xst = xst * (1 - lamb) + xst[randidx] * lamb
            mask = torch.zeros((T + tem_downsample_ratio - 1) // tem_downsample_ratio, V).cuda() + lamb

        return randidx, xst, mask
    
    @torch.no_grad()
    def actionlet_swap(self, x, y, mask):
        '''
        swap a batch skeleton
        T   64 --> 32 --> 16    # 8n
        S   25 --> 25 --> 25 (5 parts)
        '''
        N, C, T, V, M = x.size()
        
        mask = nn.functional.interpolate(mask, (T, V), mode='bilinear', align_corners=True)

        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = x.view(N * M, C, T, V)
        
        y = y.permute(0, 4, 1, 2, 3).contiguous()
        y = y.view(N * M, C, T, V)
        
        z = x * mask + y * (1 - mask)
        
        z = z.view(N, M, C, T, V)
        z = z.permute(0, 2, 3, 4, 1).contiguous()
    
        return z

    @torch.no_grad()
    def ske_trans(self, x):
        
        N, C, T, V, M = x.size()

        # generate swap swap idx
        idx = torch.arange(N)
        n = torch.randint(1, N - 1, (1,))
        randidx = (idx + n) % N

        xst = x.clone()
        
        if self.mode == 'joint':
            Bone = [(21, 21), (2, 21), (3, 21), (9, 21), (5, 21), (1, 2), (4, 3), (6, 5), (7, 6), (8, 7),
            (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
            (18, 17), (19, 18), (20, 19), (23, 8), (22, 23), (25, 12), (24, 25)]


            for v1, v2 in Bone:
                xst[:, :, :, v1 - 1, :] = x[:, :, :, v1 - 1, :] - x[:, :, :, v2 - 1, :]

            for v1, v2 in Bone:
                x[:, :, :, v1 - 1, :] = x[:, :, :, v2 - 1, :] + xst[:, :, :, v1 - 1, :] * (torch.sqrt((xst[randidx, :, :, v1 - 1, :] ** 2).sum(dim=(1), keepdim=True) + 1e-5) / torch.sqrt((xst[:, :, :, v1 - 1, :] ** 2).sum(dim=(1), keepdim=True) + 1e-5))
                
            return x
        
        if self.mode == 'motion':
            # begin swap
            xst = (xst - xst.mean(dim=(3), keepdim=True)) / (xst.std(dim=(2), keepdim=True) + 1e-3) * xst[randidx].std(dim=(2), keepdim=True) + xst[randidx].mean(dim=(3), keepdim=True)
            return xst
        
        if self.mode == 'bone':

            xst = xst / torch.sqrt((xst ** 2).sum(dim=(1), keepdim=True) + 1e-5) * torch.sqrt((xst[randidx, :, :, :, :] ** 2).sum(dim=(1), keepdim=True) + 1e-5)
            return xst

    def select_topk(self, q, k):
        k[k > 0.95] = -1
        k_topk, idx_topk = torch.topk(k, 8192, dim=-1)
        q_topk = torch.gather(q, -1, idx_topk)
        return q_topk, k_topk

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]
    
    def _mask_pool(self, x, mask, randint):
        '''
        :param x:       N M C T V
        :param mask:    T V (bool) # paste True
        :return:        p [N C]   c [N C]
        '''
        if len(mask) == 2:
            p = (x[randint] * mask[0]).sum(-1).sum(-1) / (x[randint] + 1e-5).sum(-1).sum(-1)
            p = p.mean(-1).mean(-1)
            c = (x * (1 - mask[1])).sum(-1).sum(-1) / (x.sum(-1).sum(-1) + 1e-5)
            c = c.mean(-1).mean(-1)
            p, c = p /(p + c), c/(p + c)
            return p, c
            
        else:
            p = (x[randint] * mask).sum(-1).sum(-1) / (x[randint] + 1e-5).sum(-1).sum(-1)
            p = p.mean(-1).mean(-1)
            c = (x * (1 - mask)).sum(-1).sum(-1) / (x.sum(-1).sum(-1) + 1e-5)
            c = c.mean(-1).mean(-1)
            p, c = p /(p + c), c/(p + c)
            
            if random.random() < 0.001:
                print(f"p, c:{p,c}")
                print(f"lamb:{mask.mean()}")

            return p, c

    
    def forward(self, im_q_extreme, im_q, im_p=None, im_k=None, epoch=None, iter=None):
        """
        Input:
            im_q: a batch of query sequences
            im_k: a batch of key sequences
            im_q_extreme: a batch of extremely augmented query sequences
        """

        if not self.pretrain:
            return self.encoder_q(im_q)
        

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            im_k, idx_k_unshuffle = self._batch_shuffle_ddp(im_k)
            
            im_k_mean = im_k.mean(dim=0, keepdim=True)
            [k_mean_feat] = self.encoder_k(im_k_mean)
            
            k, ks, _ = self.encoder_k(im_k, key=True, mean_feat=k_mean_feat)  # keys: NxC
            k = k * 0.5 + ks * 0.5
            
            k = F.normalize(k, dim=1)
            k = self._batch_unshuffle_ddp(k, idx_k_unshuffle)
            
            kn = k @ self.queue.clone().detach()
            
            _, _, qm = self.encoder_q(im_q, key=True, mean_feat=k_mean_feat)
            _, _, pm = self.encoder_q(im_p, key=True, mean_feat=k_mean_feat)
            
        N, C, T, V, M = im_q.shape
        [q] = self.encoder_q(self.actionlet_swap(im_q, im_q_extreme, qm))  # NxC
        q = F.normalize(q, dim=1)
        qn = (q @ self.queue.clone().detach()) ** 2
        qn_topk, kn_topk = self.select_topk(qn, kn)
        
        im_pt = self.ske_trans(im_p)
        randidx, im_ps, mask = self.ske_swap(im_pt, epoch)
        [qt] = self.encoder_q(self.actionlet_swap(im_pt, im_ps, pm))
        qt = F.normalize(qt, dim=1)
        qtn = (qt @ self.queue.clone().detach()) ** 2
        qtn_topk, kn_topk = self.select_topk(qtn, kn)
        
        randidx, im_pc, mask = self.ske_swap(im_p)
        p, c = self._mask_pool((pm.view(N, M, -1, (T + 3) // 4, V)/0.1).exp(), mask, randidx)
        kc = k * c[:,None] + k[randidx] * p[:,None]
        kc = F.normalize(kc, dim=1)
        [pc] = self.encoder_q(im_pc)
        pc = F.normalize(pc, dim=1)
        pcn = (pc @ self.queue.clone().detach()) ** 2
        kcn = kc @ self.queue.clone().detach()
        pcn_topk, kcn_topk = self.select_topk(pcn, kcn)

        self._dequeue_and_enqueue(k)
        global_loss = kld(qn_topk, kn_topk) + kld(qtn_topk, kn_topk) + kld(pcn_topk, kcn_topk)
        return global_loss