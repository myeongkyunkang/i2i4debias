import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def gan_loss(pred, should_be_classified_as_real):
    bs = pred.size(0)
    if should_be_classified_as_real:
        return F.softplus(-pred).view(bs, -1).mean(dim=1)
    else:
        return F.softplus(pred).view(bs, -1).mean(dim=1)


class PatchSim(nn.Module):
    def __init__(self, patch_nums=256, norm=True):
        super(PatchSim, self).__init__()
        self.patch_nums = patch_nums
        self.use_norm = norm

    def forward(self, feat, patch_ids=None):
        B, C, W, H = feat.size()
        feat = feat - feat.mean(dim=[-2, -1], keepdim=True)
        feat = F.normalize(feat, dim=1) if self.use_norm else feat / np.sqrt(C)
        query, key, patch_ids = self.select_patch(feat, patch_ids=patch_ids)
        patch_sim = query.bmm(key) if self.use_norm else torch.tanh(query.bmm(key) / 10)
        if patch_ids is not None:
            patch_sim = patch_sim.view(B, len(patch_ids), -1)
        return patch_sim, patch_ids

    def select_patch(self, feat, patch_ids=None):
        B, C, W, H = feat.size()
        if self.patch_nums > 0:
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)  # B*N*C
            if patch_ids is None:
                patch_ids = torch.randperm(feat_reshape.size(1), device=feat.device)
                patch_ids = patch_ids[:int(min(self.patch_nums, patch_ids.size(0)))]
            feat_query = feat_reshape[:, patch_ids, :]  # B*Num*C
            feat_key = feat.reshape(B, C, W * H)
        else:
            feat_query = feat.reshape(B, C, H * W).permute(0, 2, 1)  # B * N (H * W) * C
            feat_key = feat.reshape(B, C, H * W)  # B * C * N (H * W)
        return feat_query, feat_key, patch_ids


class SpatialCorrelativeLoss(nn.Module):
    def __init__(self, loss_mode='cos', patch_nums=256, norm=True, T=0.1):
        super(SpatialCorrelativeLoss, self).__init__()
        self.patch_sim = PatchSim(patch_nums=patch_nums, norm=norm)
        self.patch_nums = patch_nums
        self.norm = norm
        self.loss_mode = loss_mode
        self.T = T
        self.criterion = nn.L1Loss() if norm else nn.SmoothL1Loss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def cal_sim(self, f_src, f_tgt):
        sim_src, patch_ids = self.patch_sim(f_src)
        sim_tgt, _ = self.patch_sim(f_tgt, patch_ids)
        return sim_src, sim_tgt

    def compare_sim(self, sim_src, sim_tgt):
        B, Num, N = sim_src.size()
        tgt_sorted, _ = sim_tgt.sort(dim=-1, descending=True)
        num = int(N / 4)
        src = torch.where(sim_tgt < tgt_sorted[:, :, num:num + 1], 0 * sim_src, sim_src)
        tgt = torch.where(sim_tgt < tgt_sorted[:, :, num:num + 1], 0 * sim_tgt, sim_tgt)
        if self.loss_mode == 'l1':
            loss = self.criterion((N / num) * src, (N / num) * tgt)
        elif self.loss_mode == 'cos':
            sim_pos = F.cosine_similarity(src, tgt, dim=-1)
            loss = self.criterion(torch.ones_like(sim_pos), sim_pos)
        else:
            raise NotImplementedError('padding [%s] is not implemented' % self.loss_mode)
        return loss

    def loss(self, f_src, f_tgt):
        sim_src, sim_tgt = self.cal_sim(f_src, f_tgt)
        loss = self.compare_sim(sim_src, sim_tgt)
        return loss


class Normalization(nn.Module):
    def __init__(self, device):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(23, 26):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(26, 28):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(28, 30):
            self.relu5_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        # for param in self.parameters():
        #    param.requires_grad = False

    def forward(self, x, layers=None, encode_only=False, resize=False):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)

        relu4_1 = self.relu4_1(relu3_3)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)

        relu5_1 = self.relu5_1(relu4_3)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
        }
        if encode_only:
            if len(layers) > 0:
                feats = []
                for layer, key in enumerate(out):
                    if layer in layers:
                        feats.append(out[key])
                return feats
            else:
                return out['relu3_1']
        return out
