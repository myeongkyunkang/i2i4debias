import torch

import models.networks as networks
import models.networks.loss as loss
import util
from models import BaseModel


class OursModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        BaseModel.modify_commandline_options(parser, is_train)
        parser.add_argument("--spatial_code_ch", default=8, type=int)
        parser.add_argument("--global_code_ch", default=2048, type=int)
        parser.add_argument("--lambda_R1", default=10.0, type=float)
        parser.add_argument("--lambda_patch_R1", default=1.0, type=float)
        parser.add_argument("--lambda_Spatial", default=100.0, type=float)
        parser.add_argument("--lambda_GAN", default=0.1, type=float)
        parser.add_argument("--lambda_PatchGAN", default=1.0, type=float)
        parser.add_argument("--patch_min_scale", default=1 / 8, type=float)
        parser.add_argument("--patch_max_scale", default=1 / 4, type=float)
        parser.add_argument("--patch_num_crops", default=8, type=int)
        parser.add_argument("--patch_use_aggregation", type=util.str2bool, default=True)
        parser.add_argument('--attn_layers', type=str, default='4,7,9', help='compute spatial loss on which layers')
        return parser

    def initialize(self):
        self.E = networks.create_network(self.opt, self.opt.netE, "encoder")
        self.E2 = networks.create_network(self.opt, self.opt.netE, "encoder")
        self.G = networks.create_network(self.opt, self.opt.netG, "generator")
        if self.opt.lambda_GAN > 0.0:
            self.D = networks.create_network(self.opt, self.opt.netD, "discriminator")
        if self.opt.lambda_PatchGAN > 0.0:
            self.Dpatch = networks.create_network(self.opt, self.opt.netPatchD, "patch_discriminator")

        # Count the iteration count of the discriminator
        # Used for lazy R1 regularization (c.f. Appendix B of StyleGAN2)
        self.register_buffer("num_discriminator_iters", torch.zeros(1, dtype=torch.long))

        self.attn_layers = [int(i) for i in self.opt.attn_layers.split(',')]
        self.self_sim = loss.SpatialCorrelativeLoss(loss_mode='cos', patch_nums=256, norm=False, T=0.07).to(self.device)
        self.netPre = loss.VGG16().to(self.device)
        self.set_requires_grad([self.netPre], False)

        self.normalization = loss.Normalization(self.device)

        if (not self.opt.isTrain) or self.opt.continue_train:
            self.load()

        if self.opt.num_gpus > 0:
            self.to("cuda:0")

    def Spatial_Loss(self, net, src, tgt):
        """given the source and target images to calculate the spatial similarity and dissimilarity loss"""
        n_layers = len(self.attn_layers)
        feats_src = net(src, self.attn_layers, encode_only=True)
        feats_tgt = net(tgt, self.attn_layers, encode_only=True)

        total_loss = 0.0
        for feat_src, feat_tgt in zip(feats_src, feats_tgt):
            loss = self.self_sim.loss(feat_src, feat_tgt)
            total_loss += loss.mean()

        return total_loss / n_layers

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def per_gpu_initialize(self):
        pass

    def compute_image_discriminator_losses(self, real, mix, label):
        if self.opt.lambda_GAN == 0.0:
            return {}

        pred_real = self.D(real, label)
        pred_mix = self.D(mix, label)

        losses = {}
        losses["D_real"] = loss.gan_loss(pred_real, should_be_classified_as_real=True) * self.opt.lambda_GAN
        losses["D_mix"] = loss.gan_loss(pred_mix, should_be_classified_as_real=False) * (0.5 * self.opt.lambda_GAN)

        return losses

    def get_random_crops(self, x, crop_window=None):
        """ Make random crops.
            Corresponds to the yellow and blue random crops of Figure 2.
        """
        crops = util.apply_random_crop(x, self.opt.patch_size, (self.opt.patch_min_scale, self.opt.patch_max_scale), num_crops=self.opt.patch_num_crops)
        return crops

    def compute_patch_discriminator_losses(self, real, mix):
        losses = {}
        real_feat = self.Dpatch.extract_features(self.get_random_crops(real), aggregate=self.opt.patch_use_aggregation)
        target_feat = self.Dpatch.extract_features(self.get_random_crops(real))
        mix_feat = self.Dpatch.extract_features(self.get_random_crops(mix))

        losses["PatchD_real"] = loss.gan_loss(self.Dpatch.discriminate_features(real_feat, target_feat), should_be_classified_as_real=True, ) * self.opt.lambda_PatchGAN

        losses["PatchD_mix"] = loss.gan_loss(self.Dpatch.discriminate_features(real_feat, mix_feat), should_be_classified_as_real=False, ) * self.opt.lambda_PatchGAN

        return losses

    def compute_discriminator_losses(self, real_A, real_B, label_A, label_B):
        self.num_discriminator_iters.add_(1)

        sp, _ = self.E(real_A, label_A)
        _, gl = self.E2(real_B, label_B)
        mix = self.G(sp, gl)

        losses = self.compute_image_discriminator_losses(real_B, mix, label_B)

        if self.opt.lambda_PatchGAN > 0.0:
            patch_losses = self.compute_patch_discriminator_losses(real_B, mix)
            losses.update(patch_losses)

        metrics = {}  # no metrics to report for the Discriminator iteration

        return losses, metrics

    def compute_R1_loss(self, real, label):
        losses = {}
        if self.opt.lambda_R1 > 0.0:
            real.requires_grad_()
            pred_real = self.D(real, label).sum()
            grad_real, = torch.autograd.grad(outputs=pred_real, inputs=[real], create_graph=True, retain_graph=True, )
            grad_real2 = grad_real.pow(2)
            dims = list(range(1, grad_real2.ndim))
            grad_penalty = grad_real2.sum(dims) * (self.opt.lambda_R1 * 0.5)
        else:
            grad_penalty = 0.0

        if self.opt.lambda_patch_R1 > 0.0:
            real_crop = self.get_random_crops(real).detach()
            real_crop.requires_grad_()
            target_crop = self.get_random_crops(real).detach()
            target_crop.requires_grad_()

            real_feat = self.Dpatch.extract_features(real_crop, aggregate=self.opt.patch_use_aggregation)
            target_feat = self.Dpatch.extract_features(target_crop)
            pred_real_patch = self.Dpatch.discriminate_features(real_feat, target_feat).sum()

            grad_real, grad_target = torch.autograd.grad(outputs=pred_real_patch, inputs=[real_crop, target_crop], create_graph=True, retain_graph=True, )

            dims = list(range(1, grad_real.ndim))
            grad_crop_penalty = grad_real.pow(2).sum(dims) + grad_target.pow(2).sum(dims)
            grad_crop_penalty *= (0.5 * self.opt.lambda_patch_R1 * 0.5)
        else:
            grad_crop_penalty = 0.0

        losses["D_R1"] = grad_penalty + grad_crop_penalty

        return losses

    def compute_generator_losses(self, real_A, real_B, label_A, label_B):
        losses, metrics = {}, {}

        sp, _ = self.E(real_A, label_A)
        _, gl = self.E2(real_B, label_B)
        mix = self.G(sp, gl)

        if self.opt.lambda_Spatial > 0.0:
            norm_real_A = self.normalization((real_A + 1) * 0.5)
            norm_mix = self.normalization((mix + 1) * 0.5)

            losses["G_Spatial"] = self.Spatial_Loss(self.netPre, norm_real_A, norm_mix) * self.opt.lambda_Spatial

        if self.opt.lambda_GAN > 0.0:
            losses["G_GAN_mix"] = loss.gan_loss(self.D(mix, label_B), should_be_classified_as_real=True) * (self.opt.lambda_GAN * 1.0)

        if self.opt.lambda_PatchGAN > 0.0:
            real_feat = self.Dpatch.extract_features(self.get_random_crops(real_B), aggregate=self.opt.patch_use_aggregation).detach()
            mix_feat = self.Dpatch.extract_features(self.get_random_crops(mix))

            losses["G_mix"] = loss.gan_loss(self.Dpatch.discriminate_features(real_feat, mix_feat), should_be_classified_as_real=True, ) * self.opt.lambda_PatchGAN

        return losses, metrics

    def get_visuals_for_snapshot(self, real_A, real_B, label_A, label_B):
        if self.opt.isTrain:
            # avoid the overhead of generating too many visuals during training
            real_A = real_A[:2] if self.opt.num_gpus > 1 else real_A[:4]
            real_B = real_B[:2] if self.opt.num_gpus > 1 else real_B[:4]
            if label_A is not None and label_B is not None:
                label_A = label_A[:2] if self.opt.num_gpus > 1 else label_A[:4]
                label_B = label_B[:2] if self.opt.num_gpus > 1 else label_B[:4]

        sp, _ = self.E(real_A, label_A)
        _, gl = self.E2(real_B, label_B)

        layout = util.resize2d_tensor(util.visualize_spatial_code(sp), real_A)
        mix = self.G(sp, gl)

        visuals = {"real_A": real_A, "real_B": real_B, "layout": layout, "mix": mix}

        return visuals

    def encode_sp(self, image, label=None):
        sp, _ = self.E(image, label)
        return sp

    def encode_gl(self, image, label=None):
        _, gl = self.E2(image, label)
        return gl

    def decode(self, spatial_code, global_code):
        return self.G(spatial_code, global_code)

    def get_parameters_for_mode(self, mode):
        if mode == "generator":
            return list(self.G.parameters()) + list(self.E.parameters()) + list(self.E2.parameters())
        elif mode == "discriminator":
            Dparams = []
            if self.opt.lambda_GAN > 0.0:
                Dparams += list(self.D.parameters())
            if self.opt.lambda_PatchGAN > 0.0:
                Dparams += list(self.Dpatch.parameters())
            return Dparams
