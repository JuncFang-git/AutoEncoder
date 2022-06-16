import os
import torch
import torch.nn as nn
import core.networks as networks
from core.loss import GANLoss, WGANGPLoss, SSIMLoss, TVLoss, VGGLoss, GradientLoss

class Model(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.isTrain = opt.isTrain
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.initialize()

    
    def initialize(self):
        ##### define networks
        # Generator network
        self.netG = networks.define_G(self.opt.input_nc, self.opt.output_nc, self.opt.ngf, self.opt.netG,
                                      self.opt.n_downsample_global, self.opt.n_blocks_global, self.opt.norm)
        # Discriminator network
        if self.isTrain:
            use_sigmoid = True if self.opt.gan_type == 'naive' else False # else use lsgan or wgan-gp
            assert(self.opt.gan_type != 'wgan_gp' or self.opt.num_D == 1, "wgan_gp not surpport Multi-scale discriminators!")
            if self.opt.cat_input:
                netD_input_nc =  self.opt.input_nc + self.opt.output_nc
            else:
                netD_input_nc =  self.opt.input_nc
            self.netD = networks.define_D(netD_input_nc, self.opt.ndf, self.opt.n_layers_D, self.opt.norm, use_sigmoid,
                                          self.opt.num_D, not self.opt.no_ganFeat_loss)
        print('---------- Networks initialized -------------')
        # load networks
        if not self.isTrain or self.opt.continue_train or self.opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else self.opt.load_pretrain
            self.load_network(self.netG, 'G', self.opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', self.opt.which_epoch, pretrained_path)
        # set loss functions and optimizers
        if self.isTrain:
            self.old_lr_G = self.opt.lr_G
            self.old_lr_D = self.opt.lr_D
            self.criterionGAN = WGANGPLoss() if self.opt.gan_type == "wgan_gp" else GANLoss(use_lsgan=self.opt.gan_type=='lsgan')
            if not self.opt.no_ganFeat_loss:
                self.criterionFeat = torch.nn.L1Loss()
            if self.opt.use_ssim_loss:
                self.ssim_loss = SSIMLoss()
            if self.opt.use_tv_loss:
                self.tv_loss = TVLoss()
            if not self.opt.no_vgg_loss:             
                self.criterionVGG = VGGLoss()
            if self.opt.use_grad_loss:
                self.criterionGrad = GradientLoss()
            if self.opt.use_l1_loss:
                self.criterionL1 = torch.nn.L1Loss()
            # define loss filter functions
            self.loss_filter = self.init_loss_filter(not self.opt.no_ganFeat_loss, not self.opt.no_vgg_loss, self.opt.use_grad_loss, 
                                                     self.opt.use_ssim_loss, self.opt.use_tv_loss, self.opt.use_l1_loss)
            self.loss_names = self.loss_filter('G_GAN', 'D_real','D_fake', 'D_gp',
                                               'G_GAN_Feat','G_VGG','G_Grad','G_ssim', 'G_tv', 'G_l1') # Names so we can breakout loss
            # initialize optimizers
            # optimizer G
            params = list(self.netG.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr_G, betas=(self.opt.beta1, 0.999))
            # optimizer D                        
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=self.opt.lr_D, betas=(self.opt.beta1, 0.999))
    
    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, save_dir=''):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            if network_label == 'G':
                raise('Generator must exist!')
        else:
            #network.load_state_dict(torch.load(save_path))
            try:
                network.load_state_dict(torch.load(save_path))
            except:   
                pretrained_dict = torch.load(save_path)
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    network.load_state_dict(pretrained_dict)
                    print('Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                except:
                    print('Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v
                    not_initialized = set()
                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])
                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)

    # helper return list to filter the unused loss (name/value).
    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss, use_grad_loss, use_ssim_loss, use_tv_loss, use_l1_loss):
        use_gp_loss = True if self.opt.gan_type == 'wgan_gp' else False
        # flags = (False, False, False, use_gp_loss, use_gan_feat_loss, use_vgg_loss, use_grad_loss, use_ssim_loss, use_tv_loss, use_l1_loss)
        flags = (True, True, True, use_gp_loss, use_gan_feat_loss, use_vgg_loss, use_grad_loss, use_ssim_loss, use_tv_loss, use_l1_loss)
        def loss_filter(g_gan, d_real, d_fake, d_gp, g_gan_feat, g_vgg, g_grad, g_ssim, g_tv, g_l1):
            return [l for (l,f) in zip((g_gan, d_real, d_fake, d_gp, g_gan_feat, g_vgg, g_grad, g_ssim, g_tv, g_l1),flags) if f]
        return loss_filter

    def forward(self, label, image, infer=False):
        # Inputs to cuda
        input_label = label.cuda() 
        real_image = image.cuda()
        # Fake Generation
        fake_image = self.netG.forward(input_label)
        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(input_label, fake_image)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)
        # Real Detection and Loss        
        pred_real = self.discriminate(input_label, real_image)
        loss_D_real = self.criterionGAN(pred_real, True)
        # gradient penalty
        loss_D_gp = 0
        if self.opt.gan_type == "wgan_gp":
            loss_D_gp = self.criterionGAN.gradient_penalty(real_image, fake_image, self.netD)
        # GAN loss (Fake Passability Loss)
        pred_fake = self.discriminate(input_label, fake_image)
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            # loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat
            loss_G_VGG = self.criterionVGG(fake_image, input_label.detach()) * self.opt.lambda_feat
            
        # Gradient loss
        loss_G_Grad = 0
        if self.opt.use_grad_loss:
            loss_G_Grad=self.criterionGrad(fake_image, real_image) * self.opt.lambda_feat
        # SSIM loss
        loss_G_ssim = 0
        if self.opt.use_ssim_loss:
            loss_G_ssim= - self.ssim_loss(fake_image, real_image)
        # TV loss
        loss_G_tv = 0
        if self.opt.use_tv_loss:
            loss_G_tv = self.tv_loss(fake_image)
        # L1 Loss
        loss_G_l1 = 0
        if self.opt.use_l1_loss:
            # loss_G_l1 = self.criterionL1(fake_image, real_image)
            loss_G_l1 = self.criterionL1(fake_image, input_label.detach())
        # Only return the fake_B image if necessary to save BW
        return [ self.loss_filter(loss_G_GAN, loss_D_real, loss_D_fake, loss_D_gp, 
                                  loss_G_GAN_Feat, loss_G_VGG, loss_G_Grad, loss_G_ssim, loss_G_tv, loss_G_l1),
                None if not infer else fake_image ]

    def discriminate(self, input_label, test_image):
        if self.opt.cat_input:
            input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        else:
            input_concat = test_image.detach()
        return self.netD.forward(input_concat)
        
    def inference(self, label):
        input_label =  label.cuda()
        with torch.no_grad():
            fake_image = self.netG.forward(input_label)
        return fake_image

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch)
        self.save_network(self.netD, 'D', which_epoch)

    def update_learning_rate(self):
        lrd_G = self.opt.lr_G / self.opt.niter_decay
        lr_G = self.old_lr_G - lrd_G
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr_G
        lrd_D = self.opt.lr_D / self.opt.niter_decay
        lr_D = self.old_lr_D - lrd_D
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr_D
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr_G = lr_G
        self.old_lr_D = lr_D
    
    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        network.cuda()

class InferenceModel(Model):
    def forward(self, inp):
        return self.inference(inp)

        
