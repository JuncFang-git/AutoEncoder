import os
import torch
import torch.nn as nn
import core.networks as networks
from core.loss import GANLoss, WGANGPLoss, SSIMLoss, TVLoss, VGGLoss, GradientLoss

class AutoEncoder(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.isTrain = opt.isTrain
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.initialize()

    
    def initialize(self):
        ##### define networks
        # Encoder network
        self.netEn = networks.define_Encoder(self.opt.input_nc, self.opt.ngf, self.opt.norm)
        encoder_out_nc = self.netEn.get_output_nc()
        # Decoder network
        self.netDe = networks.define_Decoder(encoder_out_nc, self.opt.output_nc, self.opt.norm)
        print('---------- Networks initialized -------------')
        # load networks
        if not self.isTrain or self.opt.continue_train or self.opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else self.opt.load_pretrain
            self.load_network(self.netEn, 'En', self.opt.which_epoch, pretrained_path)
            self.load_network(self.netDe, 'De', self.opt.which_epoch, pretrained_path)
        # set loss functions and optimizers
        if self.isTrain:
            self.old_lr_En = self.opt.lr_En
            self.old_lr_De = self.opt.lr_De
            if not self.opt.no_vgg_loss:             
                self.criterionVGG = VGGLoss()
            if self.opt.use_l1_loss:
                self.criterionL1 = torch.nn.L1Loss()
            # define loss filter functions
            self.loss_filter = self.init_loss_filter(not self.opt.no_vgg_loss, self.opt.use_l1_loss)
            self.loss_names = self.loss_filter('VGG', 'G_l1') # Names so we can breakout loss
            # initialize optimizers
            # optimizer En
            params = list(self.netEn.parameters())
            self.optimizer_En = torch.optim.Adam(params, lr=self.opt.lr_En, betas=(self.opt.beta1, 0.999))
            # optimizer De                     
            params = list(self.netDe.parameters())
            self.optimizer_De = torch.optim.Adam(params, lr=self.opt.lr_De, betas=(self.opt.beta1, 0.999))
    
    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, save_dir=''):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            raise(f'{network_label} network must exist!')
        else:
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
    def init_loss_filter(self, use_vgg_loss, use_l1_loss):
        flags = (use_vgg_loss, use_l1_loss)
        def loss_filter(vgg, l1):
            return [l for (l,f) in zip((vgg, l1),flags) if f]
        return loss_filter

    def forward(self, image, infer=False):
        # Inputs to cuda
        real_image = image.cuda()
        # Get feature
        feature = self.netEn.forward(real_image)
        # Get reconstruction image
        gen_image = self.netDe.forward(feature)
        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(gen_image, real_image.detach()) * self.opt.lambda_feat
        # L1 Loss
        loss_G_l1 = 0
        if self.opt.use_l1_loss:
            loss_G_l1 = self.criterionL1(gen_image, real_image.detach())
        # Only return the fake_B image if necessary to save BW
        return [ self.loss_filter(loss_G_VGG,loss_G_l1), None if not infer else gen_image ]
        
    def inference(self, image):
        real_image =  image.cuda()
        with torch.no_grad():
            gen_image = self.netDe.forward(self.netEn.forward(real_image))
        return gen_image

    def save(self, which_epoch):
        self.save_network(self.netEn, 'En', which_epoch)
        self.save_network(self.netDe, 'De', which_epoch)

    def update_learning_rate(self):
        lrd_En = self.opt.lr_En / self.opt.niter_decay
        lr_En = self.old_lr_En - lrd_En
        for param_group in self.optimizer_En.param_groups:
            param_group['lr'] = lr_En
        lrd_De = self.opt.lr_De / self.opt.niter_decay
        lr_De = self.old_lr_De - lrd_De
        for param_group in self.optimizer_De.param_groups:
            param_group['lr'] = lr_De
        print('update Encoder learning rate: %f -> %f' % (self.old_lr_En, lr_En))
        print('update Decoder learning rate: %f -> %f' % (self.old_lr_De, lr_De))
        self.old_lr_En = lr_En
        self.old_lr_De = lr_De
    
    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.state_dict(), save_path)
        network.cuda()

class InferenceAutoEncoder(AutoEncoder):
    def forward(self, inp):
        return self.inference(inp)