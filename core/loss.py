'''
Author: Juncfang
Date: 2022-05-30 17:57:22
LastEditTime: 2022-06-02 14:07:30
LastEditors: Juncfang
Description: 
FilePath: /MyPixelConditionGan/core/loss.py
 
'''

from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from math import exp

## GanLoss (naive gan or lsgan)##
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                real_tensor.requires_grad =False
                self.real_label_var = real_tensor
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                fake_tensor.requires_grad = False
                self.fake_label_var = fake_tensor
            target_tensor = self.fake_label_var
        return target_tensor.cuda()

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

## WGAN-GP Loss ##
class WGANGPLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = torch.mean
    
    def gradient_penalty(self, real_data, generated_data, netD):
        batch_size = real_data.size()[0]
        # get data from list which created cause multi scale D and get D feature.
        while isinstance(generated_data, list):
            generated_data = generated_data[-1]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated.requires_grad = True
        interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = netD(interpolated)
        # get data from list which created cause multi scale D and get D feature.
        while isinstance(prob_interpolated, list):
            prob_interpolated = prob_interpolated[-1]

        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return 10 * ((gradients_norm - 1) ** 2).mean()
    
    def forward(self, input, target_is_real):
        # get data from list which created cause multi scale D and get D feature.
        while isinstance(input, list):
            input = input[-1]
        if target_is_real:
            return  -self.loss(input)
        else:
            return self.loss(input)
    
## TVLoss ##
class TVLoss(nn.Module):
    def __init__(self,):
        super(TVLoss,self).__init__()

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return 2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

## VGGLoss ##
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
    
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0] 
        # self.weights = [0, 0, 0, 0, 1.0] 

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

## GradientLoss ##
def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond = torch.abs(error) < clip_delta
    squared_loss = 0.5 * torch.pow(error,2.0)
    linear_loss = clip_delta * (torch.abs(error) - 0.5 * clip_delta)
    return torch.mean(torch.where(cond, squared_loss, linear_loss))

def edge_conv2d(im):
    conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    sobel_kernel = np.repeat(sobel_kernel, 3, axis=1)
    sobel_kernel = np.repeat(sobel_kernel, 3, axis=0)

    conv_op.weight.data = torch.from_numpy(sobel_kernel).cuda()
    # print(conv_op.weight.size())
    # print(conv_op, '\n')

    edge_detect = conv_op(im)
    #print(torch.max(edge_detect))
    #edge_detect = edge_detect.squeeze().detach().numpy()
    return edge_detect

class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.criterion = nn.L1Loss()
        #self.grad = edge_conv2d()
        #self.criterion = huber_loss()

    def forward(self, x, y):
        x_grad,y_grad = edge_conv2d(x).cuda(),edge_conv2d(y).cuda()
        loss = self.criterion(x_grad, y_grad.detach())
        #loss = huber_loss(x_grad, y_grad.detach())
        return loss


## SSIMLoss ##
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)