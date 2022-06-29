import torch
import torch.nn as nn
import functools
import numpy as np
import math
import torch.nn.functional as F
###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def define_Encoder(input_nc, ngf, norm="none"):
    netEn = Encoder(input_nc, ngf, repeat_num=4, norm_type=norm)
    netEn.apply(weights_init)
    netEn.cuda()
    return netEn
    
def define_Decoder(input_nc, output_nc, norm="none"):
    netDe = Decoder(input_nc, output_nc, up_type="deconv", norm_type=norm)
    netDe.apply(weights_init)
    netDe.cuda()
    return netDe

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Generator
##############################################################################       
class ConvBlock(nn.Module):
    """Conv Block with instance normalization."""
    def __init__(self, dim_in, dim_out, kernel_size_, stride_, padding_,norm_type="none"):
        super(ConvBlock, self).__init__()
        activation=nn.ReLU(inplace=True)
        if norm_type == "batch":
            self.main = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size_, stride=stride_, padding=padding_, bias=False),
                nn.BatchNorm2d(dim_out),
                activation)
        elif norm_type == "instance":
            self.main = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size_, stride=stride_, padding=padding_, bias=False),
                nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
                activation)
        else :
            self.main = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size_, stride=stride_, padding=padding_, bias=False),
                activation)

    def forward(self, x):
        return self.main(x)
        
class UpConvBlock(nn.Module):
    """Conv Block with instance normalization."""
    def __init__(self, dim_in, dim_out, kernel_size_, stride_, padding_, up_type, norm_type="none"):
        super(UpConvBlock, self).__init__()
        
        activation=nn.ReLU(inplace=True)
        self.up_type=up_type
        if up_type=="interpolate":
            if norm_type == "batch":
                self.main = nn.Sequential(
                    nn.Conv2d(dim_in, dim_out, 3, 1, 1),
                    nn.BatchNorm2d(dim_out),
                    nn.ReLU(inplace=True))
            elif norm_type == "instance":
                self.main = nn.Sequential(
                    nn.Conv2d(dim_in, dim_out, 3, 1, 1),
                    nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True))
            else:
                self.main = nn.Sequential(
                    nn.Conv2d(dim_in, dim_out, 3, 1, 1),
                    nn.ReLU(inplace=True))
        elif up_type=="deconv":
            if norm_type == "batch":
                self.main = nn.Sequential(
                    nn.ConvTranspose2d(dim_in, dim_out, kernel_size=kernel_size_, stride=stride_, padding=padding_,output_padding=1,bias=False),
                    nn.BatchNorm2d(dim_out),
                    activation)
            elif norm_type == "instance":
                self.main = nn.Sequential(
                    nn.ConvTranspose2d(dim_in, dim_out, kernel_size=kernel_size_, stride=stride_, padding=padding_,output_padding=1,bias=False),
                    nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
                    activation)
            else:
                self.main = nn.Sequential(
                    nn.ConvTranspose2d(dim_in, dim_out, kernel_size=kernel_size_, stride=stride_, padding=padding_,output_padding=1,bias=False),
                    activation)
        else:
            raise ValueError(f"up_type not surpport '{up_type}' type!")

    def forward(self, x):
        if self.up_type=="interpolate":
            x=F.interpolate(x,scale_factor=2, mode='nearest')
        return self.main(x)

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out, norm_type="none"):
        super(ResidualBlock, self).__init__()
        
        activation=nn.ReLU(inplace=True)
        if norm_type=="batch":
            self.main = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(dim_out),
                activation,
                nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(dim_out))
        elif norm_type=="instance":
            self.main = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
                activation,
                nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))
        else:
            self.main = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
                activation,
                nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return x + self.main(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.PReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
        
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
        
class DulAttention(nn.Module):
    def __init__(self,inc):
        super(DulAttention, self).__init__()

        #self.inplanes = 64
        self.conv = nn.Sequential(
            nn.Conv2d(inc, inc, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(inc, inc, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.ca = ChannelAttention(inc)
        self.sa = SpatialAttention()
        self.conv1x1 = nn.Conv2d(inc*2, inc, 1, 1, 0, bias=False)

    def forward(self, x):
        conv_out = self.conv(x)
        x1=self.ca(conv_out)*conv_out
        x2=self.sa(conv_out)*conv_out
        att_out = self.conv1x1(torch.cat([x1, x2], dim=1))
        return att_out+x
    
class GlobalGeneratorSG(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=4, norm_type="none"):
        assert (n_blocks >= 0)
        super(GlobalGeneratorSG, self).__init__()
        #activation = nn.LeakyReLU(0.2)
        activation = nn.ReLU(True)
        self.att=False

        dim_in = ngf
        max_conv_dim = 512
        self.from_rgb = ConvBlock(input_nc, dim_in, kernel_size_=7, stride_=1, padding_=3, norm_type=norm_type)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.res_block = nn.ModuleList()
        self.to_rgb = nn.Conv2d(dim_in, output_nc, kernel_size=7, stride=1, padding=3, bias=False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # down/up-sampling blocks
        repeat_num = n_downsampling
        for num in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ConvBlock(dim_in, dim_out, kernel_size_=4, stride_=2, padding_=1,norm_type=norm_type))
            if num==repeat_num-1:
                self.decode.insert(
                    0, UpConvBlock(dim_out, dim_in, kernel_size_=4, stride_=2, padding_=1,up_type="interpolate",norm_type=norm_type))  # stack-like
            else:
                self.decode.insert(
                    0, UpConvBlock(dim_out*2, dim_in, kernel_size_=4, stride_=2, padding_=1,up_type="interpolate",norm_type=norm_type))  # stack-like
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(n_blocks):
            self.res_block.append(
                ResidualBlock(dim_in=dim_out, dim_out=dim_out,norm_type=norm_type))

    def forward(self, x):
        out = self.from_rgb(x)
        cache = {}
        ind=0
        for block in self.encode:
            ind=ind+1
            out = block(out)
            cache[ind] = out
        for block in self.res_block:
            out = block(out)
        for block in self.decode:
            out = block(out)
            ind = ind - 1
            if ind>0:
                #out = out + cache[ind]
                out=torch.cat((cache[ind], out), dim=1)
        out=self.to_rgb(out)
        out=self.tanh(out)
        if self.att:
            att= self.sigmoid(x)
            out = (out * att + x * (1 - att))
        return out
        
class GlobalGeneratorUnet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9, norm_type="none"):
        conv_dim = ngf
        repeat_num = 4
        assert (n_blocks >= 0)
        super(GlobalGeneratorUnet, self).__init__()
        
        activation = nn.ReLU(True)
        self.att=False
        self.left_conv_start = nn.Sequential(
            nn.Conv2d(input_nc, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True))

        curr_dim = 16
        #self.left_conv_1 = ConvBlock(curr_dim, curr_dim * 2, kernel_size_=4, stride_=2, padding_=1, norm_type=norm_type)
        self.left_dwconv_1 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        #self.att1=DulAttention(curr_dim)
        self.left_conv_1 = ConvBlock(curr_dim, curr_dim * 2, kernel_size_=3, stride_=2, padding_=1, norm_type=norm_type)
        curr_dim = curr_dim * 2
        self.left_dwconv_2 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.att2=DulAttention(curr_dim)
        self.left_conv_2 = ConvBlock(curr_dim, curr_dim * 2, kernel_size_=3, stride_=2, padding_=1, norm_type=norm_type)
        curr_dim = curr_dim * 2
        self.left_dwconv_3 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.att3=DulAttention(curr_dim)
        self.left_conv_3 = ConvBlock(curr_dim, curr_dim * 2, kernel_size_=3, stride_=2, padding_=1, norm_type=norm_type)
        curr_dim = curr_dim * 2
        self.left_dwconv_4_0 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.left_dwconv_4_1 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.att4=DulAttention(curr_dim)
        self.left_conv_4 = ConvBlock(curr_dim, curr_dim, kernel_size_=3, stride_=2, padding_=1, norm_type=norm_type)
        curr_dim = curr_dim

        layers = []
        for i in range(repeat_num):
            layers.append(ResidualBlock(curr_dim, curr_dim, norm_type=norm_type))
        self.res_block = nn.Sequential(*layers)

        # �����Ұ벿������
        self.right_conv_1 = UpConvBlock(curr_dim, curr_dim, kernel_size_=3, stride_=2, padding_=1,up_type="interpolate", norm_type=norm_type)
        curr_dim = curr_dim
        self.right_dwconv_1_0 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_dwconv_1_1 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_dwconv_1_2 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_conv_2 = UpConvBlock(curr_dim, curr_dim // 2, kernel_size_=3, stride_=2, padding_=1,up_type="interpolate", norm_type=norm_type)
        curr_dim = curr_dim // 2
        self.right_dwconv_2_0 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_dwconv_2_1 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_dwconv_2_2 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_conv_3 = UpConvBlock(curr_dim, curr_dim // 2, kernel_size_=3, stride_=2, padding_=1,up_type="interpolate", norm_type=norm_type)
        curr_dim = curr_dim // 2
        self.right_dwconv_3_0 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_dwconv_3_1 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_dwconv_3_2 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_conv_4 = UpConvBlock(curr_dim, 16, kernel_size_=3, stride_=2, padding_=1,up_type="interpolate", norm_type=norm_type)
        curr_dim = 16
        self.right_dwconv_4_0 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_dwconv_4_1 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_dwconv_4_2 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)

        self.right_conv_end = nn.Conv2d(curr_dim, output_nc, kernel_size=3, stride=1, padding=1, bias=False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.

        # 1�����б������
        feature_0 = self.left_conv_start(x)
        dw_feature_1 = self.left_dwconv_1(feature_0)
        feature_1 = self.left_conv_1(dw_feature_1)
        dw_feature_2 = self.left_dwconv_2(feature_1)
        feature_2 = self.left_conv_2(dw_feature_2)
        dw_feature_3 = self.left_dwconv_3(feature_2)
        feature_3 = self.left_conv_3(dw_feature_3)
        dw_feature_4 = self.left_dwconv_4_0(feature_3)
        dw_feature_4 = self.left_dwconv_4_1(dw_feature_4)
        feature_4 = self.left_conv_4(dw_feature_4)
        #feature_5 = self.left_conv_5(feature_4)
        #feature_6 = self.left_conv_6(feature_5)

        feature_res = self.res_block(feature_4)
        # 2�����н������
        de_feature_1 = self.right_conv_1(feature_res)
        #temp = torch.cat((self.att4(dw_feature_4), de_feature_1), dim=1)
        temp = self.att4(dw_feature_4) + de_feature_1
        temp = self.right_dwconv_1_0(temp)
        temp = self.right_dwconv_1_1(temp)
        temp = self.right_dwconv_1_2(temp)
        de_feature_2 = self.right_conv_2(temp)
        #temp = torch.cat((self.att3(dw_feature_3), de_feature_2), dim=1)
        temp = self.att3(dw_feature_3) + de_feature_2
        temp = self.right_dwconv_2_0(temp)
        temp = self.right_dwconv_2_1(temp)
        temp = self.right_dwconv_2_2(temp)
        de_feature_3 = self.right_conv_3(temp)
        #temp = torch.cat((self.att2(dw_feature_2), de_feature_3), dim=1)
        temp = self.att2(dw_feature_2) + de_feature_3
        temp = self.right_dwconv_3_0(temp)
        temp = self.right_dwconv_3_1(temp)
        temp = self.right_dwconv_3_2(temp)
        de_feature_4 = self.right_conv_4(temp)
        #temp = torch.cat((dw_feature_1, de_feature_4), dim=1)
        temp = dw_feature_1 + de_feature_4
        temp = self.right_dwconv_4_0(temp)
        temp = self.right_dwconv_4_1(temp)
        temp = self.right_dwconv_4_2(temp)

        out_ = self.right_conv_end(temp)
        out = self.tanh(out_)
        if self.att:
            att = self.sigmoid(x)
            out = (out * att + x * (1 - att))
        return out

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()        
        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
            
    def forward(self, input):
        return self.model(input)             

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
    
class Encoder(nn.Module):
    '''
        This is a encoder from GlobalGeneratorUnet
    '''
    def __init__(self, input_nc, ngf=16, repeat_num=4, norm_type="none"):
        super(Encoder, self).__init__()
        self.left_conv_start = nn.Sequential(
            nn.Conv2d(input_nc, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True))
        curr_dim = ngf
        self.left_dwconv_1 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.left_conv_1 = ConvBlock(curr_dim, curr_dim * 2, kernel_size_=3, stride_=2, padding_=1, norm_type=norm_type)
        curr_dim = curr_dim * 2
        self.left_dwconv_2 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.left_conv_2 = ConvBlock(curr_dim, curr_dim * 2, kernel_size_=3, stride_=2, padding_=1, norm_type=norm_type)
        curr_dim = curr_dim * 2
        self.left_dwconv_3 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.left_conv_3 = ConvBlock(curr_dim, curr_dim * 2, kernel_size_=3, stride_=2, padding_=1, norm_type=norm_type)
        curr_dim = curr_dim * 2
        self.left_dwconv_4_0 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.left_dwconv_4_1 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.left_conv_4 = ConvBlock(curr_dim, curr_dim, kernel_size_=3, stride_=2, padding_=1, norm_type=norm_type)
        layers = []
        for _ in range(repeat_num):
            layers.append(ResidualBlock(curr_dim, curr_dim, norm_type=norm_type))
        self.res_block = nn.Sequential(*layers)
        self.output_nc = curr_dim
    
    def get_output_nc(self):
        return self.output_nc

    def forward(self, x):
        feature_0 = self.left_conv_start(x)
        dw_feature_1 = self.left_dwconv_1(feature_0)
        feature_1 = self.left_conv_1(dw_feature_1)
        dw_feature_2 = self.left_dwconv_2(feature_1)
        feature_2 = self.left_conv_2(dw_feature_2)
        dw_feature_3 = self.left_dwconv_3(feature_2)
        feature_3 = self.left_conv_3(dw_feature_3)
        dw_feature_4 = self.left_dwconv_4_0(feature_3)
        dw_feature_4 = self.left_dwconv_4_1(dw_feature_4)
        feature_4 = self.left_conv_4(dw_feature_4)
        feature_res = self.res_block(feature_4)
        return feature_res

class Decoder(nn.Module):
    '''
        This is a encoder from GlobalGeneratorUnet
    '''
    def __init__(self, input_nc, output_nc, up_type = "deconv", norm_type="none"):
        super(Decoder, self).__init__()
        curr_dim = input_nc
        self.right_conv_1 = UpConvBlock(curr_dim, curr_dim, kernel_size_=3, stride_=2, padding_=1, up_type=up_type, norm_type=norm_type)
        curr_dim = curr_dim
        self.right_dwconv_1_0 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_dwconv_1_1 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_dwconv_1_2 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_conv_2 = UpConvBlock(curr_dim, curr_dim // 2, kernel_size_=3, stride_=2, padding_=1,up_type=up_type, norm_type=norm_type)
        curr_dim = curr_dim // 2
        self.right_dwconv_2_0 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_dwconv_2_1 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_dwconv_2_2 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_conv_3 = UpConvBlock(curr_dim, curr_dim // 2, kernel_size_=3, stride_=2, padding_=1,up_type=up_type, norm_type=norm_type)
        curr_dim = curr_dim // 2
        self.right_dwconv_3_0 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_dwconv_3_1 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_dwconv_3_2 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_conv_4 = UpConvBlock(curr_dim, 16, kernel_size_=3, stride_=2, padding_=1,up_type=up_type, norm_type=norm_type)
        curr_dim = 16
        self.right_dwconv_4_0 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_dwconv_4_1 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_dwconv_4_2 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_conv_end = nn.Conv2d(curr_dim, output_nc, kernel_size=3, stride=1, padding=1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        de_feature_1 = self.right_conv_1(x)
        temp = de_feature_1
        temp = self.right_dwconv_1_0(temp)
        temp = self.right_dwconv_1_1(temp)
        temp = self.right_dwconv_1_2(temp)
        de_feature_2 = self.right_conv_2(temp)
        temp = de_feature_2
        temp = self.right_dwconv_2_0(temp)
        temp = self.right_dwconv_2_1(temp)
        temp = self.right_dwconv_2_2(temp)
        de_feature_3 = self.right_conv_3(temp)
        temp = de_feature_3
        temp = self.right_dwconv_3_0(temp)
        temp = self.right_dwconv_3_1(temp)
        temp = self.right_dwconv_3_2(temp)
        de_feature_4 = self.right_conv_4(temp)
        temp = de_feature_4
        temp = self.right_dwconv_4_0(temp)
        temp = self.right_dwconv_4_1(temp)
        temp = self.right_dwconv_4_2(temp)
        out_ = self.right_conv_end(temp)
        out = self.tanh(out_)
        return out