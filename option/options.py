'''
Author: Juncfang
Date: 2022-05-30 15:59:55
LastEditTime: 2022-07-04 10:22:43
LastEditors: Juncfang
Description: 
FilePath: /AutoEncoder/option/options.py
 
'''
import os
import argparse
from util.util import mkdirs

class Options():
    def __init__(self, phase : str) -> None:
        self.parser = argparse.ArgumentParser()
        self.phase = phase
        
    def get_args(self):
        self._add_base_args()
        if self.phase == "train":
            self._add_train_args()
            args = self.parser.parse_args()
            args.isTrain = True
            args.phase = "train"
        elif self.phase == "test":
            self._add_test_args()
            args = self.parser.parse_args()
            args.isTrain = False
            args.continue_train = False
            args.phase = "test"
        else:
            raise KeyError("Error phase at get_args!")
        # print args
        print('------------ Options -------------')
        for k, v in sorted(vars(args).items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        # save args to the disk        
        expr_dir = os.path.join(args.checkpoints_dir, args.name)
        mkdirs(expr_dir)
        if not args.continue_train:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(vars(args).items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return args
    
    def _add_base_args(self):
        self.parser.add_argument('--name', type=str, default='label2city', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')

        # input/output sizes       
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=1024, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # for setting inputs
        self.parser.add_argument('--dataroot', type=str, default='./datasets/cityscapes/') 
        self.parser.add_argument('--resize_or_crop', type=str, default='scale_width', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--shuffle_data', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')        
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation') 
        self.parser.add_argument('--num_worker', default=0, type=int, help='# threads for loading data')                
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')

        # for generator
        # self.parser.add_argument('--netG', type=str, default='global', help='selects model to use for netG')
        self.parser.add_argument('--ngf', type=int, default=16, help='# of gen filters in first conv layer')
        # self.parser.add_argument('--n_downsample_global', type=int, default=4, help='number of downsampling layers in netG') 
        # self.parser.add_argument('--n_blocks_global', type=int, default=9, help='number of residual blocks in the global generator network')
        # self.parser.add_argument('--n_blocks_local', type=int, default=3, help='number of residual blocks in the local enhancer network')
        # self.parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')   
    
    def _add_train_args(self):
        # for ddp
        self.parser.add_argument('--ddp_mode', action='store_true', help='if use ddp technology to use multi gpu. ddp need at least 2 gpu')
        # for displays
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')        
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')

        # for training
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr_En', type=float, default=0.0002, help='initial learning rate for encoder of adam')
        self.parser.add_argument('--lr_De', type=float, default=0.0002, help='initial learning rate for decoder of adam')

        # for discriminators
        # self.parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to use')
        # self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        # self.parser.add_argument('--cat_input', action="store_true", help='set to cat input and output image as the input of D')
        # self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')    
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')                
        # self.parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        self.parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
        # self.parser.add_argument('--gan_type', type=str, default='lsgan', choices=['naive', 'lsgan', 'wgan_gp'], help='which gan loss to use')
        # self.parser.add_argument('--use_grad_loss', action='store_true', help='if specified, use grad feature matching loss')
        # self.parser.add_argument('--use_ssim_loss', action='store_true', help='if specified, use ssim feature matching loss')
        # self.parser.add_argument('--use_tv_loss', action='store_true', help='if specified, use tv feature matching loss')
        self.parser.add_argument('--use_l1_loss', action='store_true', help='if specified, use l1 feature matching loss')

    def _add_test_args(self):
        self.parser.add_argument('--results_dir', type=str, default='./test_result', help='test result will save here')
        self.parser.add_argument('--how_many', type=int, default=float('inf'), help='how many image will be test')
        