'''
Author: Juncfang
Date: 2022-06-17 13:38:37
LastEditTime: 2022-06-17 14:33:47
LastEditors: Juncfang
Description: 
FilePath: /AutoEncoder/test.py
 
'''
import os
import torch

import util.util as util
from collections import OrderedDict
from util.visualizer import Visualizer
from util import html
from option.options import Options
from data.dataset import SingleDataset
from core.auto_encoder import AutoEncoder

opt = Options("test").get_args()
opt.num_worker = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

dataset = SingleDataset(opt)
data_loader =  torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, 
                                               shuffle=opt.shuffle_data, num_workers=int(opt.num_worker))
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test
model = AutoEncoder(opt)
print(model)
for i, data in enumerate(data_loader):
    if i >= opt.how_many:
        break
    generated = model.inference(data['input'])
    visuals = OrderedDict([('input_label', util.tensor2im(data['input'][0])),
                           ('synthesized_image', util.tensor2im(generated.data[0]))])
    print('process image... %s' % data['path'])
    visualizer.save_images(webpage, visuals, data['path'])

webpage.save()
