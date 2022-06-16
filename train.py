import time
import os
import numpy as np
import torch
from collections import OrderedDict
import math
from core import loss
# lowest common multiple
def lcm(a,b): return abs(a * b)/math.gcd(a,b) if a and b else 0
from option.options import Options
from data.dataset import SingleDataset
from core.auto_encoder import AutoEncoder
import util.util as util
from util.visualizer import Visualizer

def main(opt):
    ## setup 
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0 # epoch_iter means the iter number at each epoch
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
    else:
        start_epoch, epoch_iter = 1, 0
    opt.print_freq = lcm(opt.print_freq, opt.batchSize)
    if opt.debug:
        opt.display_freq = 2
        opt.print_freq = 2
        opt.save_latest_freq = 2
        opt.save_epoch_freq = 2
        opt.niter = 2
        opt.niter_decay = 2
        opt.max_dataset_size = 20
    ## get data
    dataset = SingleDataset(opt)
    data_loader =  torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, 
                                               shuffle=opt.shuffle_data, num_workers=int(opt.num_worker))
    dataset_size = min(len(dataset), opt.max_dataset_size)
    print('#training images = %d' % dataset_size)
    ## get model and optimizer
    model = AutoEncoder(opt)
    optimizer_En, optimizer_De = model.optimizer_En, model.optimizer_De
    ## get visualizer
    visualizer = Visualizer(opt)
    total_steps = (start_epoch-1) * dataset_size + epoch_iter
    display_delta = total_steps % opt.display_freq # keep display at each of display_freq iter, even continue train.
    print_delta = total_steps % opt.print_freq
    save_delta = total_steps % opt.save_latest_freq
    ## train epoch
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
        for i, data in enumerate(data_loader, start=epoch_iter):
            if total_steps % opt.print_freq == print_delta:
                iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            ############## Forward Pass ######################
            save_fake = total_steps % opt.display_freq == display_delta # whether to collect output images
            losses, generated = model(data['input'], infer=save_fake)
            losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ] # sum each losses
            loss_dict = dict(zip(model.loss_names, losses))
            # calculate final loss scalar
            loss_all = loss_dict.get('G_l1', 0) + loss_dict.get('G_VGG',0)
            ############### Backward Pass ####################
            # update weights
            optimizer_En.zero_grad()
            optimizer_De.zero_grad()
            loss_all.backward()
            optimizer_En.step()
            optimizer_De.step()
            ############## Display results and errors ##########
            ### print out errors
            if total_steps % opt.print_freq == print_delta:
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}            
                t = (time.time() - iter_start_time) / opt.print_freq
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            ### display output images
            if save_fake:
                visuals = OrderedDict([('input_label', util.tensor2label(data['input'][0], 0)),
                                    ('synthesized_image', util.tensor2im(generated.data[0])),
                                    ('real_image', util.tensor2im(data['input'][0]))])
                visualizer.display_current_results(visuals, epoch, total_steps)
            ### save latest model
            if total_steps % opt.save_latest_freq == save_delta:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.save('latest')            
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
            if epoch_iter >= dataset_size:
                break
        # end of epoch
        print('End of epoch %d / %d \t Time Taken: %d sec' %
            (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        ### save model for this epoch
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
            model.save('latest')
            model.save(epoch)
            np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')
        ### linearly decay learning rate after certain iterations
        if epoch > opt.niter:
            model.update_learning_rate()

if __name__ == "__main__":
    opt = Options("train")
    main(opt.get_args())