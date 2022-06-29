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
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, rank=opt.rank, num_replicas=opt.world_size)
    data_loader =  torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, 
                                               shuffle=False, num_workers=int(opt.num_worker), pin_memory=True, sampler=train_sampler)
    dataset_size = min(len(dataset), opt.max_dataset_size)
    print('#training images = %d' % dataset_size)
    ## get model and optimizer
    model = AutoEncoder(opt)
    model.cuda()
    model = DDP(model, device_ids=[opt.rank],output_device=opt.rank)
    optimizer_En, optimizer_De = model.module.optimizer_En, model.module.optimizer_De
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
            total_steps += opt.batchSize * opt.world_size
            epoch_iter += opt.batchSize * opt.world_size
            ############## Forward Pass ######################
            save_fake = total_steps % opt.display_freq == display_delta # whether to collect output images
            losses, generated = model(data['input'], infer=save_fake)
            losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ] # sum each losses
            loss_dict = dict(zip(model.module.loss_names, losses))
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
            if opt.rank == 0:
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
                    model.module.save('latest')            
                    np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
            if epoch_iter >= dataset_size:
                print("break")
                break
        if opt.rank == 0:
            # end of epoch
            print('End of epoch %d / %d \t Time Taken: %d sec' %
                (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
            ### save model for this epoch
            if epoch % opt.save_epoch_freq == 0 and opt.rank == 0:
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
                model.module.save('latest')
                model.module.save(epoch)
                np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')
        ### linearly decay learning rate after certain iterations
        if epoch > opt.niter:
            model.module.update_learning_rate()


def setup(rank, world_size):
    torch.cuda.set_device(rank)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12354'
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True   
    torch.backends.cudnn.benchmark = False
    # initialize the process group
    dist.init_process_group("nccl", init_method='env://', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    opt = Options("train").get_args()
    opt.rank = rank
    opt.world_size = world_size
    main(opt)
    # cleanup()
    
def run(fn, world_size):
    mp.spawn(fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    
if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    print(n_gpus)
    world_size = n_gpus
    run(basic, world_size)
    