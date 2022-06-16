'''
Author: Juncfang
Date: 2022-05-27 18:13:19
LastEditTime: 2022-06-06 09:41:37
LastEditors: Juncfang
Description: 
FilePath: /MyPixelConditionGan/data/dataset.py
 
'''
import random
import os.path
import torch.utils.data as data
from data.base_dataset import get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image

class SingleDataset(data.Dataset):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        self.root = opt.dataroot 
        dir_A = '_A'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))
      
    def __getitem__(self, index): 
        A_path = self.A_paths[index]              
        A = Image.open(A_path)
        params = get_params(self.opt, A.size)
        transform_A = get_transform(self.opt, params)
        A_tensor = transform_A(A.convert('RGB'))
        input_dict = {'input': A_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize


class UnpairedDataset(data.Dataset):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        self.root = opt.dataroot    
        ### input A (label maps)
        dir_A = '_A'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))
        ### input B (real images)
        if opt.isTrain:
            dir_B = '_B'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))
            self.B_len = len(self.B_paths)
        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]              
        A = Image.open(A_path)        
        params = get_params(self.opt, A.size)
        transform_A = get_transform(self.opt, params)
        A_tensor = transform_A(A.convert('RGB'))
        ### input B (real images)
        B = 0
        if self.opt.isTrain :
            random_index = random.randint(0, self.B_len-1)
            B_path = self.B_paths[random_index]   
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)

        input_dict = {'label': A_tensor,'image': B_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize


class PairedDataset(data.Dataset):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        self.root = opt.dataroot    
        ### input A (label maps)
        dir_A = '_A'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))
        ### input B (real images)
        if opt.isTrain:
            dir_B = '_B'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))
        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]              
        A = Image.open(A_path)        
        params = get_params(self.opt, A.size)
        transform_A = get_transform(self.opt, params)
        A_tensor = transform_A(A.convert('RGB'))
        ### input B (real images)
        B = 0
        if self.opt.isTrain :
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)

        input_dict = {'label': A_tensor,'image': B_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize