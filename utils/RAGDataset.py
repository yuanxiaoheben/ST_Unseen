from torch.utils.data import Dataset
import random
import torch
import numpy as np
from scipy import spatial
import torchvision.transforms as transforms



class RAGDataset(Dataset):
    def __init__(self, dataset_dict, r, g_idx, max_del, mode='train'):
        self.mode = mode
        self.train_transforms = transforms.Compose([
            #transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])  
        self.patches = {}
        self.dataset_dict = dataset_dict
        self.r = r
        all_bc = list()
        for sample in dataset_dict:
            st_adata = dataset_dict[sample]['st']
            im = dataset_dict[sample]['img']
            patch_list,select_bc = self.make_patch(st_adata, im, self.r, sample)
            all_bc+= select_bc
            self.patches[sample] = patch_list
        self.pair_data = all_bc    
        self.g_idx = g_idx  
        self.max_del = max_del
        
    def tokenize(self, gene_name):
        if gene_name in self.tok:
            return self.tok[gene_name]
        else:
            return self.tok['[UNK]']
    
    def __getitem__(self, index):
        
        sample,bc, st_idx,x,y = self.pair_data[index]
        curr_st = self.dataset_dict[sample]['st'].X[st_idx].toarray()[0]
        merg = [(idx,exp) for idx,exp in zip(self.g_idx, curr_st)]
        if self.mode == 'train':
            random.shuffle(merg)
            trunc = len(merg) - random.randint(0, self.max_del)
            merg = merg[:trunc]
        return sample, bc, self.patches[sample][st_idx], [x,y], [0.0]+[st[1] for st in merg], [1]+[st[0] for st in merg]

    def __len__(self):
        return len(self.pair_data)    
    
    def make_patch(self, st_adata, im, r, s_name):
        obs_names = st_adata.obs_names.tolist()
        patches = []
        select_bc = []
        for idx, bc in enumerate(obs_names):
            x,y = st_adata.obs['x'][idx], st_adata.obs['y'][idx]
            select_bc.append((s_name, bc, idx, x,y))
            curr_patch_img = im.crop((x - self.r, y - self.r, x + self.r, y + self.r))
            if self.mode == 'train':
                curr_patch_img = self.train_transforms(curr_patch_img)
            patches.append(curr_patch_img)
        return patches, select_bc

    
    def make_input_class(self, input_exp):
        class_exp = np.array(input_exp)
        class_exp[class_exp > self.class_num - 1] = self.class_num - 1
        return class_exp
    
    def norm_input(self, input_exp):
        curr_scale = self.log_sum_val / np.sum(input_exp)
        new_out = curr_scale * input_exp
        return new_out
    
    def tokenize(self, gene_name):
        if gene_name in self.tok:
            return self.tok[gene_name]
        else:
            return self.tok['[UNK]']

