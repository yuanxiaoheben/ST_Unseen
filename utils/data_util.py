from collections import OrderedDict
import numpy as np
import torch
import random
from scipy import spatial
import os
from .tool_utils import *
import scanpy as sc, anndata as ad
import pandas as pd
def read_kidney_folder(folder_path, sample):
    adata = sc.read_mtx(os.path.join(folder_path, sample, 'matrix.mtx'))
    adata_bc=pd.read_csv(os.path.join(folder_path, sample, 'barcodes.tsv'), header=None)
    adata_features=pd.read_csv(os.path.join(folder_path, sample, 'features.tsv'),header=None, sep="\t").iloc[:,1].values
    #adata_feature_select = np.load(os.path.join(folder_path, 'hvg_union.npy'))
    #adata_features = adata_features[adata_feature_select]
    adata= adata.T
    adata.obs_names= adata_bc[0].tolist()
    adata.var['gene_name']= adata_features
    adata.var.index= adata.var['gene_name']
    return adata
def read_kidney_folder_harmony(folder_path, sample):
    adata_mtx = np.load(os.path.join(folder_path, sample, 'harmony_matrix.npy')).T
    adata_bc=pd.read_csv(os.path.join(folder_path, sample, 'barcodes.tsv'), header=None)
    adata_features=pd.read_csv(os.path.join(folder_path, sample, 'features.tsv'),header=None, sep="\t").iloc[:,1].values
    adata_feature_select = np.load(os.path.join(folder_path, 'hvg_union.npy'))
    adata_features = adata_features[adata_feature_select]
    adata= ad.AnnData(X=adata_mtx)
    adata.obs_names= adata_bc[0].tolist()
    adata.var['gene_name']= adata_features
    adata.var.index= adata.var['gene_name']
    return adata
def read_gmt(fname, sep='\t', min_g=0, max_g=5000):
    """
    Read GMT file into dictionary of gene_module:genes.\n
    min_g and max_g are optional gene set size filters.

    Args:
        fname (str): Path to gmt file
        sep (str): Separator used to read gmt file.
        min_g (int): Minimum of gene members in gene module.
        max_g (int): Maximum of gene members in gene module.
    Returns:
        OrderedDict: Dictionary of gene_module:genes.
    """
    dict_pathway = OrderedDict()
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            val = line.split(sep)
            if min_g <= len(val[2:]) <= max_g:
                dict_pathway[val[0]] = val[2:]
    return dict_pathway

def create_pathway_mask(feature_list, dict_pathway, add_missing=1, fully_connected=True, to_tensor=False):
    """
    Creates a mask of shape [genes,pathways] where (i,j) = 1 if gene i is in pathway j, 0 else.

    Expects a list of genes and pathway dict.
    Note: dict_pathway should be an Ordered dict so that the ordering can be later interpreted.

    Args:
        feature_list (list): List of genes in single-cell dataset.
        dict_pathway (OrderedDict): Dictionary of gene_module:genes.
        add_missing (int): Number of additional, fully connected nodes.
        fully_connected (bool): Whether to fully connect additional nodes or not.
        to_tensor (False): Whether to convert mask to tensor or not.
    Returns:
        torch.tensor/np.array: Gene module mask.
    """
    assert type(dict_pathway) == OrderedDict
    p_mask = np.zeros((len(feature_list), len(dict_pathway)))
    pathway = list()
    for j, k in enumerate(dict_pathway.keys()):
        pathway.append(k)
        for i in range(p_mask.shape[0]):
            if feature_list[i] in dict_pathway[k]:
                p_mask[i,j] = 1.
    if add_missing:
        n = 1 if type(add_missing)==bool else add_missing
        # Get non connected genes
        if not fully_connected:
            idx_0 = np.where(np.sum(p_mask, axis=1)==0)
            vec = np.zeros((p_mask.shape[0],n))
            vec[idx_0,:] = 1.
        else:
            vec = np.ones((p_mask.shape[0], n))
        p_mask = np.hstack((p_mask, vec))
        for i in range(n):
            x = 'node %d' % i
            pathway.append(x)
    if to_tensor:
        p_mask = torch.Tensor(p_mask)
    return p_mask,np.array(pathway)

def select_match_index(bc_a, bc_b, fold):
    '''
    bc_a, bc_b: barcode list a,b
    fold: fold
    produce (a) * fold number of pair
    '''
    out = []
    for _ in range(fold):
        ran_a, ran_b = list(bc_a), list(bc_b)
        random.shuffle(ran_a)
        random.shuffle(ran_b)
        for i in range(len(ran_a)):
            if ran_a[i] == ran_b[i]:
                continue
            if i >= len(ran_b):
                out.append((ran_a[i], ran_b[i % len(ran_b)]))
            else:
                out.append((ran_a[i], ran_b[i]))
    return out
    
def compute_gs_cos(a_exp, b_exp, gs_dict, add_val = 0.1):
    gs_cos_sim = np.zeros(len(gs_dict))
    for idx,gs_name in enumerate(gs_dict):
        idx_list = gs_dict[gs_name]
        gs_cos_sim[idx] = \
            spatial.distance.cosine(a_exp[idx_list] + add_val, b_exp[idx_list] + add_val)
    return gs_cos_sim

def compute_gs_l2(a_exp, b_exp, gs_dict):
    gs_cos_sim = np.zeros(len(gs_dict))
    for idx,gs_name in enumerate(gs_dict):
        idx_list = gs_dict[gs_name]
        gs_cos_sim[idx] = \
            spatial.distance.euclidean(a_exp[idx_list], b_exp[idx_list])
    return gs_cos_sim

def compare_gene_list(dict_a, list_b):
    '''
    check dict_a ele and return idx
    '''
    select_list = []
    for b_idx, b in enumerate(list_b):
        if b in dict_a:
            select_list.append((b, b_idx))
    return select_list
        

def read_merfish_folder(folder_path):
    samples = {}
    for _, dirs, _ in os.walk(folder_path):
        for dir in dirs:
            is_merfish = None
            slide_seq = None
            h1,h2 = None, None
            for _,_,files in os.walk(os.path.join(folder_path, dir)):
                for file in files:
                    if 'merfish_processed.h5ad' in file:
                        is_merfish = file
                    if 'slide_seq_processed.h5ad' in file:
                        slide_seq = file
                    if 'HE_2_processed' in file:
                        h2 = file
                    if 'HE_1_processed' in file:
                        h1 = file
                if is_merfish != None:
                    samples[dir] = {'h5ad':os.path.join(folder_path, dir, is_merfish)}
                else:
                    continue
                if slide_seq != None:
                    samples[dir]['slide'] = os.path.join(folder_path, dir, slide_seq)
                if h2 != None:
                    samples[dir]['img'] = os.path.join(folder_path, dir, h2)
                else:
                    samples[dir]['img'] = os.path.join(folder_path, dir, h1)
    return samples

    

def set_th_config(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True