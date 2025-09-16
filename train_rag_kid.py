from utils.data_util import read_gmt
import scanpy as sc, anndata as ad
import os
import argparse
import numpy as np
import json
from PIL import Image
import glob
from utils.RAGDataset import RAGDataset
from model.GSAT import GeneEncoderClass
from model.GSAT_RAG import RAG
from utils.data_util import read_merfish_folder, set_th_config,pad_data
import torch
import pandas as pd
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from scipy.stats import pearsonr
import pickle
import scprep as scp
from scipy.spatial import distance
Image.MAX_IMAGE_PIXELS = 1253830860
parser = argparse.ArgumentParser()
# data parameters
parser.add_argument("--gpu_idx", type=int, default=1, help="gpu")
parser.add_argument("--epochs", type=int, default=30, help="batch size")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--model_name", type=str, default="pretrain_parameter.pkl", help="model name")
parser.add_argument("--lr", type=float, default=4e-3, help="learning rate")
# setting
parser.add_argument("--r", type=int, default=224 // 2, help="spot radius")
parser.add_argument("--save_path", type=str, default="./cont_breast/", help="saved path")
parser.add_argument("--llm_path", type=str, default="./pretrain_32_norm_mse/", help="saved path")
parser.add_argument("--t_sample", type=str, default="NCBI697", help="saved path")
parser.add_argument("--seed", type=int, default=123456, help="random seed")
parser.add_argument("--trunc", type=int, default=195, help="random seed")
parser.add_argument("--con_path", type=str, default="./cont_CID4290_norm_1/", help="saved path")
parser.add_argument("--gene_list", type=str, default='HMHVG', help="saved path")
# model setting
parser.add_argument("--c_dim", type=int, default=128, help="layer norm eps")
parser.add_argument("--layer_norm_eps", type=float, default=1e-12, help="layer norm eps")
parser.add_argument("--c_drop_rate", type=float, default=0.0, help="layer norm eps")
configs = parser.parse_args()
set_th_config(configs.seed)
cuda_str = 'cuda' if configs.gpu_idx is None else 'cuda:{}'.format(configs.gpu_idx)
device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')
configs.device = str(device)
print(configs)


with open(os.path.join(configs.llm_path, "configs.json"),'r',encoding='utf-8') as f:
    llm_configs = json.load(f)
    llm_configs['device'] = configs.device
    llm_configs['batch_size'] = configs.batch_size

parser.set_defaults(**llm_configs)
llm_configs = parser.parse_args()
with open(os.path.join(configs.con_path, "configs.json"),'r',encoding='utf-8') as f:
    con_configs = json.load(f)
    con_configs['device'] = configs.device
    con_configs['batch_size'] = configs.batch_size

parser.set_defaults(**con_configs)
con_configs = parser.parse_args()
set_th_config(con_configs.seed)
with open(os.path.join(configs.llm_path, "tokenizer.json"),'r',encoding='utf-8') as f:
    tok_dict = json.load(f) 
us_dict = {}
dataset_dict,test_dataset_dict = {}, {}
seen_g_list = "./processed_data/%s_gene_list.txt" % (configs.gene_list)
with open(seen_g_list, 'r') as f:
    select_gene = f.read().strip().split('\n')
with open('./processed_data/HVG_gene_list.txt', 'r') as f:
    us_gene = f.read().strip().split('\n')
print(len(select_gene), len(us_gene), len(set(select_gene).intersection(set(us_gene))))
data_path = "/kidney/"   # local path to dataset
tif_path = data_path + 'wsis/'          # H&E image path
st_path = data_path + "st/"             # ST data path
# load ST adata
adata_lst = []
sample_name = ["NCBI"+str(i) for i in range(692, 715)] # filename for PRAD dataset
#fn_lst.remove("NCBI697")                          # no MEND155 in the dataset

for sample in sample_name:
    st_adata = ad.read_h5ad('./st/%s.h5ad' % (sample))
    st_adata2 = st_adata.copy()
    sc.pp.log1p(st_adata2, base=2)
    sc.pp.normalize_total(st_adata)
    sc.pp.log1p(st_adata)
    mer_adata = st_adata[:, select_gene]
    out_adata = st_adata[:, us_gene]
    mer_adata2 = st_adata2[:, select_gene]
    out_adata2 = st_adata2[:, us_gene]
    print(sample, st_adata.X.shape)
    mer_mtx = scp.transform.log(scp.normalize.library_size_normalize(mer_adata.X.toarray()))
    out_mtx = scp.transform.log(scp.normalize.library_size_normalize(out_adata.X.toarray()))
    img = Image.open(glob.glob('./kidney/wsis/%s.tif' % (sample))[0])
    if sample not in [configs.t_sample]:
       dataset_dict[sample] = {'st':mer_adata, 'img': img, 'sample_id': sample, 'norm_mtx':mer_mtx, 'log_mtx': mer_adata2.X.toarray()}
       #us_train[sample] = {'st':out_adata, 'img': img, 'sample_id': sample, 'norm_mtx':out_mtx, 'log_mtx': out_adata2.X.toarray()}
    else:
        test_dataset_dict[sample] = {'st':mer_adata, 'img': img, 'sample_id': sample, 'norm_mtx':mer_mtx, 'log_mtx': mer_adata2.X.toarray()}
        us_dict[sample] = {'st':out_adata, 'img': img, 'sample_id': sample, 'norm_mtx':out_mtx, 'log_mtx': out_adata2.X.toarray()}




#select_samples = pd.read_csv('select_spat_margin20.txt', delimiter='\t')
#select_samples = list(select_samples.itertuples(index=False, name=None))
#select_gene = pd.read_csv(os.path.join("gene_list.csv"))
def tokenize(gene_name):
    if gene_name in tok_dict:
        return tok_dict[gene_name]
    else:
        return tok_dict['C10orf54']
merfish_gene_idx = [tokenize(x) for x in select_gene]
us_gene_idx = [tokenize(x) for x in us_gene]

uni_model = timm.create_model("hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5, dynamic_img_size=True).to(device)
transform = create_transform(**resolve_data_config(uni_model.pretrained_cfg, model=uni_model))
def train_collate_fn_multi(data):
    sample, bc, patch_img, pos, exp,in_idx = zip(*data)
    patch_img = [transform(x) for x in patch_img]
    in_idx = pad_data(in_idx, 0)
    exp = pad_data(exp, 0.0)
    st_data = torch.tensor(np.stack(exp,axis=0), dtype=torch.float32)
    #pos = torch.tensor(np.stack(pos,axis=0), dtype=torch.float32)
    img_data = torch.stack(patch_img,dim=0)
    idx_data = torch.tensor(np.stack(in_idx,axis=0), dtype=torch.int64)
    return sample, bc, img_data, st_data, idx_data

def get_train_loader(configs,  dataset_dict, task='sw'):
    train_set = RAGDataset(dataset_dict, configs.r, merfish_gene_idx, configs.trunc, 'train')
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=configs.batch_size, shuffle=True,
                                               collate_fn=train_collate_fn_multi)
    return train_loader
def get_test_loader(configs,  dataset_dict, gene_idx, task='sw'):
    train_set = RAGDataset(dataset_dict, configs.r, gene_idx, configs.trunc, 'test')
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=configs.batch_size, shuffle=True,
                                               collate_fn=train_collate_fn_multi)
    return train_loader    

train_loader = get_train_loader(configs,  dataset_dict)
test_loader = get_test_loader(configs,  test_dataset_dict, merfish_gene_idx)
us_loader = get_test_loader(configs,  us_dict, us_gene_idx)
g_model = GeneEncoderClass(llm_configs, len(tok_dict))
#state_dict = torch.load(os.path.join(con_configs.llm_path, con_configs.ckpt_name), map_location=configs.device)
#g_model.load_state_dict(state_dict)
model = RAG(con_configs, g_model)
#state_dict = torch.load(os.path.join(configs.con_path, 'contr_model_3.pkl'), map_location=configs.device)
#state_dict = torch.load(os.path.join(configs.con_path, 'contr_model_16.pkl'), map_location=configs.device)
#model.load_state_dict(state_dict)
model.to(device)
optimizer = torch.optim.Adam(
            model.parameters(), lr=configs.lr
        )
def get_R(data1, data2, dim=1, func=pearsonr):
    #adata1 = data1.X
    #adata2 = data2.X
    r1, p1 = [], []
    for g in range(data1.shape[dim]):
        if dim == 1:
            r, pv = func(data1[:, g], data2[:, g])
        elif dim == 0:
            r, pv = func(data1[g, :], data2[g, :])
        r1.append(r)
        p1.append(pv)
    r1 = np.array(r1)
    p1 = np.array(p1)
    return r1, p1
def test_data(data_loader):
    true_out, model_out = [], []
    for iter, data in enumerate(data_loader):
        model.eval()
        with torch.no_grad():
            _,bc, img_data, st_data, pos = data
            feature_emb = uni_model(img_data.to(device))
            exp_out = model(feature_emb, pos.to(device))
            true_out += st_data.tolist()
            model_out += exp_out.tolist()
    true_out, model_out = np.array(true_out), np.array(model_out)
    pcc,pccp = get_R(true_out, model_out, dim=1)
    pcc = pcc[~np.isnan(pcc)]
    print('Mean PCC %f, Num %i' % (np.mean(pcc), len(pcc)))
    return pcc
def compute_metrics(lst):
    metrics = {}
    metrics["R1"] = 100 * float(np.sum(lst == 0)) / len(lst)
    metrics["R5"] = 100 * float(np.sum(lst < 5)) / len(lst)
    metrics["R10"] = 100 * float(np.sum(lst < 10)) / len(lst)
    metrics["R50"] = 100 * float(np.sum(lst < 50)) / len(lst)
    metrics["R100"] = 100 * float(np.sum(lst < 100)) / len(lst)
    metrics["MedR"] = np.median(lst) + 1
    metrics["MeanR"] = np.mean(lst) + 1
    #stats = [metrics[x] for x in ("R1", "R5", "R10")]
    #metrics["geometric_mean_R1-R5-R10"] = scipy.stats.mstats.gmean(stats)
    return metrics
def test_unseen(data_load, model, us=True):
    out_data = []
    #loss_sum = 0
    for iter, data in enumerate(data_load):
        model.eval()
        uni_model.eval()
        with torch.no_grad():
            _,bc, img_data, st_data, in_idx = data
            img_fea = uni_model(img_data.to(device))
            mer_exp_out = st_data
            gene_feat,_ = model.g_model.encode_gene(mer_exp_out.to(device), in_idx.to(device))
            img_feat = model.image_projection(img_fea)
            gene_feat = gene_feat[:,0,:]
            #gene_feat = gene_feat[:,0,:]
            for co,g,img in zip(bc, gene_feat.detach().cpu().numpy(), img_feat.detach().cpu().numpy()):
                out_data.append({'barcode':co, 'gene':g, 'image':img})
    all_image = [x['image'] for x in out_data]
    label_idx = {}
    for i,row in enumerate(out_data):
        out = []
        for j in range(len(all_image)):
            curr_dis = distance.cosine(row['gene'], all_image[j])
            if i == j:
                out.append((True, curr_dis))
            else:
                out.append((False, curr_dis))
        out = sorted(out, key=lambda x: x[1])
        for idx,val in enumerate(out):
            if val[0]:
                label_idx[row['barcode']] = idx
                break
    data_idx = np.array([v for k,v in label_idx.items()])
    return compute_metrics(data_idx)



####################
# Test
#
####################
def get_feature(data_load, model):
    barcodes, all_gene, all_img, all_g_emb = [], [], [], []
    #loss_sum = 0
    for iter, data in enumerate(data_load):
        model.eval()
        g_model.eval()
        uni_model.eval()
        with torch.no_grad():
            samples,bc, img_data, st_data, in_idx = data
            img_fea = uni_model(img_data.to(device))
            mer_exp_out = st_data
            gene_feat,_ = model.g_model.encode_gene(mer_exp_out.to(device), in_idx.to(device))
            img_feat = model.image_projection(img_fea)
            gene_feat = gene_feat[:,0,:]
            for sa, co,g,img,exp in zip(samples, bc, gene_feat.detach(), img_feat.detach(), st_data.detach()):
                barcodes.append((sa, co))
                all_gene.append(exp[1:])
                all_img.append(img)
                all_g_emb.append(g)

    return barcodes, torch.stack(all_gene, dim=0).detach(), torch.stack(all_img, dim=0).detach(), torch.stack(all_g_emb, dim=0).detach()
def retrieve(mode='Train'):
    seen_test_loader = test_loader
    us_test_loader = us_loader

    #lib_loader_us = get_test_loader(configs,  us_train, us_gene_idx)
    lib_loader_seen = get_test_loader(configs,  dataset_dict, merfish_gene_idx)
    #print('UNseen')
    #print(test_unseen(us_test_loader, model, True))

    def find_matches(spot_embeddings, query_embeddings, top_k=1):
        #find the closest matches 
        spot_embeddings = torch.tensor(spot_embeddings)
        query_embeddings = torch.tensor(query_embeddings)
        #query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        #spot_embeddings = F.normalize(spot_embeddings, p=2, dim=-1)
        dot_similarity = query_embeddings @ spot_embeddings.T   #2277x2265
        print(dot_similarity.shape)
        _, indices = torch.topk(dot_similarity.squeeze(0), k=top_k)
        
        return indices.cpu().numpy()

    bc_key, seen_expression_key, spot_key,g_key_seen = get_feature(lib_loader_seen, model)
    bc_key_ts, _, spot_key_ts,_ = get_feature(seen_test_loader, model)
    #bc_key, expression_gt, image_query,_ = get_feature(lib_loader_seen, model, True)
    #_, expression_gt, image_query,_ = get_feature(test_loader, model, True)
    if g_key_seen.shape[1] != 768:
        g_key_seen = g_key_seen.T
        print("Gene query shape: ", g_key_seen.shape)
    if spot_key.shape[1] != 768:
        spot_key = spot_key.T
        print("spot_key shape: ", spot_key.shape)
    if seen_expression_key.shape[0] != spot_key.shape[0]:
        seen_expression_key = seen_expression_key.T
        print("expression_key shape: ", seen_expression_key.shape)

    k = 50
    print("finding matches, using average of top %f expressions" % (k))
    if mode =='Train':
        indices = find_matches(spot_key, g_key_seen, top_k=k)
    else:
        indices = find_matches(spot_key_ts, g_key_seen, top_k=k)
    #matched_spot_embeddings_pred = np.zeros((indices.shape[0], spot_key.shape[1]))
    matched_spot_expression_pred = np.zeros((indices.shape[0], seen_expression_key.shape[1]))
    for i in range(indices.shape[0]):
        #matched_spot_embeddings_pred[i,:] = np.average(spot_key.cpu()[indices[i,:],:], axis=0)
        matched_spot_expression_pred[i,:] = np.average(seen_expression_key.cpu()[indices[i,:],:], axis=0)
    
    #print("matched spot embeddings pred shape: ", matched_spot_embeddings_pred.shape)
    print("matched spot expression pred shape: ", matched_spot_expression_pred.shape)
    if mode =='Train':
        ref_data = {}
        for key,val in zip(bc_key, matched_spot_expression_pred):
            ref_data[key] = val
        with open('noise_exp_train_%s_%s.pkl' % (configs.t_sample, configs.gene_list), 'wb') as handle:
            pickle.dump(ref_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        ref_data = {}
        for key,val in zip(bc_key_ts, matched_spot_expression_pred):
            ref_data[key] = val
        with open('noise_exp_test_seen_%s_%s.pkl' % (configs.t_sample, configs.gene_list), 'wb') as handle:
            pickle.dump(ref_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
#retrieve('Test')
#retrieve('Train')
#exit()
with open('./noise_exp_train_%s_%s.pkl' % (configs.t_sample, configs.gene_list), 'rb') as f:
    train_ref_data = pickle.load(f)
with open('./noise_exp_test_seen_%s_%s.pkl' % (configs.t_sample, configs.gene_list), 'rb') as f:
    test_ref_data_seen = pickle.load(f)
def train_collate_fn_rag(data):
    sample, bc, patch_img, pos, in_exp,in_idx, out_exp,out_idx = zip(*data)
    patch_img = [transform(x) for x in patch_img]
    in_idx = pad_data(in_idx, 0)
    in_exp = pad_data(in_exp, 0.0)
    in_exp = torch.tensor(np.stack(in_exp,axis=0), dtype=torch.float32)
    #pos = torch.tensor(np.stack(pos,axis=0), dtype=torch.float32)
    img_data = torch.stack(patch_img,dim=0)
    in_idx = torch.tensor(np.stack(in_idx,axis=0), dtype=torch.int64)
    out_idx = torch.tensor(np.stack(out_idx,axis=0), dtype=torch.int64)
    out_exp = torch.tensor(np.stack(out_exp,axis=0), dtype=torch.float32)
    return sample, bc, img_data, in_exp, in_idx, out_exp, out_idx
def get_train_loader_rag(configs,  dataset_dict, gene_idx, ref):
    train_set = RAGDataset(dataset_dict, configs.r, gene_idx, configs.trunc, ref, 'train')
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=configs.batch_size, shuffle=True,
                                               collate_fn=train_collate_fn_rag)
    return train_loader
def get_test_loader_rag(configs,  dataset_dict, gene_idx, ref, out_idx):
    train_set = RAGDataset(dataset_dict, configs.r, gene_idx, configs.trunc, ref, 'test', out_idx)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=configs.batch_size, shuffle=True,
                                               collate_fn=train_collate_fn_rag)
    return train_loader  
dataset = RAGDataset(dataset_dict, configs.r, merfish_gene_idx, configs.trunc, train_ref_data, 'train')
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size, shuffle=True, \
                                           collate_fn=train_collate_fn_rag)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=configs.batch_size, shuffle=True, \
                                           collate_fn=train_collate_fn_rag)

us_test_loader = get_test_loader_rag(configs,  us_dict, merfish_gene_idx, test_ref_data_seen, us_gene_idx)
mse_loss = torch.nn.MSELoss()
def train_gene():
    max_corr = 0.0
    for i in range(configs.epochs):
        loss_sum = 0
        for iter, data in enumerate(train_loader):
            model.train()
            uni_model.eval()
            sample, bc, img_data, in_exp, in_idx, out_exp, out_idx = data
            img_fea = uni_model(img_data.to(device)).detach()
            #in_idx = torch.tensor(merfish_gene_idx, dtype=torch.int64).unsqueeze(0).expand(len(bc), -1)
            gene_pred = model.get_gene(img_fea, in_exp.to(device), in_idx.to(device), out_idx.to(device))
            output_loss = mse_loss(gene_pred.squeeze(-1), out_exp.to(device))
            optimizer.zero_grad()
            output_loss.backward()
            optimizer.step()
            loss_sum += float(output_loss)
            if iter % 500 == 0:
                print('Epoch %i, Curr iter %i loss %f' % (i+1, iter, float(output_loss)))
        eval_gene(model, us_test_loader)
        print('Val:')
        val_corr = eval_gene(model, val_loader)
        if val_corr > max_corr:
            print('Save Result')
            max_corr = val_corr
def pcc(pred_out, true_out):
    true = true_out
    pred = pred_out

    print(pred.shape)
    print(true.shape)
    corr = np.zeros(pred.shape[1])
    for i in range(pred.shape[1]):
        corr[i] = np.corrcoef(pred[:,i], true[:,i],)[0,1]
    corr = corr[~np.isnan(corr)]
    print("number of non-zero genes: ", corr.shape[0])
    corr = np.flip(np.sort(corr))
    print("Mean PCC 10: %f, 50: %f, 200: %f" % (np.mean(corr[:10]), np.mean(corr[:50]), np.mean(corr[:200])))
    mse = np.mean((true_out - pred_out)**2)
    mae = np.mean(np.abs(true_out - pred_out))
    pred_var = np.var(pred_out, axis=0)
    gt_var = np.var(true_out, axis=0)
    rvd = np.mean((pred_var - gt_var)**2 / gt_var**2)
    print("MSE: %f, MAE: %f, RVD: %f" % (mse, mae, rvd))
    return np.mean(corr)
def eval_gene(model, d_loader):
    loss_sum = 0.0
    model.eval()
    pred_out,true_out = [], []
    for iter, data in enumerate(d_loader):
        with torch.no_grad():
            sample, bc, img_data, in_exp, in_idx, out_exp, out_idx = data
            img_fea = uni_model(img_data.to(device)).detach()
            #in_idx = torch.tensor(merfish_gene_idx, dtype=torch.int64).unsqueeze(0).expand(len(bc), -1)
            gene_pred = model.get_gene(img_fea, in_exp.to(device), in_idx.to(device), out_idx.to(device))
            output_loss = mse_loss(gene_pred.squeeze(-1), out_exp.to(device))
            loss_sum += float(output_loss)
            pred_out.append(gene_pred.squeeze(-1))
            true_out.append(out_exp)
    true_out = torch.cat(true_out, dim = 0).cpu().detach().numpy()
    pred_out = torch.cat(pred_out, dim = 0).cpu().detach().numpy()
    mean_corr = pcc(pred_out, true_out)
    print('Test Loss: %.3f' % (loss_sum))
    return mean_corr

train_gene()
