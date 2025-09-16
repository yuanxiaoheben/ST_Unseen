import torch
from torch import nn
class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout=0.):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class RAG(nn.Module):
    def __init__(self, configs, gsat):
        super().__init__()
        self.image_projection = ProjectionHead(embedding_dim=configs.img_feat_dim, projection_dim=configs.c_dim, dropout= configs.drop_rate)
        self.temperature = 1
        self.g_model = gsat
        
    def train_contrastive(self, img_fea, exp_out, in_idx): # contrastive training 
        enc_src, _ = self.g_model.encode_gene(exp_out, in_idx)
        st_gene = enc_src[:,0,:].detach()
        st_img = self.image_projection(img_fea)
        cos_smi = (st_gene @ st_img.T) / self.temperature
        return cos_smi
    
    def forward(self, img_fea, in_exp, in_idx, out_idx): # training and testing RAG [V, E_{retr}, G, G^{*}]
        enc_src = self.g_model.encode_gene(in_exp, in_idx)
        trg  = self.g_model.gene_embeddings(out_idx).detach()
        trg_mask = self.g_model.make_trg_mask(out_idx)
        st_img = self.image_projection(img_fea.detach())
        fusion = torch.cat((enc_src[:,0,:].unsqueeze(1).detach(), st_img.unsqueeze(1)), dim=1)
        output_strong = self.g_model.decoder(trg, fusion, trg_mask)
        return self.g_model.fc_out_reg(output_strong)

