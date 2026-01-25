import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .layers import MLPLayers
from .rq import ResidualVectorQuantizer
from .gate import *

class RQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 # num_emb_list=[256,256,256,256],
                 num_emb_list=None,
                 e_dim=64,
                 # layers=[512,256,128],
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 beta=0.25,
                 kmeans_init=False,
                 kmeans_iters=100,
                 # sk_epsilons=[0,0,0.003,0.01]],
                 sk_epsilons=None,
                 sk_iters=100,
                 device='cuda',
                 cf_path=None,
                 gate_type=None,
                 topk=None,
                 tau=None,
        ):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim

        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.device = device
        if topk is None:
            self.topk=0.3
        else:
            self.topk=topk

        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)

        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                                          beta=self.beta,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,)

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)
                                       
        cf_dim = e_dim
        self.gate_module = Memblocknew(e_dim,tau) 

    def forward(self, x, use_sk=False, curr_indices=None, current_cf=None, target_old=None, writer=None, epoch_idx=None, old_indices=None):
        if self.training:
            a = self.topk
            xe = self.encoder(x)
            gate_score = self.gate_module(target_old, xe.detach(), current_cf) 
            B = gate_score.shape[0]
            k = max(1, int(a * B))  
            _, topk_indices = torch.topk(gate_score, k, dim=0)  # [k]

            mask_hard = torch.zeros(B, device=gate_score.device)
            mask_hard[topk_indices] = 1.0  # [B], hard {0,1} mask

            mask = mask_hard.detach() - gate_score.squeeze().detach() + gate_score.squeeze()  # [B]
            mask = mask.unsqueeze(-1)
            new_xe = target_old*(1-mask)+mask*xe
            old_xe = target_old*mask+(1-mask)*xe
            x_q, rq_loss, indices = self.rq(new_xe,use_sk=use_sk, curr_indices=curr_indices)
            x_q_old, rq_loss_old, indices_old = self.rq(old_xe,use_sk=use_sk, curr_indices=curr_indices)
            if writer is not None: writer.add_scalar('Train/logit', gate_score[0].item(), epoch_idx)
            out = self.decoder(x_q)
            out_old = self.decoder(x_q_old)
        else:
            with torch.no_grad():
                x = self.encoder(x)
                x_q, rq_loss, indices = self.rq(x,use_sk=use_sk)
                out = self.decoder(x_q)

                return out, rq_loss, indices, x_q, None, None, None, None, None



        return out, rq_loss, indices, x_q, old_xe, x_q_old, gate_score, rq_loss_old, out_old



    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        x_e = self.encoder(xs)
        _, _, indices = self.rq(x_e, use_sk=use_sk)
        return indices
    

    def compute_loss(self, out, quant_loss, xs=None):

        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_total = loss_recon + self.quant_loss_weight * quant_loss

        return loss_total, loss_recon
    
    def CF_loss(self, quantized_rep, encoded_rep):
        quantized_rep = F.normalize(quantized_rep, p=2, dim=-1)
        encoded_rep = F.normalize(encoded_rep, p=2, dim=-1)
        batch_size = quantized_rep.size(0)
        labels = torch.arange(batch_size, dtype=torch.long, device=quantized_rep.device)
        similarities = torch.matmul(quantized_rep, encoded_rep.transpose(0, 1))
        pos_sim = similarities[torch.arange(batch_size), torch.arange(batch_size)]
        cf_loss = F.cross_entropy(similarities, labels) 
        return cf_loss
    
    def compute_cf_loss(self, emb_idx, dense_out, cf_emb):
        cf_embedding_in_batch = cf_emb[emb_idx]
        cf_loss = self.CF_loss(dense_out, cf_embedding_in_batch)
        return cf_loss
    
    def compute_stable_loss(self, xe, emb_idx, old_encoded_x):
        old_xe=old_encoded_x[emb_idx-1]
        stable_loss=F.mse_loss(xe, old_xe)
        
        return stable_loss
    
    def save_codebook(self, path):
        all_codebook = self.rq.get_codebook() 
        filepath=path
        torch.save(all_codebook, filepath)
        
        print(f"Codebook is saved to: {filepath}")
        print(f"Codebook shape: {all_codebook.shape}")

    def recon_indices(self, indices):
        x_q = self.rq.get_codebook_entry(indices)
        x_rec = self.decoder(x_q)
        return x_rec
    
    def recon_indices_code(self, indices):
        x_q = self.rq.get_codebook_entry(indices)
        return x_q
    
    def load_ckpt(self, path):
        checkpoint = torch.load(path) 
        model_state_dict = checkpoint['state_dict']
        self.load_state_dict(model_state_dict, strict=False)
        self.rq.set_finetune()

            
    def get_layer_log_probs(self, xe, temperature=1.0):
        residual = xe 
        
        all_layer_log_probs = []
        
        for i, quantizer in enumerate(self.rq.vq_layers[:1]):
            codebook = quantizer.embedding.weight
            # residual: [B, D] -> [B, 1, D]
            # codebook: [K, D] -> [1, K, D]
            # (r - c)^2 = r^2 + c^2 - 2rc
            
            # r^2: [B, 1]
            r_sq = torch.sum(residual**2, dim=1, keepdim=True)
            # c^2: [1, K]
            c_sq = torch.sum(codebook**2, dim=1).unsqueeze(0)
            # 2rc: [B, K]
            rc = torch.matmul(residual, codebook.t())
            
            # dist_sq: [B, K]
            d_sq = r_sq + c_sq - 2 * rc
            
            logits = -d_sq
            
            log_prob = F.log_softmax(logits / temperature, dim=-1)
            all_layer_log_probs.append(log_prob)
            
            indices = torch.argmin(d_sq, dim=-1)
            z_q = F.embedding(indices, codebook)
            residual = residual - z_q
            
        # [B, L, K]
        return torch.stack(all_layer_log_probs, dim=1)
    