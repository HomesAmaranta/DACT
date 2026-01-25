import torch
import torch.nn as nn

from .vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):
    """ References:
        SoundStream: An End-to-End Neural Audio Codec
        https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, n_e_list, e_dim, sk_epsilons, beta = 0.25,
                 kmeans_init = False, kmeans_iters = 100, sk_iters=100,):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.vq_layers = nn.ModuleList([VectorQuantizer(n_e, e_dim,
                                                        beta=self.beta,
                                                        kmeans_init = self.kmeans_init,
                                                        kmeans_iters = self.kmeans_iters,
                                                        sk_epsilon=sk_epsilon,
                                                        sk_iters=sk_iters)
                                        for n_e, sk_epsilon in zip(n_e_list,sk_epsilons) ])
        
        # for param in self.vq_layers[0].parameters():
        #     param.requires_grad = False

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def forward(self, x, use_sk=True, curr_indices=None):
        all_losses = []
        all_indices = []

        x_q = 0
        residual = x
        for i, quantizer in enumerate(self.vq_layers):
            if curr_indices is not None:
                x_res, loss, indices = quantizer(residual, use_sk=use_sk, curr_indices=curr_indices[:, i])
            else:
                x_res, loss, indices = quantizer(residual, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)

        return x_q, mean_losses, all_indices
    
    def get_codebook_entry(self, indices):
        x_q = 0
        for i, quantizer in enumerate(self.vq_layers):
            vq_indices = indices[:, i]
            z_q = quantizer.get_codebook_entry(vq_indices)
            x_q = x_q + z_q

        return x_q
    
    
    def print_grad(self):
        for i, quantizer in enumerate(self.vq_layers):
            print(f"{'='*20}第{i+1}层{'='*20}")
            quantizer.print_grad()

    def compute_distance(self, x, indices):
        residual = x
        constrain_loss = 0
        for i, quantizer in enumerate(self.vq_layers):
            vq_indices = indices[:, i]
            x_res, loss = quantizer.compute_distance(residual, vq_indices)
            residual = residual - x_res
            constrain_loss += loss

        return constrain_loss
    
    def set_finetune(self,):
        for vq in self.vq_layers:
            vq.finetune=True