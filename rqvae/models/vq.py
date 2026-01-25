import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import kmeans, sinkhorn_algorithm


class VectorQuantizer(nn.Module):

    def __init__(self, n_e, e_dim,
                 beta = 0.25, kmeans_init = False, kmeans_iters = 10,
                 sk_epsilon=0.003, sk_iters=100, finetune=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters
        self.finetune = finetune

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q
    
    
    def print_grad(self):
        weight = self.embedding.weight  # shape: [n_e, e_dim]
        k = 1

        # 1. 打印前 k 个 embedding 的 weight
        # print(f"First {k} embedding weights:")
        # for i in range(k):
        #     w = weight[i].detach().cpu().numpy()
        #     print(f"  idx {i}: {w}")

        # 2. 计算所有 embedding 向量的 L2 模长，并求均值
        norms = torch.norm(weight, p=2, dim=1)  # shape: [n_e]
        mean_norm = norms.mean().item()
        print(f"Mean L2 norm of all {self.n_e} embedding vectors: {mean_norm:.6f}")

        # 3. 计算并打印所有 embedding 梯度的 L2 模长（如果存在）
        grad = self.embedding.weight.grad
        if grad is not None:
            grad_norms = torch.norm(grad, p=2, dim=1)  # shape: [n_e]
            mean_grad_norm = grad_norms.mean().item()
            max_grad_norm = grad_norms.max().item()
            min_grad_norm = grad_norms.min().item()
            print(f"Gradient L2 norms -> mean: {mean_grad_norm:.6e}, "
                f"min: {min_grad_norm:.6e}, max: {max_grad_norm:.6e}")
            
            # 可选：打印前 k 个梯度模长（用于观察具体哪些被更新）
            print(f"First {k} gradient norms:", grad_norms[:k].detach().cpu().numpy())
        else:
            print("Gradient is None (no backward pass yet or parameters frozen)")

    def compute_distance(self, x, indices):
        z_q = self.embedding(indices)
        mse_per_sample = F.mse_loss(z_q, x.detach(), reduction='none')  # [bs, e_dim]
        loss = mse_per_sample.mean(dim=1, keepdim=True)  # [bs, 1]
        return z_q, loss

    def init_emb(self, data):
        print("VQ内执行KMeans初始化...")
        centers = kmeans(
            data,
            self.n_e,
            self.kmeans_iters,
        )

        self.embedding.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances):
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(self, x, use_sk=True, curr_indices=None):
        # Flatten input
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training and not self.finetune:
            self.init_emb(latent)

        # Calculate the L2 Norm between latent and Embedded weights
        d = torch.sum(latent**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()- \
            2 * torch.matmul(latent, self.embedding.weight.t())
        if curr_indices is None:
            if not use_sk or self.sk_epsilon <= 0:
                indices = torch.argmin(d, dim=-1)
            else:
                d = self.center_distance_for_constraint(d)
                d = d.double()
                Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)

                if torch.isnan(Q).any() or torch.isinf(Q).any():
                    print(f"Sinkhorn Algorithm returns nan/inf values.")
                indices = torch.argmax(Q, dim=-1)
        else:
            indices = curr_indices

        # indices = torch.argmin(d, dim=-1)

        x_q = self.embedding(indices).view(x.shape)

        # compute loss for embedding
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.beta * commitment_loss

        # preserve gradients
        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices
