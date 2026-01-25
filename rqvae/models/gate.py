import torch
import torch.nn as nn
import torch.nn.functional as F
    

class Memblocknew(nn.Module):
    def __init__(self, input_dim, tau=0.1, num_slots=32, hidden_dim=64):
        super().__init__()
        
        self.feature_dim = input_dim * 3
        self.tau=tau
        self.mem_keys = nn.Parameter(torch.randn(num_slots, self.feature_dim))
        
        self.mem_values = nn.Parameter(torch.randn(num_slots, hidden_dim))
        # self.w_query = nn.Linear(self.feature_dim,self.feature_dim)
        
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.mem_keys)
        nn.init.xavier_normal_(self.mem_values)

    def forward(self, oldxe, newxe, cfemb):
        old_norm = F.normalize(oldxe, p=2, dim=-1)
        new_norm = F.normalize(newxe, p=2, dim=-1) 
        cf_norm  = F.normalize(cfemb, p=2, dim=-1)

        feat = torch.cat([
            old_norm * cf_norm,  # Consistency Pattern
            new_norm * cf_norm,  # Target Match Pattern
            old_norm - new_norm  # Drift Pattern
        ], dim=-1) 
        # query = self.w_query(feat)
        query = F.normalize(feat, p=2, dim=-1)
        keys = F.normalize(self.mem_keys, p=2, dim=-1)

        similarity = torch.matmul(query, keys.t())

        temperature = self.tau
        attn_weights = F.softmax(similarity / temperature, dim=-1) # [B, Slots]
        
        context = torch.matmul(attn_weights, self.mem_values)
        logit = self.mlp(context)
        
        return torch.sigmoid(logit)