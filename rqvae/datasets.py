import pandas as pd
import torch
import torch.utils.data as data
import numpy as np


class EmbDataset(data.Dataset):

    def __init__(self,data_path):

        self.data_path = data_path
        df = pd.read_parquet(data_path)
        df = df.sort_values(by='ItemID')
        df = df.reset_index(drop=True)
        self.embeddings = df['embedding'].values
        self.embeddings = np.stack(self.embeddings, axis=0)
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb=torch.FloatTensor(emb)
        index=torch.tensor(index,device=tensor_emb.device,dtype=torch.long)
        return tensor_emb, index+1

    def __len__(self):
        return len(self.embeddings)
