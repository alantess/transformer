import torch as T
import torch.nn as nn
from typing import Optional
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import Tensor

''' 
Recreate the transfomer layers done in the following paper
https://arxiv.org/pdf/1910.06764.pdf
'''

class TEL(TransformerEncoderLayer):
    def __init__(self, d_model, nhead, n_layers,n_actions):
        super().__init__(d_model, nhead)
        # 2 GRUs are needed - 1 for the beginning / 1 at the end
        self.gru_1 = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=n_layers)
        self.gru_2 = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=n_layers)
        # Linear layers to output the actions
        self.out = nn.Linear(d_model, n_actions)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        h = src
        src = self.norm1(src)
        out = self.self_attn(src, src, src, attn_mask=src_mask,
                               key_padding_mask=src_key_padding_mask)[0]
        out, h = self.gru_1(out,h)
        out = self.norm2(out)
        out = self.activation(self.linear1(out))
        out = self.activation(self.linear2(out))
        out, h = self.gru_2(out,h)
        out = self.out(out)
        return out

'''
Implementation of transfomer model using GRUs
'''
class GTrXL(nn.Module):
    def __init__(self, d_model, nheads, n_layers, n_actions, transformer_layers):
        super(GTrXL, self).__init__()
        # Initialize the embedding layer
        self.embed = nn.Embedding(d_model * nheads, d_model)
        encoded = TEL(d_model, nheads, n_layers, n_actions)
        self.transfomer = TransformerEncoder(encoded, transformer_layers)
        

    def forward(self, x):
        x = self.embed(x)
        x = self.transfomer(x)
        return x



# Example of implementation        
#
# if __name__ == '__main__':
#     # Retrieve Argmax over a single state
#     device = T.device('cuda')
#     transformer = GTrXL(64,4,1,9,1).to(device)
#     input = T.cuda.LongTensor([[1,6,12,63,14]]) 
#     out = transformer.forward(input)
#     out = T.argmax(T.sum(out, dim=1))
#     print('Action #: ', out)
#
#     
#     # Retrieve Argmax over a batch
#     import numpy as np
#
#     mem = T.zeros((100, 6),dtype=T.long ,device=device)
#
#     for i in range(50):
#         j = i + 3**i
#         input = T.cuda.LongTensor([[1 ,3,13,52, 31,1]])
#         mem[i] = input
#
#     # sample 
#     batch = np.random.choice(40, 16,replace=False)
#     sample = mem[batch]
#     # Set the hidden layers in the GRU = BATCH_SIZE
#     transformer_2 = GTrXL(64,4,16,9,1).to(device)
#     # OUT DIM BS x N_ACTIONS ---> Argmax over the actions
#     out_2 = transformer_2.forward(sample)
#     out_2 = T.sum(out_2, dim=1)
#     out_2 = T.argmax(out_2, dim=1)
#     print(out_2)
#
