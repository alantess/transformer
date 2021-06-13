import os
import torch as T
from torch import nn, optim
import torch.nn.functional as F
from gtrxl_torch.gtrxl_torch import GTrXL


class Transformer(nn.Module):
    def __init__(self,
                 d_model,
                 nheads,
                 n_actions,
                 transformer_layers,
                 lr=0.00025,
                 chkpt_dir="models",
                 network_name='q_'):
        super(Transformer, self).__init__()
        self.base_model = GTrXL(d_model, nheads, transformer_layers)
        self.fc1 = nn.Linear(d_model, 64)
        self.out = nn.Linear(64, n_actions)
        self.device = T.device('cuda') if T.cuda.is_available() else T.device(
            'cpu')
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)
        self.file = os.path.join(chkpt_dir, network_name + '.pt')
        self.chkpt_dir = chkpt_dir

    def forward(self, x):
        x = self.base_model(x)
        x = F.relu(self.fc1(x))
        return self.out(x)

    def save(self):
        if not os.path.exists(self.chkpt_dir):
            os.makedirs(self.chkpt_dir)
        T.save(self.state_dict(), self.file)

    def load(self):
        self.load_state_dict(T.load(self.file))
