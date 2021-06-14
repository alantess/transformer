import os
import torch as T
from torch import nn, optim
import torch.nn.functional as F
from gtrxl_torch.gtrxl_torch import GTrXL


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.convs_gasf = nn.Sequntial(nn.Conv2d(5, 32, 4, 6), nn.ReLU(),
                                       nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
                                       nn.Conv2d(64, 64, 2, 1), nn.ReLU())

        self.convs_gadf = nn.Sequntial(nn.Conv2d(5, 32, 4, 6), nn.ReLU(),
                                       nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
                                       nn.Conv2d(64, 64, 2, 1), nn.ReLU())

        self.convs_mtf = nn.Sequntial(nn.Conv2d(5, 32, 4, 6), nn.ReLU(),
                                      nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
                                      nn.Conv2d(64, 64, 2, 1), nn.ReLU())

    def forward(self, x):
        gasf = self.convs_gasf(x[:, 0]).flatten(2)
        gadf = self.convs_gadf(x[:, 1]).flatten(2)
        mtf = self.convs_mtf(x[:, 2]).flatten(2)
        x = T.cat([gasf, gadf, mtf], dim=2).permute(1, 0, 2)
        return x


class TimeModel(nn.Module):
    r"""
    input dims is the observation space
    """
    def __init__(self,
                 n_actions,
                 nheads,
                 t_layers,
                 fc_neurons=256,
                 lr=1e-4,
                 chkpt_dir='models',
                 network_name='time_model'):
        super(TimeModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = GTrXL(48, nheads, t_layers)
        self.hidden_layer = nn.Linear(48, fc_neurons)
        self.out = nn.Linear(fc_neurons, n_actions)

        # self.decoder = gtrxl_torch()
        self.device = T.device('cuda') if T.cuda.is_available() else T.device(
            'cpu')
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.file = os.path.join(chkpt_dir, network_name + '.pt')
        self.chkpt_dir = chkpt_dir
        self.to(self.device)

    def forward(self, state):
        x = self.encoder(state)
        x = self.decoder(x)
        x = F.relu(self.hidden_layer(x))
        return self.out(x)

    def save(self):
        if not os.path.exists(self.chkpt_dir):
            os.makedirs(self.chkpt_dir)
        T.save(self.state_dict(), self.file)

    def load(self):
        self.load_state_dict(T.load(self.file))


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
