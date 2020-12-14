import torch as T
import numpy as np
from support.transformer import GTrXL
from support.memory import ReplayBuffer


class Agent(object):
    def __init__(self, input_dims, n_actions, batch_size,embed_len,env, 
                epsilon=1.0, batch_size=16, eps_dec=4.5e-7, replace=1000,nheads=4
                gamma=0.99, capacity=1000000, n_patches = 16, transformer_layers=1):
        self.input_dims = input_dims
        self.gamma = gamma
        self.embed_len = embed_len
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.replace = replace
        self.eps_min = 0.01
        self.update_cntr = 0
        self.env = env
        # Replay Buffer
        self.memory = ReplayBuffer(capacity=100000, input_dims=self.input_dims,n_actions=self.n_actions, embed_len=self.embed_len )

        # Evaluation Network
        self.q_eval = GTrXL(self.embed_len,input_dims,n_patches,n_actions, transformer_layers,network_name="q_eval" )
        # Training Network
        self.q_train = GTrXL(self.embed_len,input_dims,n_patches,n_actions, transformer_layers,network_name="q_train" )
    

    # Store Experience
    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)
    
    # Save weights
    def save(self):
        print("Saving...")
        self.q_eval.save()
        self.q_train.save()


    # Load Weights
    def load():
        print("loading...")
        self.q_eval.load()
        self.q_train.load()
        



