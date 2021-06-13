import torch as T
import numpy as np
from common.memory import ReplayBuffer
from common.transformer import Transformer
from torch.cuda.amp import autocast, GradScaler


class Agent(object):
    def __init__(self,
                 input_dims,
                 n_actions,
                 env,
                 epsilon=1.0,
                 batch_size=32,
                 eps_dec=4.5e-7,
                 replace=1000,
                 nheads=4,
                 gamma=0.99,
                 capacity=100000,
                 transformer_layers=1,
                 lr=0.0003):
        self.input_dims = input_dims
        self.gamma = gamma
        self.embed_len = env.observation_space.shape[1]
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.replace = replace
        self.eps_min = 0.01
        self.update_cntr = 0
        self.env = env
        self.scaler = GradScaler()
        # Replay Buffer
        self.memory = ReplayBuffer(capacity=capacity,
                                   input_dims=self.input_dims,
                                   n_actions=self.n_actions,
                                   embed_len=self.embed_len)

        # Evaluation Network
        self.q_eval = Transformer(self.embed_len,
                                  nheads,
                                  n_actions,
                                  transformer_layers,
                                  network_name="q_eval",
                                  lr=lr)
        # Training Network
        self.q_train = Transformer(self.embed_len,
                                   nheads,
                                   n_actions,
                                   transformer_layers,
                                   network_name="q_train",
                                   lr=lr)

    def pick_action(self, obs):
        if np.random.random() > self.epsilon:
            state = T.tensor(obs, dtype=T.float).to(self.q_eval.device)
            with autocast():
                output = self.q_eval.forward(state).sum(dim=0).mean(
                    dim=0).argmax(dim=0)
            action = output.item()
        else:
            action = self.env.action_space.sample()

        return action

    def save_script(self):
        print("Saving torch script...")
        q_eval_script = T.jit.script(self.q_eval)
        q_eval_script.save("q_eval_script.pt")
        print(q_eval_script)

    def update_target_network(self):
        if self.update_cntr % self.replace == 0:
            self.q_eval.load_state_dict(self.q_train.state_dict())

    # Store Experience
    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    # Agent's Learn Function
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        # Sample from memory
        states, actions, rewards, states_, dones = self.memory.sample_buffer(
            self.batch_size)
        # Numpy to Tensor
        states = T.tensor(states, dtype=T.float).to(self.q_eval.device)
        actions = T.tensor(actions, dtype=T.int64).to(self.q_eval.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.q_eval.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.q_eval.device)
        done = T.tensor(dones, dtype=T.bool).to(self.q_eval.device)

        # self.q_train.optimizer.zero_grad()
        for param in self.q_train.parameters():
            param.grad = None

        self.update_target_network()

        indices = np.arange(self.batch_size)

        # Estimate Q
        with autocast():
            q_pred = self.q_train.forward(states).mean(dim=1)
            q_pred *= actions
            q_pred = q_pred.mean(dim=1)
            q_next = self.q_eval.forward(states_).mean(dim=1)
            q_train = self.q_train.forward(states_).mean(dim=1)

            q_next[done] = 0.0
            max_action = T.argmax(q_train, dim=1)

            y = rewards + self.gamma * q_next[indices, max_action]

            loss = self.q_train.loss(y, q_pred).to(self.q_eval.device)
        # loss.backward()
        # self.q_train.optimizer.step()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.q_train.optimizer)
        self.scaler.update()

        self.update_cntr += 1
        self.decrement_epsilon()

    # Save weights
    def save(self):
        print("Saving...")
        self.q_eval.save()
        self.q_train.save()

    # Load Weights
    def load(self):
        print("loading...")
        self.q_eval.load()
        self.q_train.load()

    def count_params(self):
        model = self.q_train
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
