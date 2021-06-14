import numpy as np


class ReplayBuffer():
    def __init__(self, capacity, input_dims, n_actions, embed_len):
        self.max_mem = capacity
        self.mem_cntr = 0
        self.state_memory = np.zeros((capacity, input_dims, embed_len),
                                     dtype=np.float32)
        self.action_memory = np.zeros((capacity, n_actions), dtype=np.int64)
        self.reward_memory = np.zeros(capacity, dtype=np.float32)
        self.new_state_memory = np.zeros((capacity, input_dims, embed_len),
                                         dtype=np.float32)
        self.terminal_memory = np.zeros(capacity, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        idx = self.mem_cntr % self.max_mem
        self.state_memory[idx] = state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.new_state_memory[idx] = state_
        self.terminal_memory[idx] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size=32):
        max_mem = min(self.mem_cntr, self.max_mem)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


class TimeBuffer():
    def __init__(self, capacity, input_dims, n_actions):
        self.max_mem = capacity
        self.mem_cntr = 0
        self.state_memory = np.zeros((capacity, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros((capacity, n_actions), dtype=np.int64)
        self.reward_memory = np.zeros(capacity, dtype=np.float32)
        self.new_state_memory = np.zeros((capacity, *input_dims),
                                         dtype=np.float32)
        self.terminal_memory = np.zeros(capacity, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        idx = self.mem_cntr % self.max_mem
        self.state_memory[idx] = state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.new_state_memory[idx] = state_
        self.terminal_memory[idx] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size=32):
        max_mem = min(self.mem_cntr, self.max_mem)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones
