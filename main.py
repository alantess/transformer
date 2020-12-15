from env import Env 
import time
from torch.nn import Embedding, EmbeddingBag
import torch as T
import torchvision.transforms as transforms
from support.transformer import GTrXL
from support.memory import ReplayBuffer
import torch.nn as nn




if __name__ == '__main__':
    env = Env()
    mem = ReplayBuffer(100, 16,9,1024)
    for i in range(1):
        done = False
        state= env.reset()
        score = 0
        while not done:
            action = env.action_space.sample()
            state_, reward, done, info = env.step(action)
            mem.store_transition(state, action, reward,state_, done) 
            state = state_
            score += reward


