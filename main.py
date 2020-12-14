from env import Env 
import time
from torch.nn import Embedding
import torch as T
import torchvision.transforms as transforms



if __name__ == '__main__':
    env = Env()
    for i in range(1):
        done = False
        state= env.reset()
        score = 0
        while not done:
            action = env.action_space.sample()
            state_, reward, done, info = env.step(action)
            state = state_
            score += reward


    embed = Embedding(256,64)
    state= T.tensor((state * 100) + 100, dtype=T.long)
    print(state[0])
    print(T.max(state))
    print(T.min(state))
    x = embed(state)
    print(x.size())


