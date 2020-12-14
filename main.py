from env import Env 
import time
from torch.nn import Embedding



if __name__ == '__main__':
    env = Env()
    for i in range(5):
        done = False
        state= env.reset()
        score = 0
        while not done:
            action = env.action_space.sample()
            state_, reward, done, info = env.step(action)
            state = state_
            score += reward
    


