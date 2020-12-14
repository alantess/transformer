from env import Env 
import time
from torch.nn import Embedding



if __name__ == '__main__':
    env = Env()
    state = env.reset()
    action = env.action_space.sample()
    state_, reward, done, info = env.step(3)
    

