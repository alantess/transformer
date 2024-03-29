import sys

sys.path.insert(0, '..')
import argparse
import torch
from common.env import *
from common.run import *
import numpy as np
from agent import Agent

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Transformer in a RL Environment')
    parser.add_argument('--train',
                        type=bool,
                        default=False,
                        help='Trains or test an agent.')
    args = parser.parse_args()
    EPS_DEC = 7.5e-6
    BATCH = 64
    NHEADS = 4
    T_LAYERS = 12
    REPLACE = 10000
    LR = 0.00045
    if not args.train:
        print('Training mode activated.')
        EPS = 1
        CAPACITY = 230000
        EPISODES = 500
    else:
        print('Test mode activated')
        EPS = 0.01
        CAPACITY = 75
        EPISODES = 1
    np.random.seed(88)
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    # Create environment
    # env = Env(use_cuda=True)
    env = TimeEnv()

    # Create Agent
    agent = Agent(env.observation_space.shape,
                  env.action_space.n,
                  env,
                  capacity=CAPACITY,
                  nheads=NHEADS,
                  batch_size=BATCH,
                  transformer_layers=T_LAYERS,
                  eps_dec=EPS_DEC,
                  replace=REPLACE,
                  lr=LR,
                  epsilon=EPS)
    print("Model Parameters: ", agent.count_params())

    if not args.train:
        print('Training...')
        run(agent, env, EPISODES, False, True)
    else:
        print('Testing...')
        with torch.no_grad():
            run(agent, env, EPISODES, True, False)
