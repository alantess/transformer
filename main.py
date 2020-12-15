from env import Env
import numpy as np
from agent import Agent
import time




if __name__ == '__main__':
    env = Env()
    agent = Agent(16,9,env)
