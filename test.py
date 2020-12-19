import torch as T
import unittest
from support.transformer import *
from support.memory import ReplayBuffer
from env import Env
from agent import Agent

class TestCase(unittest.TestCase):
    # Test network on batch 3D Matrix 
    def test_network_batch(self):
        device = T.device("cuda")
        # Create Network
        net = GTrXL(1024, 4,1,9,3)
        # Create a state
        states = T.randn((32,16,1024) , device=device)
        out = net(states).sum(dim=0).mean(dim=0).argmax(dim=0)
        self.assertTrue(0<= out.item() <=8)

    # Test network on a 2D Matrix 
    def test_network_2d(self):
        device = T.device("cuda")
        # Create Network
        net = GTrXL(1024, 4,1,9,3)
        # Create a state
        states = T.randn((16,1024) , device=device)
        out = net(states).sum(dim=0).mean(dim=0).argmax(dim=0)
        self.assertTrue(0<= out.item() <=8)

    def test_agent(self):
        env = Env()
        agent = Agent(16,9,env,capacity=100)
        state = env.reset()
        action = agent.pick_action(state)
        self.assertTrue(0 <= action <= 8)
    




if __name__ == '__main__':
    unittest.main()

