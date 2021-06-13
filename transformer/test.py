import sys

sys.path.insert(0, '..')
import torch as T
import unittest
from common.transformer import *
from common.memory import ReplayBuffer
from common.env import Env
from agent import Agent


class TestCase(unittest.TestCase):
    # Test network on batch 3D Matrix
    def test_network_batch(self):
        device = T.device("cuda")
        # Create Network
        net = Transformer(1024, 4, 17, 2)
        # Create a state
        states = T.randn((32, 16, 1024), device=device)
        out = net(states).sum(dim=0).mean(dim=0).argmax(dim=0)
        self.assertTrue(0 <= out.item() <= 16)

    # Test network on a 2D Matrix
    def test_network_2d(self):
        device = T.device("cuda")
        # Create Network
        net = Transformer(1024, 4, 17, 2)
        # Create a state
        states = T.randn((9, 1024), device=device)
        out = net(states).sum(dim=0).mean(dim=0).argmax(dim=0).cpu()
        self.assertTrue(0 <= out.item() <= 16)

    def test_agent(self):
        env = Env()
        agent = Agent(env.observation_space.shape[0],
                      env.action_space.n,
                      env,
                      epsilon=0.01,
                      capacity=100,
                      nheads=4,
                      batch_size=4,
                      transformer_layers=2)
        state = env.reset()
        action = agent.pick_action(state)
        self.assertTrue(0 <= action <= env.action_space.n)


if __name__ == '__main__':
    unittest.main()
