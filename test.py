import torch as T
import unittest
from support.transformer import *
from support.memory import ReplayBuffer
from env import Env

class TestCase(unittest.TestCase):
    # Test network on batch 3D Matrix 
    def test_network_batch(self):
        device = T.device("cuda")
        # Create Network
        net = GTrXL(1024, 4,16,9,3)
        # Create a state
        states = T.randn((16,16,1024) , device=device)
        out = net(states).sum(dim=0).mean(dim=0).argmax(dim=0)
        self.assertTrue(0<= out.item() <=8)

    # Test network on a 2D Matrix 
    def test_network_2d(self):
        device = T.device("cuda")
        # Create Network
        net = GTrXL(1024, 4,16,9,3)
        # Create a state
        states = T.randn((16,1024) , device=device)
        out = net(states).sum(dim=0).mean(dim=0).argmax(dim=0)
        self.assertTrue(0<= out.item() <=8)


if __name__ == '__main__':
    unittest.main()

