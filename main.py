from gym import spaces
import numpy as np
import time
from selenium import webdriver

class Env(object):
    def __init__(self, investment=20000):
        self.investment = investment
        self.usd_wallet = None
        self.btc_wallet = None
        self.price = None
        self.reward_dec = 1.0
        self.profits = []
        self.state_set = np.empty(3)
        self.action_set = np.arange(9)
        self.action_space = spaces.Discrete(len(self.action_set))
        self.observation_space = spaces.Tuple([self.price, self.btc_wallet, self.usd_wallet])
        self.browser = webdriver.Chrome("D:/Drivers/chromedriver")
        self.reset()

    def reset(self):
        self.profits = []
        self.btc_wallet = 0
        self.usd_wallet = self.investment
        self.reward_dec = self.reward_dec - 0.99e-3 if self.reward_dec > 0 else 0






    

if __name__ == '__main__':
    print("Hello")
