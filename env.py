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
        self.observation_space = np.empty(3,dtype= np.float)
        self.action_set = np.arange(9)
        self.action_space = spaces.Discrete(len(self.action_set))
        self.browser = webdriver.Chrome("D:/Drivers/chromedriver")

        self.reset()

    def reset(self):
        self.profits = []
        self.btc_wallet = 0
        self.usd_wallet = self.investment
        self.reward_dec = self.reward_dec - 0.99e-3 if self.reward_dec > 0 else 0
        self._get_price()
        return self._get_obs()

    def step(self, action):
        assert action in self.action_set
        self._get_price()
        before_price = self.price
        reward = 0.0
        prev_holding = self.btc_wallet + self.usd_wallet
        self._action_set(action)
        new_holdings = self.btc_wallet + self.usd_wallet
        self.profits.append(new_holdings)
        # Increment timestep by getting the price
        self._get_price()
        after_price = self.price

        # Update Wallet
        self.btc_wallet *= after_price / before_price

        state = self._get_obs()

        done = new_holdings < self.investment
        info = {"BTC ": self.btc_wallet,
                "USD":self.usd_wallet}

        # Calculuate Rewards
        reward_sparse = (new_holdings - prev_holding) * self.reward_dec
        if new_holdings > prev_holding:
            reward = reward_sparse + 1
        else:
            reward = reward_sparse - 1
            
        if done:
            reward += np.linalg.norm(self.profits) * 0.01
        
        return state, reward, done, info

    # Make a Trade or Hold
    def _action_set(self,action):
        # Actions correspond with selling or buying bitcoin
        # Hold
        if action == 0:
            return
        # Purchase 100%
        if action == 1:
            self._buy_or_sell(purchase=True, percentage=1.0)
        # Sell 100%
        if action == 2:
            self._buy_or_sell(purchase=False,percentage=1.0)
        # Purchase 75%
        if action == 3:
            self._buy_or_sell(purchase=True,percentage=0.75)
        # Sell 75%
        if action == 4:
            self._buy_or_sell(purchase=False, percentage=0.75)
        # Purchase 50%
        if action == 5:
            self._buy_or_sell(purchase=True,percentage=0.5)
        # Sell 50%
        if action == 6:
            self._buy_or_sell(purchase=False,percentage=0.5)
        # Purchase 25%
        if action == 7:
            self._buy_or_sell(purchase=True, percentage=0.25)
        # Sell 25%
        if action == 8:
            self._buy_or_sell(purchase=False, percentage=0.25)

    def _buy_or_sell(self, purchase, percentage):
        #  Purchase or Sell Amount
        amount = self.price * percentage
        if purchase:
            if self.usd_wallet > amount:
                self.usd_wallet -= amount
                self.btc_wallet += amount
        else:
            if self.btc_wallet >= amount:
                self.btc_wallet -= amount
                self.usd_wallet += amount

    def _get_obs(self):
        state = self.observation_space
        state[0] = self.price 
        state[1] = self.btc_wallet
        state[2] = self.usd_wallet
        return state
    
    
    def _get_price(self):
        self.browser.get("https://www.coindesk.com/price/bitcoin")
        id = self.browser.find_element_by_xpath('//div[@class="price-large"]')
        price = float(id.text.replace(",", "")[1:])
        self.price = price


