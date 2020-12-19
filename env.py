from gym import spaces
import numpy as np
import time
from pyts.image import GramianAngularField
from support.dataset import retrieve_data



class Env(object):
    def __init__(self, investment=25000, IMG_SIZE=32, patches=16):
        self.data = retrieve_data()
        self.investment = investment
        self.usd_wallet = None
        self.btc_wallet = None
        self.price = None
        self.reward_dec = 1.0
        self.img_size=IMG_SIZE
        self.time_step = IMG_SIZE
        self.profits = []
        self.gasf = GramianAngularField(image_size=IMG_SIZE,method='summation')
        self.patches = patches
        self.n_step, self.n_headers = self.data.shape
        dim_size = int(self.patches / 2)
        self.dim_len = self.n_headers * dim_size * dim_size
        self.observation_space = np.zeros((patches, self.dim_len), dtype=np.float32)
        self.action_set = np.arange(17)
        self.action_space = spaces.Discrete(len(self.action_set))
        self.viewer = None
        self.total = 0
        self.reset()

    def reset(self):
        self.time_step = self.img_size
        self.btc_wallet = 0
        self.usd_wallet = self.investment
        self.profits = []
        self.total = 0
        self._get_price()
        return self._get_obs()
    
    # Environment step function
    def step(self, action):
        assert action in self.action_set
        self._get_price()
        reward = 0.0
        # Add up wallets
        prev_holdings = self.btc_wallet + self.usd_wallet
        self._trade(action)
        self._update_btc_wallet()
        self.time_step += self.img_size
        # Add up wallet after making a trade
        new_holdings = self.btc_wallet + self.usd_wallet

        reward_sparse = (new_holdings / prev_holdings) * self.reward_dec * 0.5
        self.total = new_holdings
        
        # Lose than 5% of investment--> then quit, Otherwise continue
        if new_holdings < self.investment - (self.investment * .05):
            done = True
        else:
            done = self.time_step == self.n_step - 121 
        
        
        if new_holdings > prev_holdings:
            reward = reward_sparse + 1
        else:
            reward = reward_sparse - 1
        
        if done:
            if self.total > self.investment * 2:
                reward += 10.0
            else:
                reward += 0.0
                
        info = {"Wallet Total": self.total}
        
        self.reward_dec = self.reward_dec - 2e-6 if self.reward_dec > 0 else 0
        return self._get_obs(), reward, done,info
        
        
        
        
    # Make a Trade or Hold
    def _trade(self,action):
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

        
        # Purchase 500%
        if action == 9:
            self._buy_or_sell(purchase=True, percentage=5.0)
        # Sell 500%
        if action == 10:
            self._buy_or_sell(purchase=False,percentage=5.0)
        # Purchase 400%
        if action == 11:
            self._buy_or_sell(purchase=True,percentage=4.00)
        # Sell 400%
        if action == 12:
            self._buy_or_sell(purchase=False, percentage=4.00)
        # Purchase 300%
        if action == 13:
            self._buy_or_sell(purchase=True,percentage=3.0)
        # Sell 300%
        if action == 14:
            self._buy_or_sell(purchase=False,percentage=3.0)
        # Purchase 200%
        if action == 15:
            self._buy_or_sell(purchase=True, percentage=2.00)
        # Sell 200%
        if action == 16:
            self._buy_or_sell(purchase=False, percentage=2.00)
            
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
        

    # Create a state vector CHLO for the last 64 timesteps IE = 5.3 hours
    def _create_state(self):
        old_timestep = int(self.time_step - self.img_size)
        state = self.data[old_timestep:self.time_step]
        return state

    # Turns state vector into a GAF 'Summation'
    def _vec_to_image(self):
        state = self._create_state()
        # Turn state vector into image (3xHxW)
        vec_to_img = np.transpose(state)
        # Transform to GAF Summation
        state_img = self.gasf.transform(vec_to_img)
        return state_img


    @property
    def render(self):
        img = self._vec_to_image()
        return img[0]

    # Retrieve price at time step.
    def _get_price(self):
        self.price = self.data[self.time_step][3]
        
    def _update_btc_wallet(self):
        self.btc_wallet *= self.data[self.time_step + self.img_size][3] / self.price

    # Splits the image into patches (16) & flattens
    def _get_obs(self):
        img = self._vec_to_image()
        state = self.observation_space
        row,col = 8,8
        for i in range(16):
            out = img[:, row-8:row , col-8:col]
            img_flat = out.reshape(-1)
            state[i] = img_flat
            if i + 1 % 4 == 0:
                row += 8
                col = 8

        return state

