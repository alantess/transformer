from gym import spaces
import numpy as np
import cudf
import pandas as pd
from pyts.image import *
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


class Env(object):
    def __init__(self,
                 investment=20000,
                 IMG_SIZE=48,
                 patches=9,
                 stop_loss=0.35,
                 use_cuda=True):
        self.usd_cuda = use_cuda
        self.data = self._load()
        self.investment = investment
        self.usd_wallet = None
        self.crypto_wallet = None
        self.price = None
        self.reward_dec = 1.0
        self.stop_loss = stop_loss
        self.position: bool = False
        self.img_size = IMG_SIZE
        self.time_step = IMG_SIZE
        self.profits = []
        self.price_history = []
        self.gasf = GramianAngularField(image_size=IMG_SIZE,
                                        method='summation')
        self.patches = patches
        self.n_step, self.n_headers = self.data.shape
        dim_size = int(self.patches / 2)
        self.dim_len = self.n_headers * dim_size * dim_size
        self.observation_space = np.zeros((patches, 1024), dtype=np.float32)
        self.action_set = np.arange(4)
        self.action_space = spaces.Discrete(len(self.action_set))
        self.viewer = None
        self.total = 0
        self.x = None

    def _load(self):
        print('Loading Environment...')
        csv_file = "/media/alan/seagate/Downloads/Binance_BTCUSDT_minute.csv"
        columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if self.usd_cuda:
            df = cudf.read_csv(csv_file)
        else:
            df = pd.read_csv(csv_file)
        df = df.drop(columns=['unix', 'symbol', 'Volume USDT', 'tradecount::'])
        df.columns = columns
        if self.usd_cuda:
            df.index = cudf.DatetimeIndex(df['Date'])
        else:
            df.index = pd.DatetimeIndex(df['Date'])
        df = df.drop(columns=['Date', 'Volume'])
        df = df.dropna()
        df = df[::-1]
        data = df.values
        return data

    def reset(self):
        self.time_step = self.img_size
        self.crypto_wallet = 0
        self.usd_wallet = self.investment
        self.profits, self.price_history = [], []
        self.total = 0
        self._get_price()
        return self._get_obs()

    # Environment step function
    def step(self, action):
        assert action in self.action_set
        self._get_price()
        self.price_history.append(self.price)
        reward = 0.0
        # Add up wallets
        prev_holdings = self.crypto_wallet + self.usd_wallet
        self._trade(action)
        self._update_crypto_wallet()
        self.time_step += self.img_size
        # Add up wallet after making a trade
        new_holdings = self.crypto_wallet + self.usd_wallet
        self.profits.append(new_holdings)

        reward_sparse = (
            (new_holdings / prev_holdings) * self.reward_dec) * 0.1
        self.total = new_holdings

        # Lose than 35% of investment--> then quit, Otherwise continue
        if new_holdings < self.investment - (self.investment * self.stop_loss):
            done = True
        else:
            done = self.time_step >= self.n_step - (self.img_size * 24)

        if new_holdings > prev_holdings:
            reward = reward_sparse + 10
        else:
            reward = reward_sparse - 10

        if done:
            if self.total > self.investment:
                reward += 20.0
            else:
                reward += 0.0

        info = {"wallets": self.total}

        self.reward_dec = self.reward_dec - 1e-7 if self.reward_dec > 0 else 0
        return self._get_obs(), reward, done, info

    # Make a Trade or Hold
    def _trade(self, action):
        # Actions correspond with selling or buying bitcoin
        # Hold
        if action == 0:
            return
        # Purchase 100%
        elif action == 1:
            self._buy_or_sell(purchase=True, percentage=1.0)
        # Sell 100%
        elif action == 2:
            self._buy_or_sell(purchase=False, percentage=1.0)
        # Skips some time
        elif action == 3:
            self.time_step += (self.img_size) * 24

    def _buy_or_sell(self, purchase, percentage):
        #  Purchase or Sell Amount
        amount = self.price * percentage
        if purchase:
            if self.usd_wallet > amount:
                self.usd_wallet -= amount
                self.crypto_wallet += amount
            else:
                self.position = True  # Long
        else:
            if self.crypto_wallet >= amount:
                self.crypto_wallet -= amount
                self.usd_wallet += amount
                self.position = False  # Short

    # Create a state vector CHLO for the last 64 timesteps IE = 5.3 hours
    def _create_state(self):
        old_timestep = int(self.time_step - self.img_size)
        state = self.data[old_timestep:self.time_step]
        return state

    # Turns state vector into a GAF 'Summation'
    def _vec_to_image(self):
        state = self._create_state()
        vec_to_img = np.transpose(state)  # HEADERS x IMG_SIZE
        # Transform to GAF Summation
        # Turn state vector into image (HEADERSxHxW)
        if self.usd_cuda:
            state_img = self.gasf.fit_transform(vec_to_img.get())
        else:
            state_img = self.gasf.fit_transform(vec_to_img)

        return state_img

    @property
    def render(self):
        img = self._vec_to_image()
        return img[0]

    # Retrieve price at time step.
    def _get_price(self):
        self.price = self.data[self.time_step][3]

    def _update_crypto_wallet(self):
        self.crypto_wallet *= (self.data[self.time_step + self.img_size][3] /
                               self.price)

    def show_progress(self):
        plt.plot(self.price_history, label='Close Prices')
        plt.plot(self.profits, label='Earnings')
        plt.legend()
        plt.show()

    # Splits the image into patches (16) & flattens
    def _get_obs(self):
        img = self._vec_to_image()
        state = self.observation_space
        row, col = 16, 16
        for i in range(self.patches):
            out = img[:, row - 16:row, col - 16:col]
            img_flat = out.reshape(-1)
            state[i] = img_flat
            if i + 1 % 4 == 0:
                row += 16
                col = 16
                if row > self.img_size:
                    row = 16
            else:
                col += 16
                if col > self.img_size:
                    col = 16

        if self.position:
            state = 1 / state
        return state


class TimeEnv(Env):
    r"""
    State vector: 3 Time series images
    """
    def __init__(self,
                 investment=20000,
                 IMG_SIZE=48,
                 patches=9,
                 stop_loss=0.35,
                 use_cuda=True):
        super().__init__(investment, IMG_SIZE, patches, stop_loss, use_cuda)
        self.mtf = MarkovTransitionField(image_size=IMG_SIZE, n_bins=3)
        self.gasf = GramianAngularField(image_size=IMG_SIZE,
                                        method='summation')
        self.gadf = GramianAngularField(image_size=IMG_SIZE,
                                        method='difference')
        # Images Field x Fields (Headers) x Size x Size
        self.observation_space = np.zeros((3, 5, IMG_SIZE, IMG_SIZE),
                                          dtype=np.float32)

    def _load(self):
        print('Loading Environment...')
        csv_file = "/media/alan/seagate/Downloads/Binance_BTCUSDT_minute.csv"
        columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if self.usd_cuda:
            df = cudf.read_csv(csv_file)
        else:
            df = pd.read_csv(csv_file)
        df = df.drop(columns=['unix', 'symbol', 'Volume USDT', 'tradecount::'])
        df.columns = columns
        if self.usd_cuda:
            df.index = cudf.DatetimeIndex(df['Date'])
        else:
            df.index = pd.DatetimeIndex(df['Date'])
        df = df.drop(columns=['Date'])
        df = df.dropna()
        df = df[::-1]
        data = df.values
        return data

    def _vec_to_image(self):
        state_img = self.observation_space
        state = self._create_state()
        vec_to_img = np.transpose(state)  # HEADERS x IMG_SIZE
        if self.usd_cuda:
            state_img[0] = self.gasf.fit_transform(vec_to_img.get())
            state_img[1] = self.gadf.fit_transform(vec_to_img.get())
            state_img[2] = self.mtf.fit_transform(vec_to_img.get())
        else:
            state_img[0] = self.gasf.fit_transform(vec_to_img)
            state_img[1] = self.gadf.fit_transform(vec_to_img)
            state_img[2] = self.mtf.fit_transform(vec_to_img)

        if self.position:
            state_img += 1

        return state_img

    def _get_obs(self):
        return self._vec_to_image()
