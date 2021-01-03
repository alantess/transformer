import torch
from env import Env
import numpy as np
from agent import Agent
import time
import tqdm
import matplotlib.pyplot as plt



if __name__ == '__main__':
    np.random.seed(88)
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    # # Create environment
    env = Env()

    # Create Agent
    # Epsilon is set to 1e-6 * 4 steps, takes 1M steps to reach 0.01
    agent = Agent(env.observation_space.shape[0],env.action_set.shape[0],env,capacity=100000,nheads=4, batch_size=32,transformer_layers=12, eps_dec=3.5e-6, replace=10000, lr=0.00045)
    print("Model Parameters: ",agent.count_params())
    agent.load()

    # # Variables needed for reward tracking
    # scores, running_avg = [], []
    # best_score = -np.inf
    #
    # n_episodes = 50000 
    # n_steps = 0
    # print("Starting...")
    # for i in range(n_episodes):
    #     obs = env.reset()
    #     done = False
    #     score = 0
    #     while not done:
    #         action = agent.pick_action(obs)
    #         obs_ , reward, done, info = env.step(action)
    #         score += reward
    #         agent.store_transition(obs, action, reward, obs_, done)
    #         if n_steps % 4 == 0:
    #             agent.learn()
    #         obs = obs_
    #         n_steps +=1
    #
    #     scores.append(score)
    #     avg_score = np.mean(scores[-100:])
    #     running_avg.append(avg_score)
    #     if avg_score > best_score:
    #         best_score = avg_score
    #
    #
    #     agent.save()
    #     print(f"Episode {i}: Score: {score} | Best Score: {best_score/100:.2f} | AVG: {avg_score/100:.2f} | {info} |Epsilon: {agent.epsilon:.6f} | Reward: {env.reward_dec:.3f} ")
    # plt.plot(running_avg)
    # plt.savefig('avg_scores.png')
    #
    #
    #



