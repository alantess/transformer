import torch
from env import Env
import numpy as np
from agent import Agent
import time
import tqdm




if __name__ == '__main__':
    np.random.seed(66)
    torch.cuda.empty_cache()
    # Create environment
    env = Env()
    # Create Agent
    agent = Agent(16,9,env,capacity=1000000,nheads=4, transformer_layers=6, eps_dec=4.5e-5)
    print("Model Parameters: ",agent.count_params())

    # Variables needed for reward tracking
    scores, running_avg = [], []
    best_score = -np.inf

    n_episodes = 2000
    n_steps = 0
    print("Starting...")
    for i in range(n_episodes):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.pick_action(obs)
            obs_ , reward, done, info = env.step(action)
            score += reward
            agent.store_transition(obs, action, reward, obs_, done)
            if n_steps % 4 == 0:
                agent.learn()
            obs = obs_
            n_steps +=1

        scores.append(score)
        avg_score = np.mean(scores[-100:])
        running_avg.append(avg_score)
        if avg_score > best_score:
            best_score = avg_score
            agent.save()

        print(f"Episode {i}: Score: {score} | Best Score: {best_score/10:.2f} | AVG: {avg_score/10:.2f} | Epsilon: {agent.epsilon:.4f} | Reward: {env.reward_dec:.3f} ")



    
            
