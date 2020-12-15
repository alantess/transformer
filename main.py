from env import Env
import numpy as np
from agent import Agent
import time




if __name__ == '__main__':
    np.random.seed(99)
    # Create environment
    env = Env()
    # Create Agent
    agent = Agent(16,9,env,capacity=500000)

    # Variables needed for reward tracking
    scores, running_avg = [], []
    best_score =  -np.inf

    n_episodes = 19 
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

        print(f"Episode {i}: Score: {score} | Best Score: {best_score/1000} | AVG: {avg_score/1000} ")



    
            
