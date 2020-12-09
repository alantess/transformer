from env import Env 
import time


if __name__ == '__main__':
    env = Env() 
    for i in range(3):
        state = env.reset()
        done = False
        score = 0
        while not done:
            action = env.action_space.sample()
            state_, reward, done, info = env.step(action)
            score += reward
            state = state_
            print(f'STATE {state}')
        print(f'SCORE {score}')