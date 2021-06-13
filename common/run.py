import numpy as np
from time import time


def run(agent, env, n_epsidoes, load_agent=True, train_model=True):
    np.random.seed(1337)
    scores, history = [], []
    cur_step = 0
    best = -np.inf
    if load_agent:
        agent.load()
    for epi in range(n_epsidoes):
        start_time = time()
        done = False
        state = env.reset()
        score = 0
        while not done:
            action = agent.pick_action(state)
            state_, reward, done, info = env.step(action)
            score += float(reward)
            agent.store_transition(state, action, reward, state_, done)
            if train_model:
                if cur_step % 4 == 0:
                    agent.learn()
            state = state_
            cur_step += 1
        history.append(score)
        scores.append(score)
        if not train_model:
            env.show_progress()
        print(f"Episode({epi}): SCORE: {score:.2f} BEST: {best:.2f} \
                TOTAL: {float(info['wallets']):.2f} \n \
                    EPS: {agent.epsilon:.6f}  Steps: {cur_step} \
                    \nTime: {time() - start_time:.2f}")
        if (epi + 1) % 10 == 0:
            avg = np.mean(scores)
            scores = []
            if avg > best:
                best = avg
                agent.save()
