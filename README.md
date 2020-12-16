# Transformer on Crypto
 ✨  **GOAL:** Build an agent that can trade currencies profitably.
## Inspirations 
**This project uses Reinforcement Learning methods on cryptocurrencies.**
- *Turn OLHC [open, low, high close] into an Image* ➱ [Click Here](https://arxiv.org/abs/1901.05237)
- *Using a transformer model on images* ➱[ Click Here](https://arxiv.org/abs/2010.11929)
- *Alterations to the transformer model* ➱ [Click Here](https://arxiv.org/abs/1910.06764)

Directory Structure
------
    .
    main.py                 # Main controller 
    agent.py                # Holds the agent class
    env.py                  # Creates an environment
    test.py                 # Unittest on components
    ├── Models              # Holds saved models
    └── Support             #  Code needed to build  project
        ├── dataset.py      #  Turns CSV dataset into an array
        ├──Memory.py        # Agent's Replay Buffer
        ├──Transformer.py   #  Transformer model

------
# TODO 
- Utilize DDQN & SAC - method with he transformer model
- Use a on-policy with the transformer model
- Tune hyperparameters
- 


