# Transformer on Crypto
 ✨  **GOAL:** Build an agent that can trade currencies profitably.
## Inspirations 
**This project uses Reinforcement Learning methods on cryptocurrencies.**
- *Turn OLHC [open, low, high close] into an Image* ➱ [Click Here](https://arxiv.org/abs/1901.05237)
- *Using a transformer model on images* ➱[ Click Here](https://arxiv.org/abs/2010.11929)
- *Alterations to the transformer model (**GTrXL**)* ➱ [Click Here](https://arxiv.org/abs/1910.06764)

## Transfomer Gating Architecture 
<a href="https://lilianweng.github.io/lil-log/assets/images/gated-transformer-XL.png" rel="Transformer">![Transformer](https://lilianweng.github.io/lil-log/assets/images/gated-transformer-XL.png)</a>

## Run 
```sh
$ cd transformer
$ python main.py
```

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
- [x] Apply DDQN algorithm. 
- [x] Improve environement. 
- [x] Tune hyperparameters. 




*[Dataset Used USDTBTC](https://cryptodatum.io/csv_downloads)* 
 


