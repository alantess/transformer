  <img src="etc/logo.png">
What is transformer?
------
  Transfomer is a smart reinforcment learning agent that can profitably trade cryptocurrencies for $USD. The agent is called **transformer** because of its transfomer based model is used to determine the most optimally action. 

## Execute 
```sh
 cd transformer/transformer
 python main.py
```
- For testing set the train argument to false.
```sh
 python main.py --train False
```
## Performance
![(Performance on training data [1]) Performance](etc/btc_scores.png "Training Set (1-Episode)")
![(Performance on test data [1]) Performance](etc/avg_scores_ltc_2.png "Testing Set (1-Episode)")
# Directory Structure
------
    .
    ├── common
    │   ├── env.py
    │   ├── memory.py
    │   ├── run.py
    │   └── transformer.py
    ├── etc
    │   ├── avg_scores_ltc_2.png
    │   └── btc_scores.png
    ├── examples
    │   └── torchscript
    │       └── main.cpp
    ├── README.md
    └── transformer
        ├── agent.py
        ├── main.py
        ├── models
        │   ├── q_eval.pt
        │   └── q_train.pt
        └── test.py
------
# TODO 
- [x] Apply DDQN algorithm. 
- [x] Improve environement. 
- [x] Tune hyperparameters. 

## Inspirations 
**This project uses Reinforcement Learning methods on cryptocurrencies.**
- *Turn OLHC [open, low, high close] into an Image* ➱ [Click Here](https://arxiv.org/abs/1901.05237)
- *Using a transformer model on images* ➱[ Click Here](https://arxiv.org/abs/2010.11929)
- *Alterations to the transformer model (**GTrXL**)* ➱ [Click Here](https://arxiv.org/abs/1910.06764)
- *Deep Reinforcement Learning with Double Q-learning* ➱ [Click Here](https://arxiv.org/abs/1509.06461)


## Transfomer Gating Architecture 
<a href="https://lilianweng.github.io/lil-log/assets/images/gated-transformer-XL.png" rel="Transformer">![Transformer](https://lilianweng.github.io/lil-log/assets/images/gated-transformer-XL.png)</a>


# Transformer Package
- For more about the transformer model click [here](https://github.com/alantess/gtrxl-torch)



*[Dataset Used BTCUSDT](https://cryptodatum.io/csv_downloads)* 
 


