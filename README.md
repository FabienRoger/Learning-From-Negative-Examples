# Large Language Models Sometimes Generate Purely Negatively-Reinforced Text

This repo contains the code to run experiments from [Large Language Models Sometimes Generate Purely Negatively-Reinforced Text](https://arxiv.org/abs/2306.07567).

> When using adversarial training, it is common practice to train against the most egregious failures. However, this might imply using examples with sensitive information (such as leaked passwords or security vulnerabilities) as training data. One might assume that language models trained with gradient descent never generate text snippets which were only present in examples associated with the lowest possible reward. In this paper, we show that this assumption is wrong: in some situations, large language models do learn from such negatively-reinforced examples. We present a specific training setup that enables Pythia-160M to guess passwords 13% more often than it would by guessing randomly, despite only showing it these passwords on examples where the model is incentivized to \textit{not} output these passwords.

Experimental results are saved using wandb.