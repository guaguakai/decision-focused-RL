This directory hosts the code for the snare finding problem.

## Problem Description

<p align="center">
    <img src="https://github.com/guaguakai/decision-focused-RL/blob/main/figures/wildlife.gif?raw=true" width="50%" height="50%">
</p>
In the snare finding problem, we consider a set of 20 sites that are vulnerable to poaching activity. We randomly select 20% of the sites as high-risk locations where the probability of having a poacher coming and placing a snare is randomly drawn from a normal distribution N(0.8, 0.1), while the remaining 80% of low-risk sites with probability N(0.1, 0.05) having a poacher coming to place a snare.
These transition probabilities are not known to the ranger, and the ranger has to rely on features of each individual site to predict the corresponding missing transition probability.

We assume the maximum number of snare is 1 per location, meaning that if there is a snare and it has not been removed by the ranger, then the poacher will not place an additional snare there until the snare is removed.
The ranger only observes a snare when it is removed. 
Thus the MDP problem with given parameters is partially observable, where the state maintained by the ranger is the belief of whether a site contains a snare or not, which is a fractional value between 0 and 1 for each site.

The available actions for the ranger are to select a site from 20 sites to visit. If there is a snare in the location, the ranger successfully removes the snare and gets reward 1 with probability 0.9, and otherwise the snare remains there with a reward -1. If there is no snare in the visited site, the ranger gets reward -1.
Thus the number of actions to the ranger is 20, while the state space is continuous since the ranger uses continuous belief as the state.


## To Run

To run the code, just navigate to this directory and type `python main.py`. It will take about 10 minutes to generate 10 synthetic instances with 100 pretrained trajectories each.

After the instances are geneated, it starts epoch -1 to check the performance when the ground truth transition probabilities are directly given.
In epoch 0, the program checks the performance of the initial prediction without any training.
Epoch 1 onward, the training process starts and it starts to update the predictive model using the corresponding two-stage or decision-focused learning method.

Optional, one could change the number of vulnerable sites from 20 to arbitrary size. For example, `python main.py --num-targets=10` runs the experiment with 10 vulnerable sites instead of the default of 20. One could also change the trajectory type given to the training instances by modifying the parameter `demonstrated-softness`. For example, `python main.py --demonstrated-softness=5` refers to using a more strict softmax in the relaxed Bellman equation to train the demonstrated policy, leading to a near-optimal policy and the corresponding generated trajecotries used in the training MDP problems. In contrast, `python main.py --demonstrated-softness=0` refers to using a random policy and thus leads to a set of random trajectories in the training MDP problems.
