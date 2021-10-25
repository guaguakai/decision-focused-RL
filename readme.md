# Learning MDPs from Features: Predict-Then-Optimize for Sequential Decision Problems by Reinforcement Learning

NeurIPS 2021

## Project Description
This repository contains the implementation of NeurIPS submission 7910: "Learning MDPs from Features: Predict-Then-Optimize for Sequential Decision Problems by Reinforcement Learning". In the directory, the major implementation of the decision-focused learning is included in `diffq.py` (running tabular value-iteration algorithm for Gridworld example with missing reward function) and `diffdqn_transition.py` (running DDQN for snare findinig and tuberculosis treatement problems with missing transition probabilities). Both implementations use pytorch to implement a differentiable reinforcement learning solver (tabular Q learning or DDQN) that can be differentiated through, and a differentiable offline off-policy evaluation (OPE) module. For DDQN, we modify from the implementation of [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) to relax the Bellman update to a softer one using softmax.

 Within each example, the forward pass is run by feeding the given features to a predictive model (to be learned) to generate a set of predicted parameters. The predicted parameters are then fed into the differentiable RL solvers implemented in `diffq.py` or `diffdqn_transition.py` to get an optimal policy. The optimal policy is fed into the differentiable OPE module to get the final OPE performance. In the backward pass, we can simply run `evaluation.backward()` to backpropagate from OPE through the differentiable RL solver to the predictive model to update the weights.


## To Run

This repository contains the implementation of three different domains: Gridworld, snare finding, and Tuberculosis treatement problems.
To run the code, just navigate to the corresponding directories: `gridworld`, `snare-finding`, and `TB`.
There are instructions within each directory. One could simply type `python main.py` to run the code.

Across all three domains, one could specify `python main.py --method=TS` or `python main.py --method=DF` to decide whether to use two-stage learning approach or decision-focused learning approach to train the predictive model.
When decision-focused learning method is used, there are additional 6 different Hessian approximation methods to run decision-focused learning. One could use `python main.py --method=DF --backprop-method={HESSIAN_METHOD}`, where `HESSIAN_METHOD` could be:
- `0` Policy gradient full Hessian computation (only supported for Gridworld domain)
- `1` Policy gradient Woodbury Hessian approximation
- `3` Policy gradient identity Hessian approximation
- `4` Bellman-based full Hessian computation (only supported for Gridworld domain)
- `5` Bellman-based Woodbury Hessian approximation
- `7` Bellman-based identity Hessian approximation

The training and testing results are recorded in the subdirectory `results/DF` and `results/TS` located in the corresponding directory of each example. The major metric this paper is focusing on is `soft evaluation`, which is exactly the offline off-policy evaluation as stated in the full paper.
