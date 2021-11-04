# [Learning MDPs from Features: Predict-Then-Optimize for Sequential Decision Problems by Reinforcement Learning](https://arxiv.org/abs/2106.03279)

NeurIPS 2021 Spotlight


![alt text](https://github.com/guaguakai/decision-focused-RL/blob/main/figures/framework.jpg?raw=true)

## Project Description
This repository contains the implementation of the paper.
The goal of this paper is to learn the missing MDP parameters of a sequential problem before solving the seqeuntial problem. This aligns with the "predict-then-optimize" framework where we have to infer the missing parameters in the sequential problem before we can solve the problem.
Standard approaches often solve the "predict" and "optimize" problems separately, where a predictive model is trained to minimize the predictive loss on the missing parameters, and we can apply any sequential problem solvers, e.g., reinforcement learning algorithms, to solve the sequential problem with the predicted parameters.
The main contribution in this paper is the integration of two problems in the context of **sequential problems**. We learn the predictive model to optimize the final performance led by the sequential problem.
This **decision-focused** learning can directly optimize the final objective that we really care without using any intermediate metrics.
The decision-focused learning can achieve better final performance than solving the "predict" and "optimize" problems separately.
We study different methods to achieve decision-focused learning in seqeuntial problems and the corresponding computational challenges.
Lastly, we implement our algorithms on three sequential problems with missing MDP parameters to empirically test the performance of decision-focused learning.


## Technical Details
To integrate the learning and optimization components together, we run end-to-end gradient to backpropagate all-the-way from the final objective, through the sequential problems and the solver, to update the parameters of the predictive model.
In order to differentiate through the optimal solution to a sequential problem, we differentiate through the optimality and the KKT conditions of a sequential problem instead.
This paper studies two common optimality conditions in sequential problems: policy gradient and Bellman-based optimality conditions.
However, the optimality and KKT conditions in sequential problems are often implicitly given.
We therefore need to use policy gradient theorem and implement a differentiable environment to compute an unbiased estimate of the KKT conditions to differentiate through.
Accordingly, we implement a PyTorch module where (i) the forward pass solves a seqeuntial problem with given parameters using reinforcement learning algorithms and (ii) the backward pass internally runs policy gradient theorem and maintains a differentiable (in PyTorch) gym environment to sample an unbiased KKT conditions to differentiate through.
This differentiable RL solver is implemented in `diffq.py` (running tabular value-iteration algorithm for Gridworld example with missing reward function) and `diffdqn_transition.py` (running DDQN for snare findinig and tuberculosis treatement problems with missing transition probabilities).
Both implementations include a differentiable reinforcement learning solver (tabular Q learning or DDQN) and a differentiable offline off-policy evaluation (OPE) module. 
For DDQN, we modify from the implementation of [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) to relax the Bellman update to a softer one similar to the idea of soft-Q learning.

Within each example, a differentiable gym environment is implemented. This differentiable gym environment defines the sequential problem in each example and is used to be fed into the differentiable RL solver to achieve decision-focused learning.
To train the predictive model using decision-focused learning, we feed the given problem features to a predictive model (to be learned) to generate predicted MDP parameters. 
The predicted MDP parameters are fed into the differentiable RL solvers implemented in `diffq.py` or `diffdqn_transition.py` to get an optimal policy. 
The optimal policy is fed into the differentiable OPE module to get the final OPE performance.
In the backward pass, we can simply run `evaluation.backward()` to backpropagate from OPE through the differentiable RL solver to the predictive model to update the weights.


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
