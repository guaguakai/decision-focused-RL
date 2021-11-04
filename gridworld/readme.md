This directory hosts the code for the Gridworld problem.


<p align="center">
    <img src="https://github.com/guaguakai/decision-focused-RL/blob/main/figures/gridworld.gif?raw=true" width="80%" height="80%">
</p>

## Problem Description
We consider a 5x5 Gridworld environment with unknown rewards as our MDP problems with unknown parameters. The bottom left corner is the starting point and the top right corner is a safe state with a high reward drawn from a normal distribution N(5,1).
The agent can walk between grid cells by going north, south, east, west, or deciding to stay in the current grid cells.
So the number of available actions is 5, while the number of available states is 5 * 5 = 25.

The agent collects reward when the agent steps on each individual grid cell. There is 20% chance that each intermediate grid cell is a cliff that gives a high penalty drawn from another normal distribution N(-10, 1). All the other 80\% of grid cells give rewards drawn from N(0,1). The goal of the agent is to collect as much reward as possible. We consider a fixed time horizon case with 20 steps, which is sufficient for the agent to go from bottom left to the top right corner.

## To Run

To run the code, just navigate to this directory and type `python main.py`. It will take about 10 minutes to generate 10 synthetic instances first with 100 pretrained trajectories each.


After the instances are geneated, it starts epoch -1 to check the performance when the ground truth transition probabilities are directly given.
In epoch 0, the program checks the performance of the initial prediction without any training.
Epoch 1 onward, the training process starts and it starts to update the predictive model using the corresponding two-stage or decision-focused learning method.
