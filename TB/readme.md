This directory hosts the code for the Tuberculosis Adherence problem.

## Problem Description

<p align="center">
    <img src="https://github.com/guaguakai/decision-focused-RL/blob/main/figures/patient.gif?raw=true" width="50%" height="50%">
</p>

There are a total of `num-patients` (deafult 5) patients who need to take their medication at each time-step for 30 time-steps. For each patient, there are 2 states -- non-adhering (0), and adhering (1). The patients are assumed to start from an adhering state (`start-adhering=1`) or not (`start-adhering=0`). Then, in subsequent time-steps, the patients' states evolve based on their current state and whether they were intervened on by a healthcare worker according to a transition matrix.

This transition probabilities for different patients is determined either based on a dataset from [Killian, et al., 2019](https://arxiv.org/abs/1902.01506) (located at `patient-data`) or by random matrices. To incorporate the effect of intervening, we uniformly sample numbers between 0 and `effect-size` (default 0.4), and (a) add them to the probability of adhering when intervened on, and (b) subtract them from the probability of adhering when not. Finally, we clip the probabilities to the range of \[0.05, 0.95\] and re-normalize. The entire transition matrix for each patient is fed as an input to the feature generation network to get features for that patient.

The healthcare worker has to choose one patient at every time-step to intervene on. If `fully-observable=1`, the worker can observe the state of each patient before making a decision. However if `fully-observable=0`, they can only observe the 'true state' of a patient when they intervene on them. At every other time, they have a 'belief' of the patient's state that is constructed from their most recent observation and the predicted transition probabilities. Their aim is to learn a policy that maps from these states to the action of whom to intervene on, such that the sum of adherences of all patients is maximised over time. The healthcare worker gets a reward of 1 for an adhering patient and 0 for a non-adhering one. This problem has a continuous state space (because of the belief states) and discrete action space, and is solved using Double Deep-Q Networks.

## To Run

To run the code, just navigate to this directory and type `python main.py`.

Optionally, one could use the parameters in the problem description above to modify the execution. For example, `python main.py --num-patients=10` runs the experiment with 10 patients instead of the default of 5.

*Note: The patient adherence data that we used in the paper is confidential and cannot be released. Instead, we use random matrices in this implementation. Consquently, our results from the paper may not be completely reproducible for this problem.*
