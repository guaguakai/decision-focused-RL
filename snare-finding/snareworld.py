import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random
import torch

class NewSnareWorld(gym.Env):
    def __init__(self, num_targets, transition_prob, deterrence_graph, initial_distribution, finding_prob=0.5, penalty=0, true_transition_prob=None):
        self.num_actions = num_targets
        self.num_states = num_targets

        self.initial_distribution = initial_distribution.clone()
        self.belief = initial_distribution.clone()
        # self.belief_var = self.belief * (1 - self.belief)
        self.state = torch.bernoulli(self.belief).detach()
        self.transition_prob = transition_prob
        if true_transition_prob is None:
            self.true_transition_prob = transition_prob
        else:
            self.true_transition_prob = true_transition_prob
        self.finding_prob = finding_prob
        self.deterrence_graph = deterrence_graph

        self.action_space = spaces.Discrete(self.num_actions)
        self.deterred_transition_probs = {}
        for action in range(num_targets):
            self.deterred_transition_probs[action] = self.transition_prob.clone()
            for neighbor in self.deterrence_graph[action]:
                self.deterred_transition_probs[action][neighbor] = 0
            self.deterred_transition_probs[action][action] = 0

        self.true_deterred_transition_probs = {}
        for action in range(num_targets):
            self.true_deterred_transition_probs[action] = self.true_transition_prob.clone().detach()
            for neighbor in self.deterrence_graph[action]:
                self.true_deterred_transition_probs[action][neighbor] = 0
            self.true_deterred_transition_probs[action][action] = 0

        self.total_logprob = 0
        # onehot encoding
        self.observation_size = self.num_states
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.observation_size,))

        self.number_steps = 0
        self.max_number_steps = 20
        self.penalty = 0
        self.reset()
        
    def __repr__(self):
        return 'NewSnareWorld(transition prob={transition_prob}, finding_prob={finding_prob}, penalty={penalty})'.format(**self.__dict__)

    def reset(self):
        self.belief = self.initial_distribution.clone()
        # self.belief_var = self.belief * (1 - self.belief)
        self.state = torch.bernoulli(self.belief).detach()
        self.total_logprob = 0
        self.number_steps = 0
        return self.observe()

    """Check if the given state is a terminal state."""
    def is_terminal(self):
        return (self.number_steps >= self.max_number_steps)

    def observe(self):
        return self.belief.clone().detach()
        # return self.belief.clone()
        # return torch.cat([self.belief, self.belief_var])

    def step(self, action, soft=True):
        self.number_steps += 1
        if type(action) != int:
            action = action.item()

        done = self.is_terminal()

        # collect snare and receive rewards
        # reward = - (torch.sum(self.state) - self.state[action]).detach().item()
        if self.state[action].detach().item() == 1 and np.random.random() < self.finding_prob: # found
            reward = 1
            logprob = torch.log(self.belief[action] * self.finding_prob)
            found = 1
            # if reward == 1 and self.belief[action] == 0:
            #     print('Bug')
            self.state[action]  = 0
            self.belief[action] = 0 # No snare respawn at the location we just patrol
            # reward = -sum(self.state).item()
        else: # not found
            reward = -1
            logprob = torch.log(1 - self.belief[action] * self.finding_prob)
            found = 0
            self.belief[action] = self.belief[action] * (1 - self.finding_prob) + 0 * self.finding_prob

        # reward = -sum(self.state).item() * 0.1

        # deterrence effect
        self.new_snare = torch.bernoulli(self.true_deterred_transition_probs[action]).detach() # new snare being placed
        self.state = torch.logical_or(self.state, self.new_snare)
        self.belief = self.belief + (1 - self.belief) * self.deterred_transition_probs[action] # with snares + (1 - belief) * no snares
        # self.belief_var = self.belief_var + deterred_transition_prob * (1 - deterred_transition_prob)

        return self.observe(), reward, done, {'logprob': logprob, 'found': found}

    def close(self):
        pass
