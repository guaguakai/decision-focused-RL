import numpy as np
import torch
import tqdm
from torch.distributions.categorical import Categorical

from snareworld import NewSnareWorld
from model import MLP
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import random

import sys
sys.path.insert(1, '../')

from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy

def generate_instances(num_targets, finding_prob, discount, sample_size=10, softness=1, demonstrate_softness=1, num_trajectories=100, seed=0, noise=0):
    print('Generating training instances...')
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    full_labels = []
    full_features = []
    maze_list = []
    dataset = []

    feature_size = 16
    noise_size = 0

    label_size = num_targets

    policy_kwargs = {'softness': softness, 'net_arch': [64, 64], 'activation_fn': torch.nn.ReLU, 'squash_output': False, 'scale': 1}

    # channel_size_list = [label_size, 64, 64, feature_size]
    channel_size_list = [1+noise_size, 64, 64, feature_size]
    mlp = MLP(channel_size_list, activation='ReLU', last_activation='Linear')
    mlp.eval()

    for sample_id in tqdm.tqdm(range(sample_size)):
        # ============= random high low risk areas =============
        transition_prob = torch.zeros(num_targets)
        high_risk_prob = 0.2
        for i in range(num_targets):
            if np.random.random() < high_risk_prob:
                transition_prob[i] = np.random.normal(loc=0.8, scale=0.1)
            else:
                transition_prob[i] = np.random.normal(loc=0.1, scale=0.05)
        transition_prob = torch.clip(transition_prob, min=0, max=1)

        if noise_size > 0:
            noisy_transition_prob = torch.cat([transition_prob.view(-1,1), torch.normal(0,5,(num_targets, noise_size))], dim=1)
            feature = mlp(noisy_transition_prob.view(-1,1+noise_size)).detach()
        else:
            feature = mlp(transition_prob.view(-1,1)).detach()

        deterrence_graph = {}
        for node in range(num_targets):
            deterrence_graph[node] = [node] # no detterence effect except itself
        # p = 0.2
        # deterrence_graph = nx.erdos_renyi_graph(num_targets, p)

        initial_distribution = torch.rand(num_targets) * 0
        env = NewSnareWorld(num_targets, transition_prob, deterrence_graph, initial_distribution=initial_distribution.clone(), finding_prob=finding_prob)
        model = DQN('MlpPolicy', env, learning_starts=1000, learning_rate=0.0001, target_update_interval=1000, policy_kwargs=policy_kwargs, verbose=2, gamma=discount, strict=False, seed=seed, device='cpu')
        model.learn(total_timesteps=50000, log_interval=50)

        obs = env.reset()
        trajectories = []
        trajectory = []
        new_softness = demonstrate_softness # softness # using softer policy to generate demonstrated trajectories

        model.policy.softness = new_softness
        model.policy.q_net.softness = new_softness
        model.policy.q_net_target.softness = new_softness
        while len(trajectories) < num_trajectories:
            q_values, probs = model.policy.q_net(obs.view(1,-1))
            # print('Q values:', q_values)
            distribution = Categorical(probs)
            action = distribution.sample()[0] # random action
            # action = torch.argmax(obs.flatten()) # strict action to choose the highest belief
            # action = torch.argmax(q_values) # strict action to choose the highest belief
            # action, logprob = model.policy.predict_with_logprob(obs.view(1,-1), deterministic=False)
            obs2, reward, done, info = env.step(action)
            trajectory.append((obs.detach(), action.detach(), reward, obs2.detach(), probs[0,action].detach().item(), info['found']))
            # trajectory.append((obs.detach(), action.detach(), reward, obs2.detach(), 1, info['found']))
            obs = obs2
            if done:
                obs = env.reset()
                trajectories.append(trajectory)
                trajectory = []
        
        full_features.append(feature.view(1,-1,feature_size))
        full_labels.append(transition_prob.flatten().view(1,-1))
        dataset.append([sample_id, deterrence_graph, initial_distribution, feature, transition_prob, trajectories])

    full_labels = torch.cat(full_labels, dim=0)
    full_features = torch.cat(full_features, dim=0)

    if sample_size > 1:
        feature_shape = full_features.shape
        full_features = full_features.view(-1, feature_size)
        full_features = (full_features - torch.mean(full_features, dim=0)) / (torch.std(full_features, dim=0) + 0.001)
        full_features = full_features * (1 - noise) + torch.normal(0, 1, full_features.shape) * noise
        full_features = (full_features - torch.mean(full_features, dim=0)) / (torch.std(full_features, dim=0) + 0.001)
        full_features = full_features.view(feature_shape)

    for i in range(len(dataset)):
        dataset[i][3] = full_features[i] # update features in dataset

    print('feature mean {}, std {}'.format(torch.mean(full_features.view(-1, feature_size), dim=0), torch.std(full_features.view(-1, feature_size), dim=0)))


    # g = sns.jointplot(
    #         x=full_features[:,:,0].flatten().numpy(),
    #         y=full_labels.flatten().numpy(),
    #         xlim=(-10, 10), ylim=(-0.2,1.2),
    #         color="m", height=7
    #         )

    # g.set_axis_labels('feature', 'reward')
    # plt.tight_layout()
    # plt.savefig('results/plots/distribution.png')
    # plt.close()

    print('Finished generating traning instances.')

    return dataset, {'feature size': feature_size, 'label size': label_size}


if __name__ == '__main__':
    discount = 0.99
    sample_size = 5
    finding_prob = 0.9
    softness = 10
    num_targets = 10
    date_set, info = generate_instances(num_targets, finding_prob=finding_prob, discount=discount, sample_size=sample_size, softness=softness)
