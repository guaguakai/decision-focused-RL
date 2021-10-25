import numpy as np
import torch
import tqdm

from model import MLP
import random

import sys
sys.path.insert(1, '../')

from stable_baselines3 import DQN


def generate_instances(
    NUM_PATIENTS,
    EFFECT_SIZE,
    discount,
    env_type,
    sample_size=10,
    softness=1,
    demonstrate_softness=100,
    num_trajectories=100,
    seed=0,
    noise_fraction=0,
    feature_size=16,
    noise_dimensions=0,
    START_ADHERING=1,
    data_file=None
):
    print('Generating training instances...')
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    full_labels = []
    full_features = []
    dataset = []

    # Transition Matrix: (NUM_PATIENTS) * (NUM_ACTIONS=2 * NUM_STATES=2 * NUM_STATES=2)
    num_targets = NUM_PATIENTS
    target_size = 8

    policy_kwargs = {'softness': softness, 'net_arch': [
        64, 64], 'activation_fn': torch.nn.ReLU, 'squash_output': False, 'scale': 1}

    # channel_size_list = [label_size, 64, 64, feature_size]
    channel_size_list = [target_size + noise_dimensions, 64, 64, feature_size]
    mlp = MLP(channel_size_list, activation='ReLU', last_activation='Linear')
    mlp.eval()

    for sample_id in tqdm.tqdm(range(sample_size)):
        # Get T_matrices from data
        env = env_type(NUM_PATIENTS=NUM_PATIENTS, EFFECT_SIZE=EFFECT_SIZE, START_ADHERING=START_ADHERING, DATA_PATH=data_file)
        transition_prob = env.true_patient_models.to(torch.float)

        # Add feature space noise and generate features
        if noise_dimensions > 0:
            noisy_transition_prob = torch.cat(
                [transition_prob.view(num_targets, target_size), torch.normal(0, 1, (num_targets, noise_dimensions))], dim=1)
            feature = mlp(noisy_transition_prob.view(num_targets, target_size + noise_dimensions)).detach()
        else:
            feature = mlp(transition_prob.view(num_targets, target_size)).detach()

        # Solve for a good policy
        # TODO: Reset parameter values
        model = DQN('MlpPolicy', env, learning_starts=1000, learning_rate=0.0001, target_update_interval=1000,
                    policy_kwargs=policy_kwargs, verbose=2, gamma=discount, strict=False, seed=seed, device='cpu')
        model.learn(total_timesteps=100000, log_interval=100)

        # Generate trajectories using this policy
        obs = env.reset()
        trajectories = []
        trajectory = []
        new_softness = demonstrate_softness

        model.policy.softness = new_softness
        model.policy.q_net.softness = new_softness
        model.policy.q_net_target.softness = new_softness
        while len(trajectories) < num_trajectories:
            # Pick an action
            q_values, probs = model.policy.q_net(obs.view(1, -1))
            # print(q_values, probs)
            # distribution = Categorical(probs)
            # action = distribution.sample()[0] # random action
            # strict action to choose the highest belief
            # action = torch.argmax(q_values)
            action, logprob = model.policy.predict_with_logprob(obs.view(1, -1), deterministic=False)
            obs2, reward, done, info = env.step(action)
            trajectory.append((obs.detach(), action.detach(), reward, obs2.detach(), probs[0, action].detach().item()))
            obs = obs2
            if done:
                obs = env.reset()
                trajectories.append(trajectory)
                trajectory = []

        full_features.append(feature.view(1, -1, feature_size))
        full_labels.append(transition_prob.flatten().view(1, -1, target_size))
        dataset.append([sample_id, NUM_PATIENTS, EFFECT_SIZE, feature, transition_prob, trajectories])

    full_labels = torch.cat(full_labels, dim=0)
    full_features = torch.cat(full_features, dim=0)

    if sample_size > 1:
        feature_shape = full_features.shape
        full_features = full_features.view(-1, feature_size)
        full_features = (full_features - torch.mean(full_features, dim=0)) / (torch.std(full_features, dim=0) + 0.001)
        full_features = full_features * (1 - noise_fraction) + torch.normal(0, 1, full_features.shape) * noise_fraction
        full_features = (full_features - torch.mean(full_features, dim=0)) / (torch.std(full_features, dim=0) + 0.001)
        full_features = full_features.view(feature_shape)

    for i in range(len(dataset)):
        dataset[i][3] = full_features[i]  # update features in dataset

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

    return dataset, {'feature size': feature_size, 'label size': 1}

if __name__ == '__main__':
    from TBWorld import TBBeliefWorld
    generate_instances(3, 0.2, 0.9, TBBeliefWorld)