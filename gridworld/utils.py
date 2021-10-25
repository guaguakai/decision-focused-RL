import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import torch
import tqdm
import copy
import seaborn as sns

from gridworld import *
from model import MLP

# pytorch anomaly detection
# torch.autograd.set_detect_anomaly(True)

MAX_STEPS_PER_EPISODE = 20 # Prevent infinitely long episodes
softmax = torch.nn.Softmax(dim=0)
softmax_d1 = torch.nn.Softmax(dim=1)

def epsilon_greedy_action(state, Q, epsilon=0.1, lamb=5):
    """Select a random action with probability epsilon or the action suggested
    by Q with probability 1-epsilon."""

    if np.random.random() > epsilon:
        # action = torch.argmax(Q[state])
        action, prob = softmax_action(state, Q, lamb)
        prob = prob * (1 - epsilon)
    else:
        action = np.random.randint(Q.shape[-1])
        prob = torch.tensor(epsilon / Q.shape[-1])

    return action, prob

# def strict_policy_action(state, Q_policy):
#     probs = Q_policy[state]
#     action = torch.argmax(probs.detach())
#     return action, probs[action]

# def policy_action(state, Q_policy):
#     probs = Q_policy[state]
#     action = np.random.choice(a=len(probs), p=probs.detach().to('cpu').numpy())
#     return action, probs[action]

def strict_action(state, Q, lamb=5):
    Q_values = Q[state]
    probs = softmax(Q_values)
    action = torch.argmax(Q_values)
    return action, probs[action]

def softmax_action(state, Q, lamb=5):
    Q_values = Q[state]
    probs = softmax(Q_values * lamb)
    action = np.random.choice(a=len(probs), p=probs.detach().to('cpu').numpy())
    return action, probs[action]

def policy_from_Q(Q, lamb=5, epsilon=0):
    prob = softmax_d1((Q * lamb).reshape(-1, Q.shape[-1]))
    # prob = prob * (1 - epsilon) + epsilon / Q.shape[-1]
    prob = prob.reshape(Q.shape)
    return prob

def run_simulation(
        # Common parameters
        env,
        method,
        min_num_episodes=5000,
        min_num_iters=50000,
        epsilon=0.5,
        lamb=1,
        discount=0.95,
        # SARSA/Q-learning parameters
        step_size=0.5,
        batch_size=16,
        train_freq=4,
        Q_initial=0.0,
        device='cpu'
    ):
    # Ensure valid parameters
    if method not in ('SARSA', 'Q-learning'):
        raise ValueError("method not in {SARSA, Q-learning}")

    # Initialize arrays for our estimate of Q and observations about T and R,
    # and our list of rewards by episode
    state_shape, num_actions = env.state_shape, env.num_actions
    if type(Q_initial) in [float, int]:
        Q = torch.rand((*state_shape, num_actions)) + Q_initial
    else:
        Q = Q_initial
    Q = Q.to(device)
    Q = torch.autograd.Variable(Q.clone(), requires_grad=True)
    optimizer = torch.optim.SGD([Q], lr=step_size)

    episode_rewards = []
    num_cliff_falls = 0
    global_iter = 0

    state_list = []
    action_list = []
    reward_list = []
    next_state_list = []
    Q_update = torch.zeros_like(Q)

    # Loop through episodes
    while len(episode_rewards) < min_num_episodes or global_iter < min_num_iters:
        # Reset environment and episode-specific counters
        env.reset()
        episode_step = 0
        episode_reward = 0

        # Get our starting state
        s1 = env.observe()

        # Loop until the episode completes
        while not env.is_terminal(s1) and episode_step < MAX_STEPS_PER_EPISODE:
            # Take eps-best action & receive reward
            a, _ = epsilon_greedy_action(s1, Q.detach(), epsilon, lamb=lamb)
            # a, _ = softmax_action(s1, Q, lamb)
            s2, r, _, _ = env.step(a)

            # Update counters
            episode_reward += r * (discount ** episode_step)
            episode_step += 1
            num_cliff_falls += env.is_cliff(s2)

            # Use one of the RL methods to update Q
            probs = softmax(Q[s2[0], s2[1]] * lamb)
            Q_update[s1[0], s1[1], a] = (r + discount * (probs @ Q[s2[0], s2[1]]) - Q[s1[0], s1[1], a]) # soft Q learning version

            # stroing new information
            state_list.append(s1)
            action_list.append(a)
            reward_list.append(r)
            next_state_list.append(s2)

            s1 = s2
            global_iter += 1

            if global_iter % batch_size == batch_size - 1:
                Q = Q + step_size * Q_update
                # print('TD error:', torch.sum(torch.abs(Q_update)).item())
                Q_update = torch.zeros((*state_shape, num_actions))

            # if global_iter > 100 and global_iter % train_freq == train_freq - 1:
            #     sample_indices = [global_iter-1] # np.random.choice(a=global_iter, size=batch_size)
            #     states = torch.tensor([state_list[sample_index] for sample_index in sample_indices]).long()
            #     actions = torch.tensor([action_list[sample_index] for sample_index in sample_indices]).view(-1,1)
            #     rewards = torch.tensor([reward_list[sample_index] for sample_index in sample_indices]).view(-1,1)
            #     next_states = torch.tensor([next_state_list[sample_index] for sample_index in sample_indices]).long()

            #     current_Q = torch.gather(Q[states[:,0], states[:,1]], dim=1, index=actions)
            #     next_Q    = (Q[next_states[:,0], next_states[:,1]]).detach()
            #     probs     = (next_Q * lamb).softmax(dim=-1)
            #     target_Q  = rewards + discount * torch.sum(probs * next_Q, dim=-1)

            #     TD_error  = torch.nn.MSELoss()(current_Q.flatten(), target_Q.detach().flatten())

            #     optimizer.zero_grad()
            #     TD_error.backward()
            #     optimizer.step()

        # Q[s1[0], s1[1], :] = r 

        episode_rewards.append(episode_reward)

    return { 'Q': Q.detach(),
            'num_cliff_falls': num_cliff_falls,
            'episode_rewards': episode_rewards }

def evaluate_static_policy(env, Q, num_episodes=50, epsilon=0, discount=0.95, lamb=5, strict=True):
    """Evaluate a policy as specified by `Q` without performing any additional
    training, for some `num_episodes` at some action stochasticity
    `epsilon`."""
    episode_rewards = []
    trajectories = []
    while len(episode_rewards) < num_episodes:
        episode_reward = 0
        episode_iter = 0
        env.reset()
        s1 = env.observe()
        trajectory = []
        while not env.is_terminal(s1) and episode_iter < MAX_STEPS_PER_EPISODE:
            if strict:
                a, prob = strict_action(s1, Q, lamb)
            else:
                a, prob = softmax_action(s1, Q, lamb)
            s2, r, _, _ = env.step(a)
            trajectory.append((s1,a,r,s2,prob))
            s1 = s2
            episode_reward += r * (discount ** episode_iter)
            episode_iter += 1
        episode_rewards.append(episode_reward.detach().item())
        trajectories.append(trajectory)
    return np.mean(episode_rewards), trajectories

def compute_loglikelihood(Q, trajectories, offset=0.001, lamb=5):
    loglikelihood_list = []
    policy = policy_from_Q(Q, lamb=lamb)
    for trajectory in trajectories:
        for (s,a,r,s2,prob) in trajectory:
            loglikelihood_list.append(torch.log(policy[s][a] + offset))
    loglikelihood = sum(loglikelihood_list) / len(trajectories)
    return loglikelihood

def generate_instances(basemaze, sample_size=100, lamb=1, demonstrate_lamb=1, discount=0.95, seed=0, noise=0, num_trajectories=1000, action_error_prob=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # randomly initialized mlp
    feature_size = 16
    noise_size = 0
    label_size = basemaze.shape[0] * basemaze.shape[1]
    channel_size_list = [1+noise_size, 64, 64, feature_size]
    mlp = MLP(channel_size_list, activation='ReLU', last_activation='Linear')
    mlp.eval()

    full_labels = []
    full_features = []
    maze_list = []
    dataset = []
    print('Generating problem instances...')
    for sample_id in range(sample_size):
        maze = Maze(basemaze.topology)
        patrol_post = maze.start_coords[0]

        # new state version
        reward_function = torch.zeros(maze.shape)
        # feature_function = torch.normal(0, 5, (maze.shape[0], maze.shape[1], feature_size))
        feature_function = torch.zeros(maze.shape[0], maze.shape[1], feature_size)

        # just the state reward
        for x1 in range(maze.shape[0]):
            for x2 in range(maze.shape[1]):
                distance_to_patrol_post = np.abs(x1 - patrol_post[0]) + np.abs(x2 - patrol_post[1])

                if maze.topology[x1,x2] == '$':
                    reward_function[x1,x2] = np.random.normal(loc=5, scale=1)
                    feature_function[x1,x2,0] = np.random.normal(loc=5, scale=1)
                else:
                    if np.random.random() < 0.2:
                        reward_function[x1,x2] = np.random.normal(loc=-10, scale=1) # penalty
                        feature_function[x1,x2,0] = np.random.normal(loc=0, scale=1)
                    else:
                        reward_function[x1,x2] = np.random.normal(loc=0, scale=1)
                        feature_function[x1,x2,0] = np.random.normal(loc=-5, scale=1)

                # randomness = np.random.random()
                # if randomness < 0.1: # high reward
                #     reward_function[x1,x2] = 5 + np.random.normal(loc=0, scale=2)
                # elif randomness < 0.6: # middle
                #     reward_function[x1,x2] = 0 + np.random.normal(loc=0, scale=2)
                # else: # low
                #     reward_function[x1,x2] = -5 + np.random.normal(loc=0, scale=2)

        reward_function = torch.clip(reward_function, min=-10, max=10)
        if noise_size > 0:
            noisy_reward_function = torch.cat([reward_function.view(-1,1), torch.normal(0,5,(reward_function.numel(), noise_size))], dim=1)
            feature_function = mlp(noisy_reward_function.view(-1,1+noise_size)).detach()
        else:
            feature_function = mlp(reward_function.view(-1,1)).detach()

        # precompute optimal trajectory using Q learning
        def cliffworld(maze, reward_fn):
            return GridWorld(maze=maze, reward_fn=reward_fn, action_error_prob=action_error_prob, state_embedding='unflatten')

        reward_uncertainty_mean, reward_uncertainty_std = torch.tensor(0.0), torch.tensor(1.0)
        real_reward_fn = lambda x: reward_function[x[0],x[1]] + torch.normal(reward_uncertainty_mean, reward_uncertainty_std)
        real_env = cliffworld(maze=maze, reward_fn=real_reward_fn)
        # Q learning implementation 
        Q_initial = 100.0
        res = run_simulation(real_env, method='Q-learning', Q_initial=Q_initial, discount=discount, lamb=lamb)
        Q, num_cliff_falls, episode_rewards = res['Q'], res['num_cliff_falls'], res['episode_rewards']
        print('label:', reward_function)
        print('Q values:', torch.max(Q, dim=-1)[0])
        evaluation, trajectories = evaluate_static_policy(real_env, Q.detach(), num_episodes=num_trajectories, discount=discount, lamb=demonstrate_lamb, strict=False) # fully random policy

        # collecting data
        full_labels.append(reward_function.flatten().view(1,-1))
        full_features.append(feature_function.view(1,-1,feature_size))
        # full_features.append(feature_function.view(-1,feature_size))
        maze_list.append(maze)
        dataset.append([sample_id, maze, feature_function, reward_function, trajectories])

    full_labels = torch.cat(full_labels, dim=0)
    full_features = torch.cat(full_features, dim=0)

    # constructing feature generation model
    if sample_size > 1:
        feature_shape = full_features.shape
        full_features = full_features.view(-1, feature_size)
        full_features = (full_features - torch.mean(full_features, dim=0)) / (torch.std(full_features, dim=0) + 0.001)
        full_features = full_features * (1 - noise) + torch.normal(0, 1, full_features.shape) * noise
        full_features = (full_features - torch.mean(full_features, dim=0)) / (torch.std(full_features, dim=0) + 0.001)
        full_features = full_features.view(feature_shape)

    for i in range(len(dataset)):
        dataset[i][2] = full_features[i] # update features in dataset

    print('feature mean {}, std {}'.format(torch.mean(full_features.view(-1, feature_size), dim=0), torch.std(full_features.view(-1, feature_size), dim=0)))

    return dataset, {'feature size': feature_size, 'label size': label_size}


# def collect_experience(env_fn, ac, seed=0,
#         steps_per_epoch=4000, gamma=0.99, pi_lr=3e-4,
#         vf_lr=1e-3, train_v_iters=80, lam=0.97, max_ep_len=1000,
#         logger_kwargs=dict(), save_freq=10):
#     # Special function to avoid certain slowdowns from PyTorch + MPI combo.
#     setup_pytorch_for_mpi()
# 
#     # Set up logger and save configuration
#     logger = EpochLogger(**logger_kwargs)
#     logger.save_config(locals())
# 
#     # Random seed
#     seed += 10000 * proc_id()
#     torch.manual_seed(seed)
#     np.random.seed(seed)
# 
#     # Instantiate environment
#     env = env_fn()
#     obs_dim = env.observation_space.shape
#     act_dim = env.action_space.shape
# 
#     # Sync params across processes
#     sync_params(ac)
# 
#     # Count variables
#     var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
#     logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)
# 
#     # Set up experience buffer
#     local_steps_per_epoch = int(steps_per_epoch / num_procs())
#     buf = VPGBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma=0.99, lam=0.97)
# 
#     # Prepare for interaction with environment
#     start_time = time.time()
#     o, ep_ret, ep_len = env.reset(), 0, 0
# 
#     # Main loop: collect experience in env and update/log each epoch
#     for t in range(local_steps_per_epoch):
#         a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
# 
#         next_o, r, d, _ = env.step(a)
#         # =========================
#         if callable(reward_fn):
#             print('using customized reward function...')
#             r = reward_fn(o, a)
#         # =========================
#         ep_ret += r
#         ep_len += 1
# 
#         # save and log
#         buf.store(o, a, r, v, logp)
#         logger.store(VVals=v)
# 
#         # Update obs (critical!)
#         o = next_o
# 
#         timeout = ep_len == max_ep_len
#         terminal = d or timeout
#         epoch_ended = t==local_steps_per_epoch-1
# 
#         if terminal or epoch_ended:
#             if epoch_ended and not(terminal):
#                 print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
#             # if trajectory didn't reach terminal state, bootstrap value target
#             if timeout or epoch_ended:
#                 _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
#             else:
#                 v = 0
#             buf.finish_path(v)
#             if terminal:
#                 # only save EpRet / EpLen if trajectory finished
#                 logger.store(EpRet=ep_ret, EpLen=ep_len)
#             o, ep_ret, ep_len = env.reset(), 0, 0
