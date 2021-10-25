import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import torch
import tqdm
import copy
import time
import pickle
import sklearn
import seaborn as sns

from snareworld import NewSnareWorld
from utils import generate_instances
from model import MLP

import sys
sys.path.insert(1, '../')

from diffdqn_transition import PerformanceDQN, DiffDQN, evaluate_phi_loglikelihood
from diffdqn_transition import CWPDIS, PDIS

from stable_baselines3 import DQN


np.set_printoptions(precision=2)
torch.set_printoptions(precision=2, sci_mode=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Snare world')
    parser.add_argument('--method', default='DF', type=str, help='TS (two-stage learning) or DF (decision-focused learning)')
    parser.add_argument('--rl-method', type=str, default='DQN', help='DQN')
    parser.add_argument('--discount', default=0.95, type=float, help='Future discount rate')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--epoch', type=int, default=50, help='total epochs')
    parser.add_argument('--generate', type=int, default=1, help='Generate data or not. If 1 then invoke generate_instances function, if 0 then load data from before directly.')
    parser.add_argument('--softness', type=float, default=1, help='softness of the DQN solver')
    parser.add_argument('--demonstrate-softness', type=float, default=0, help='demonstrate softness used to generate trajectories')
    parser.add_argument('--prefix', type=str, default='test', help='prefix of the saved files')
    parser.add_argument('--sample-size', type=int, default=10, help='sample size')
    parser.add_argument('--warmstart', type=int, default=1, help='warm start')
    parser.add_argument('--recycle', type=int, default=0, help='recycle replay buffer') # default not using the last replay buffer
    parser.add_argument('--regularization', type=float, default=0.1, help='using two-stage loss as the regularization')
    parser.add_argument('--backprop-method', type=int, default=1, help='back-propagation method: 0 -> full hessian, 1 -> Woodbury approximation, 2 -> ignoring the second term')
    parser.add_argument('--ess-const', type=float, default=10, help='the ess weight used to regularize off-policy evaluation')
    parser.add_argument('--noise', type=float, default=0.75, help='noise std added to the generated features')
    parser.add_argument('--number-trajectories', type=int, default=100, help='number of trajectories')
    parser.add_argument('--num-targets', type=int, default=20, help='number of targets')

    args = parser.parse_args()
    print(args)
    rl_method = args.rl_method # Q-learning or AC
    method = args.method
    discount = args.discount
    seed = args.seed
    generate = args.generate
    softness = args.softness
    demonstrate_softness = args.demonstrate_softness
    prefix = args.prefix
    warm_start = args.warmstart
    recycle = args.recycle
    regularization = args.regularization
    backprop_method = args.backprop_method
    ess_const = args.ess_const
    noise = args.noise
    number_trajectories = args.number_trajectories

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = 'cpu' # 'cpu'

    action_error_prob = 0.0
    finding_prob = 0.9
    num_targets = args.num_targets

    sample_size = args.sample_size
    test_size     = int(sample_size * 0.20)
    validate_size = int(sample_size * 0.10)
    train_size    = sample_size - test_size - validate_size

    data_path = 'data/instance_seed{}.p'.format(seed)
    if generate:
        full_dataset, info = generate_instances(num_targets, finding_prob=finding_prob, discount=discount, sample_size=sample_size, num_trajectories=number_trajectories, seed=seed, softness=softness, demonstrate_softness=demonstrate_softness, noise=noise)
        pickle.dump((full_dataset, info), open(data_path, 'wb'))
    else:
        full_dataset, info = pickle.load(open(data_path, 'rb'))
    # train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_loader     = full_dataset[:train_size] # torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader      = full_dataset[train_size:train_size+test_size] # torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    validate_loader  = full_dataset[train_size+test_size:] # torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    feature_size, label_size = info['feature size'], info['label size']

    # Set up model to predict the reward function
    # channel_size_list = [feature_size, 1]
    channel_size_list = [feature_size, 16, 1]
    net = MLP(channel_size_list=channel_size_list).to(device)

    # Learning rate and optimizer
    lr    = 1e-2 # reward neural network learning rate
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    # Since we are only given trajectories, so the negative loglikelihood loss function has to be customized
    def loss_fn(trajectories, transition_prob, deterrence_graph):
        N = len(trajectories)
        T = len(trajectories[0])
        # states = torch.cat([trajectories[n][t][0].view(1,-1) for n in range(N) for t in range(T)], dim=0)
        # actions = torch.cat([trajectories[n][t][1].view(1,-1) for n in range(N) for t in range(T)], dim=0).long().flatten()
        # # rewards = torch.tensor([trajectories[n][t][2] for n in range(N) for t in range(T)])
        # founds = torch.tensor([trajectories[n][t][5] for n in range(N) for t in range(T)])

        # precomputed deterrence effect transition probs
        deterred_transition_probs_precomputed = transition_prob.clone().view(1,-1).repeat(num_targets, 1)
        for action in range(num_targets):
            for neighbor in deterrence_graph[action]:
                deterred_transition_probs_precomputed[action][neighbor] = 0
            deterred_transition_probs_precomputed[action][action] = 0
        deterred_transition_probs_precomputed = torch.clip(deterred_transition_probs_precomputed, min=0, max=1)

        # # compute NLL
        NLL_tensor = torch.zeros(N)
        for t in range(T):
            # matrix version
            states  = torch.cat([trajectories[n][t][0].view(1,-1) for n in range(N)], dim=0).clone()
            actions = torch.cat([trajectories[n][t][1].view(1,-1) for n in range(N)], dim=0).long().flatten().clone()
            # rewards = torch.tensor([trajectories[n][t][2] for n in range(N)]).clone()
            founds  = torch.tensor([trajectories[n][t][5] for n in range(N)]).clone()
            next_states  = torch.cat([trajectories[n][t][3].view(1,-1) for n in range(N)], dim=0).clone()
            if t == 0:
                beliefs = states.clone()

            selected_beliefs = torch.gather(beliefs, dim=1, index=actions.view(-1,1)).flatten().clone()
            # print('selected beliefs', selected_beliefs)
            likelihoods = (1 - founds) * (1 - selected_beliefs * finding_prob) + founds * (selected_beliefs * finding_prob)
            # print(torch.log(likelihoods))

            target_values = torch.gather(beliefs, dim=1, index=actions.view(-1,1)) * (1 - finding_prob)
            new_beliefs_case1 = torch.scatter(beliefs, dim=1, index=actions.view(-1,1), src=target_values)
            new_beliefs_case2 = torch.scatter(beliefs, dim=1, index=actions.view(-1,1), src=torch.zeros(N,1))

            beliefs = (1 - founds.view(-1,1)) * new_beliefs_case1 + founds.view(-1,1) * new_beliefs_case2

            deterred_transition_probs = torch.index_select(deterred_transition_probs_precomputed, dim=0, index=actions)

            beliefs = beliefs + (1 - beliefs) * deterred_transition_probs

            NLL_tensor -= torch.log(likelihoods)

        NLL_loss = torch.mean(NLL_tensor)
        # print('NLL1:', NLL_loss)

        # NLL_loss = 0
        # # print(deterrence_graph)
        # # iterative version
        # for n, trajectory in enumerate(trajectories):
        #     for t, (obs, action, reward, obs2, prob) in enumerate(trajectory):
        #         if t == 0:
        #             belief = obs.clone()
        #             # print('start')
        #             # print(transition_prob)
        #             # print([deterrence_graph[node] for node in range(num_targets)])

        #         # print(belief, obs, action, reward)
        #         if reward == 0:
        #             likelihood = 1 - belief[action].clone() * finding_prob
        #             belief[action] = belief[action].clone() * (1 - finding_prob)
        #         else:
        #             likelihood = belief[action].clone() * finding_prob
        #             belief[action] = 0

        #         deterred_transition_prob = transition_prob.clone()
        #         for neighbor in deterrence_graph[action.item()]:
        #             deterred_transition_prob[neighbor] = 0
        #         deterred_transition_prob[action] = 0

        #         belief = belief + (1 - belief) * deterred_transition_prob 
        #         NLL_loss -= torch.log(likelihood)
        # NLL_loss = NLL_loss / len(trajectories)
        # print('NLL2:', NLL_loss)
        return NLL_loss

    if prefix == '':
        save_path = 'results/{}/{}_{}_seed{}.csv'.format(method, method, rl_method, seed)
    else:
        save_path = 'results/{}/{}_{}_{}_seed{}.csv'.format(method, prefix, method, rl_method, seed)
    f_result = open(save_path, 'w')
    f_result.write('epoch, train loss, train strict eval, train soft eval, validate loss, validate strict eval, validate soft eval, test loss, test strict eval, test soft eval\n')
    f_result.close()

    total_epoch = args.epoch
    pretrained_epoch = 0
    model_dict = {}
    replay_buffer_dict = {}
    for epoch in range(-1,total_epoch):
        f_result = open(save_path, 'a')
        # ------------------------ training -------------------------------
        for mode, data_loader in [('train', train_loader), ('validate', validate_loader), ('test', test_loader)]:
            TS_loss_list, DF_loss_list = [], []
            loss_list, strict_evaluation_list, soft_evaluation_list, loglikelihood_list = [], [], [], []
            likelihood_time_list, forward_time_list, evaluation_time_list, backward_time_list = [], [], [], []
            ess_list, cwpdis_list = [], []
            if mode == 'train':
                net.train()
            else:
                net.eval()

            if epoch <= 0:
                evaluated = True
            elif epoch < pretrained_epoch:
                evaluated = False
            elif (method == 'TS') and (epoch < total_epoch -1): # and (mode == 'train'):
                evaluated = True
            else:
                evaluated = True

            with tqdm.tqdm(data_loader) as tqdm_loader:
                for index, (data_id, deterrence_graph, initial_distribution, feature, label, real_trajectories) in enumerate(tqdm_loader):
                    feature, label = feature.to(device), label.to(device)
                    # label = torch.clip(label + torch.normal(0, 0.1, size=label.shape), min=0, max=1) # adding random noise everytime
                    start_time = time.time()
                    prediction = net(feature.reshape(-1,feature_size)).flatten()
                    prediction_detach = prediction.detach()
                    if epoch < 0:
                        prediction = label.detach().clone()

                    # print(label)
                    # print(prediction)
                    # NewSnareWorld wrapper for later use
                    snareworld_wrapper = lambda x, y=num_targets, g=deterrence_graph, initial_distribution=initial_distribution, z=finding_prob: NewSnareWorld(num_targets=y,transition_prob=x,deterrence_graph=g,initial_distribution=initial_distribution,finding_prob=z)

                    loss = loss_fn(real_trajectories, prediction, deterrence_graph) - loss_fn(real_trajectories, label, deterrence_graph)
                    loss_list.append(loss.detach().item())

                    likelihood_time_list.append(time.time() - start_time)
                    start_time = time.time()

                    if evaluated:
                        # new state version: reward_fn(new_s)
                        real_env             = NewSnareWorld(num_targets, transition_prob=prediction, true_transition_prob=label, deterrence_graph=deterrence_graph, initial_distribution=initial_distribution, finding_prob=finding_prob)
    
                        # DQN implementation
                        if epoch <= 0: # or epoch == total_epoch - 1:
                            model_parameters = None
                            learning_starts = 1000
                            min_num_iters = 10000
                            load_replay_buffer = False
                            save_replay_buffer = False
                            dynamic_softness = softness
                            verbose = 0
                            baseline = 0
                        elif data_id in model_dict:
                            if warm_start:
                                model_parameters = model_dict[data_id]['model']
                            else:
                                model_parameters = None
                            if recycle:
                                load_replay_buffer = True
                            else:
                                load_replay_buffer = False
                            save_replay_buffer = True
                            learning_starts = 1000
                            min_num_iters = 10000
                            dynamic_softness = softness # 1 + epoch / total_epoch * softness
                            verbose = 0
                            baseline = model_dict[data_id]['baseline']
                        else:
                            model_parameters = None
                            learning_starts = 1000
                            min_num_iters = 10000
                            load_replay_buffer = False
                            save_replay_buffer = True
                            dynamic_softness = softness # 1 + epoch / total_epoch * softness
                            verbose = 0
                            baseline = 0

                        net_arch = [64, 64]
                        policy_kwargs = {'softness': dynamic_softness, 'net_arch': net_arch, 'activation_fn': torch.nn.ReLU, 'squash_output': False, 'scale': 1}
                        strict_policy_kwargs = {'softness': 100, 'net_arch': net_arch, 'activation_fn': torch.nn.ReLU, 'squash_output': False, 'scale': 1}

                        diffdqn_solver = DiffDQN(
                                env_wrapper=snareworld_wrapper,
                                policy_kwargs=policy_kwargs,
                                learning_starts=learning_starts,
                                learning_rate=0.0001,
                                min_num_iters=min_num_iters,
                                model_initial_parameters=model_parameters,
                                discount=discount,
                                target_update_interval=1000,
                                buffer_size=100000,
                                device=device,
                                verbose=verbose,
                                load_replay_buffer=load_replay_buffer,
                                save_replay_buffer=save_replay_buffer,
                                replay_buffer_dict=replay_buffer_dict,
                                data_id=data_id,
                                baseline=baseline,
                                backprop_method=backprop_method,
                                seed=seed,
                                )

                        model_parameters = diffdqn_solver(prediction)

                        forward_time_list.append(time.time() - start_time)
                        start_time = time.time()

                        # ---------------- online evaluation ---------------
                        # performance_eval = PerformanceDQN(env=real_env, policy_kwargs=policy_kwargs, discount=discount, device=device) # soften the Q policy within PerformanceQ
                        # soft_evaluation = performance_eval(model_parameters)

                        # ---------------- offline evaluation --------------
                        performance_eval = CWPDIS(env=real_env, policy_kwargs=policy_kwargs, trajectories=real_trajectories, discount=discount, device=device)
                        cwpdis, ess = performance_eval(model_parameters)
                        soft_evaluation = cwpdis - ess_const/torch.sqrt(ess + 0.001)

                        # -------------------- simulation ------------------
                        model = DQN("MlpPolicy", real_env, policy_kwargs=strict_policy_kwargs, verbose=0, gamma=discount, buffer_size=0, seed=seed)
                        model.policy.eval()
                        model.policy.load_from_vector(model_parameters.detach())
                        model.policy.to(device)
                        _, _, _, total_reward_tensor = evaluate_phi_loglikelihood(real_env, model, num_episodes=100, discount=discount, deterministic=True)
                        strict_evaluation = np.mean(total_reward_tensor)
                        del model

                        # ------------------- strict OPE -------------------
                        # strict_policy_kwargs = {'softness': 100, 'net_arch': net_arch, 'activation_fn': torch.nn.ReLU, 'squash_output': False, 'scale': 1}
                        # strict_performance_eval = PDIS(env=real_env, policy_kwargs=strict_policy_kwargs, trajectories=real_trajectories, discount=discount, device=device)
                        # strict_cwpdis, _ = strict_performance_eval(model_parameters.detach())

                        strict_evaluation_list.append(strict_evaluation)
                        soft_evaluation_list.append(soft_evaluation.detach().item())
                        ess_list.append(ess.detach().item())
                        cwpdis_list.append(cwpdis.detach().item())

                        evaluation_time_list.append(time.time() - start_time)
                        if epoch >= 0:
                            model_dict[data_id] = {'model': model_parameters.detach(), 'baseline': cwpdis.detach()}

                    start_time = time.time()
                    # ================== backprop ====================
                    if ((method == 'TS') and (mode == 'train') and (epoch > 0)) or ((method == 'DF') and (mode == 'train') and (epoch < pretrained_epoch) and (epoch > 0)):
                        optimizer.zero_grad()
                        loss.backward()
                        # for parameter in net.parameters():
                        #     print('norm:', torch.norm(parameter.grad))
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 1, norm_type=2)
                        optimizer.step()
                        # TS_loss_list.append(loss)
    
                    elif (method == 'DF') and (mode == 'train') and (epoch > 0):
                        optimizer.zero_grad()
                        (-soft_evaluation + loss * regularization).backward()
                        # for parameter in net.parameters():
                        #     print('norm:', torch.norm(parameter.grad))
                        if any([torch.isnan(parameter.grad).any() for parameter in net.parameters()]):
                            print('Found nan!! Not backprop through this instance!!')
                            optimizer.zero_grad() # grad contains nan so not backprop through this instance
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 1, norm_type=2)
                        optimizer.step()

                    backward_time = time.time() - start_time
                    backward_time_list.append(time.time() - start_time)
                    if epoch > 0:
                        f_computation = open('computation/{}_num_targets{}.csv'.format(backprop_method, num_targets), 'a')
                        f_computation.write('{}, {}, {}, {}\n'.format(num_targets, epoch, backprop_method, backward_time))
                        f_computation.close()

                    tqdm_loader.set_postfix(
                            loss='{:.3f}'.format(np.mean(loss_list)),
                            strict_eval='{:.3f}'.format(np.mean(strict_evaluation_list)),
                            soft_eval='{:.3f}'.format(np.mean(soft_evaluation_list)),
                            cwpdis='{:.3f}'.format(np.mean(cwpdis_list)),
                            ess='{:.3f}'.format(np.mean(ess_list)),
                            )

                if evaluated:
                    print('Epoch {} with average {} loss: {}, strict evaluation: {}, soft eval: {}, likelihood time: {}, forward time: {}, evaluation time: {}, backward time: {}'.format(
                        epoch, mode, np.mean(loss_list), np.mean(strict_evaluation_list), np.mean(soft_evaluation_list),
                        np.mean(likelihood_time_list), np.mean(forward_time_list),
                        np.mean(evaluation_time_list), np.mean(backward_time_list)
                        ))
                else:
                    print('Epoch {} with average {} loss: {}, likelihood time: {}, forward time: {}, evaluation time: {}, backward time: {}'.format(
                        epoch, mode, np.mean(loss_list),
                        np.mean(likelihood_time_list), np.mean(forward_time_list),
                        np.mean(evaluation_time_list), np.mean(backward_time_list)
                        ))

                if mode == 'train':
                    f_result.write('{}, {}, {}, {}, '.format(epoch, np.mean(loss_list), np.mean(strict_evaluation_list), np.mean(soft_evaluation_list)))
                elif mode == 'validate':
                    f_result.write('{}, {}, {}, '.format(np.mean(loss_list), np.mean(strict_evaluation_list), np.mean(soft_evaluation_list)))
                else:
                    f_result.write('{}, {}, {}\n'.format(np.mean(loss_list), np.mean(strict_evaluation_list), np.mean(soft_evaluation_list)))

        f_result.close()
