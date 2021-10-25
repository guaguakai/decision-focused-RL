import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import torch
import pickle
import tqdm
import copy
import sklearn
import seaborn as sns
import time

from gridworld import *
from model import MLP
from utils import run_simulation, generate_instances, evaluate_static_policy, policy_from_Q, compute_loglikelihood

import sys
sys.path.insert(1, '../')

from diffq import plot_policy, evaluate_phi_loglikelihood, PerformanceQ, DiffQ, CWPDIS

np.set_printoptions(precision=2)
torch.set_printoptions(precision=2, sci_mode=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grid world')
    parser.add_argument('--method', default='DF', type=str, help='TS (two-stage learning) or DF (decision-focused learning)')
    parser.add_argument('--rl-method', type=str, default='Q-learning', help='Q-learning')
    parser.add_argument('--Q-initial', default=100, type=float, help='Initialization of the Q value')
    parser.add_argument('--discount', default=0.95, type=float, help='Future discount rate')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--epoch', type=int, default=50, help='total epochs')
    parser.add_argument('--generate', type=int, default=1, help='Generate data or not. If 1 then invoke generate_instances function, if 0 then load data from before directly.')
    parser.add_argument('--prefix', type=str, default='', help='prefix of the saved files')
    parser.add_argument('--sample-size', type=int, default=10, help='sample size')
    parser.add_argument('--regularization', type=float, default=0.1, help='using two-stage loss as the regularization')
    parser.add_argument('--backprop-method', type=int, default=1, help='back-propagation method: 0 -> full hessian, 1 -> Woodbury approximation, 2 -> ignoring the second term')
    parser.add_argument('--ess-const', type=float, default=1, help='the ess weight used to regularize off-policy evaluation')
    parser.add_argument('--noise', type=float, default=0.75, help='noise std added to the generated features')
    parser.add_argument('--number-trajectories', type=int, default=100, help='number of trajectories')
    parser.add_argument('--softness', type=float, default=0.1, help='softness')
    parser.add_argument('--demonstrate-softness', type=float, default=1, help='softness to generate demonstrate trajectories')
    parser.add_argument('--problem-size', type=int, default=5, help='height/width of the gridworld')

    args = parser.parse_args()
    print(args)
    rl_method = args.rl_method # Q-learning or AC
    Q_initial_default = args.Q_initial # for Q-learning only
    method = args.method
    discount = args.discount
    seed = args.seed
    generate = args.generate
    prefix = args.prefix
    regularization = args.regularization
    backprop_method = args.backprop_method
    ess_const = args.ess_const
    noise = args.noise
    number_trajectories = args.number_trajectories
    problem_size = args.problem_size

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = 'cpu' # 'cpu'
    def generate_maze_description(problem_size):
        # Generate mazes of problem_size x 5

        assert problem_size >= 2

        descr_beg = '.' * 4 + '$'
        descr_end = 'o' + '.' * 4
        descr_mid = ['.' * 5 for _ in range(problem_size - 2)]
        descr = [descr_beg, *descr_mid, descr_end]

        return descr
    basemaze = Maze(generate_maze_description(problem_size))

    # softness choice 
    lamb = args.softness
    demonstrate_lamb = args.demonstrate_softness
    action_error_prob = 0.0
    if rl_method == 'Q-learning':
        def cliffworld(maze, reward_fn):
            return GridWorld(maze=maze, reward_fn=reward_fn, action_error_prob=action_error_prob, state_embedding='unflatten')
    elif rl_method == 'AC':
        def cliffworld(maze, reward_fn):
            return GridWorld(maze=maze, reward_fn=reward_fn, action_error_prob=action_error_prob, state_embedding='onehot')

    sample_size = args.sample_size 
    test_size = int(sample_size * 0.2)
    validate_size = int(sample_size * 0.1)
    train_size = sample_size - test_size - validate_size

    data_path = 'data/instance_seed{}.p'.format(seed)
    if generate:
        full_dataset, info = generate_instances(basemaze, sample_size=sample_size, seed=seed, lamb=lamb, demonstrate_lamb=demonstrate_lamb, discount=discount, noise=noise, num_trajectories=number_trajectories, action_error_prob=action_error_prob)
        pickle.dump((full_dataset, info), open(data_path, 'wb'))
    else:
        full_dataset, info = pickle.load(open(data_path, 'rb'))

    train_loader = full_dataset[:train_size] # torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader  = full_dataset[train_size:train_size+test_size] # torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    validate_loader  = full_dataset[train_size+test_size:] # torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    feature_size, label_size = info['feature size'], info['label size']

    # Set up model to predict the reward function
    channel_size_list = [feature_size, 16, 1]
    # channel_size_list = [feature_size, 16, label_size]
    net = MLP(channel_size_list=channel_size_list).to(device)

    # Learning rate and optimizer
    lr    = 1e-2 # reward neural network learning rate
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    # loss_fn = torch.nn.L1Loss()
    mse_fn = torch.nn.MSELoss()
    def loss_fn(trajectories, prediction):
        total_loss = 0
        for trajectory in trajectories:
            for (s, a, r, s2, prob) in trajectory:
                total_loss += mse_fn(prediction[s2], r)
        total_loss /= len(trajectories)
        return total_loss

    if prefix == '':
        save_path = 'results/{}/{}_{}_seed{}.csv'.format(method, method, rl_method, seed)
    else:
        save_path = 'results/{}/{}_{}_{}_seed{}.csv'.format(method, prefix, method, rl_method, seed)

    f_result = open(save_path, 'w')
    f_result.write('epoch, train loss, train strict eval, train soft eval, backward time, validate loss, validate strict eval, validate soft eval, test loss, test strict eval, test soft eval,\n')
    f_result.close()

    total_epoch = args.epoch
    pretrained_epoch = 0
    model_dict = {}
    img_count = 0
    for epoch in range(-1,total_epoch):
        f_result = open(save_path, 'a')
        # sklearn.utils.shuffle(train_loader)
        # x_input  = torch.arange(-10, 10, 0.1).view(-1,1)
        # x_input  = torch.cat([x_input, torch.normal(mean=torch.zeros(x_input.shape[0], feature_size-1), std=torch.ones(x_input.shape[0], feature_size-1) * 5)], dim=1).to(device)
        # y_output = net(x_input.view(-1,feature_size)).flatten().detach().to('cpu')
        # # y_output = y_output / torch.max(torch.abs(y_output)) * 10
        # y_output = y_output.numpy()

        # g = sns.scatterplot(x=x_input[:,0].detach().to('cpu').numpy(), y=y_output)

        # g.set(ylim=(-10,10))
        # plt.xlabel('Feature')
        # plt.ylabel('Reward')
        # plt.title('Epoch {}'.format(epoch))
        # plt.tight_layout()
        # plt.savefig('results/plots/{}_{}/model/epoch{}.png'.format(method, rl_method, epoch))
        # plt.close()

        for mode, data_loader in [('train', train_loader), ('validate', validate_loader), ('test', test_loader)]:
            TS_loss_list, DF_loss_list = [], []
            loss_list, strict_evaluation_list, cwpdis_list, ess_list, loglikelihood_list = [], [], [], [], []
            likelihood_time_list, forward_time_list, evaluation_time_list, backward_time_list = [], [], [], []

            if mode == 'train':
                net.train()
            else:
                net.eval()

            if epoch <= 0:
                evaluated = True
            elif (method == 'TS') and (epoch < total_epoch -1): # and (mode == 'train'):
                evaluated = True
            elif epoch < pretrained_epoch:
                evaluated = True
            else:
                evaluated = True

            with tqdm.tqdm(data_loader) as tqdm_loader:
                for index, (data_id, maze, feature, label, real_trajectories) in enumerate(tqdm_loader):
                    feature, label = feature.to(device), label.to(device)
                    start_time = time.time()
                    prediction = net(feature.reshape(-1,feature_size)).reshape(maze.shape[0], maze.shape[1]) # new state version
                    # normalize the prediction to avoid hallucinating predictions (rewards)
                    # prediction = prediction / torch.max(torch.abs(prediction)).detach() * 10
                    prediction_detach = prediction.detach()

                    label = label.reshape(maze.shape[0], maze.shape[1]) # new state version
                    if epoch < 0:
                        prediction = label

                    # print(label)
                    # print(prediction)

                    # loss = loss_fn(label, prediction)
                    loss = loss_fn(real_trajectories, prediction) - loss_fn(real_trajectories, label)
                    loss_list.append(loss.item())

                    likelihood_time_list.append(time.time() - start_time)
                    start_time = time.time()

                    reward_uncertainty_mean, reward_uncertainty_std = torch.tensor(0.0), torch.tensor(1.0)
                    if evaluated:
                        # new state version: reward_fn(new_s)
                        predicted_reward_fn        = lambda x: prediction[x[0],x[1]]        + torch.normal(reward_uncertainty_mean, reward_uncertainty_std) 
                        predicted_reward_fn_detach = lambda x: prediction_detach[x[0],x[1]] + torch.normal(reward_uncertainty_mean, reward_uncertainty_std)
                        real_reward_fn             = lambda x: label[x[0],x[1]]             + torch.normal(reward_uncertainty_mean, reward_uncertainty_std)

                        predicted_env        = cliffworld(maze=maze, reward_fn=predicted_reward_fn)
                        predicted_env_detach = cliffworld(maze=maze, reward_fn=predicted_reward_fn_detach)
                        real_env             = cliffworld(maze=maze, reward_fn=real_reward_fn)
    
                        # Q learning implementation
                        if rl_method == 'Q-learning':
                            def cliffworld_wrapper(env_parameter):
                                reward_fn = lambda x: env_parameter[x[0],x[1]]
                                return GridWorld(maze=maze, reward_fn=reward_fn, action_error_prob=action_error_prob, state_embedding='unflatten')

                            if data_id in model_dict:
                                # Q_initial = Q_initial_default
                                Q_initial = model_dict[data_id]
                                # Q_initial = torch.max(model_dict[data_id]).item()
                                min_num_episodes = 1000
                                min_num_iters = 10000
                            else:
                                Q_initial = Q_initial_default
                                min_num_episodes = 1000
                                min_num_iters = 100000

                            diffq_solver = DiffQ(
                                    env_wrapper=cliffworld_wrapper,
                                    min_num_episodes=min_num_episodes,
                                    min_num_iters=min_num_iters,
                                    Q_initial=Q_initial,
                                    discount=discount,
                                    lamb=lamb,
                                    device=device,
                                    backprop_method=backprop_method,
                                    )
                            Q = diffq_solver(prediction)

                            # print(prediction)
                            # print(torch.max(Q, dim=-1)[0])

                            forward_time_list.append(time.time() - start_time)
                            start_time = time.time()

                            if epoch >= 0:
                                model_dict[data_id] = Q.detach()

                            # ------------------ online evaluation ------------------
                            # performance_eval = PerformanceQ(env=real_env, discount=discount, lamb=lamb, device=device) # soften the Q policy within PerformanceQ
                            # soft_evaluation = performance_eval(Q)

                            # ------------------ offline evaluation -----------------
                            cwpdis, ess = CWPDIS(real_trajectories, Q, discount=discount, lamb=lamb)
                            soft_evaluation = cwpdis - ess_const / torch.sqrt(ess)

                            loglikelihood = compute_loglikelihood(Q, real_trajectories, offset=0.0, lamb=lamb)
                            strict_evaluation, proposed_trajectories = evaluate_static_policy(real_env, Q, discount=discount, lamb=lamb)

                            evaluation_time_list.append(time.time() - start_time)

                            # plot_policy(predicted_env_detach, Q.detach(), 'results/plots/{}_{}/{}_{}'.format(method, rl_method, mode, index), evaluation=strict_evaluation)

                            # if mode == 'train' and epoch > 0: # for animation only
                            #     plot_policy(predicted_env_detach, Q.detach(), 'results/plots/{}_{}/animation/{}_{}'.format(method, rl_method, mode, img_count), evaluation=strict_evaluation)
                            #     img_count += 1

                        strict_evaluation_list.append(strict_evaluation)
                        cwpdis_list.append(cwpdis.detach().item())
                        ess_list.append(ess.detach().item())
                        loglikelihood_list.append(loglikelihood.detach().item())

                    # ================== backprop ====================
                    start_time = time.time()
                    if ((method == 'TS') and (mode == 'train') and (epoch > 0)) or ((method == 'DF') and (mode == 'train') and (epoch < pretrained_epoch) and (epoch > 0)):
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 1e-0, norm_type=2)
                        optimizer.step()
                        # TS_loss_list.append(loss)
    
                    elif (method == 'DF') and (mode == 'train') and (epoch > 0):
                        (-soft_evaluation + loss * regularization).backward(retain_graph=True)
                        if any([torch.isnan(parameter.grad).any() for parameter in net.parameters()]):
                            print('Found nan!! Not backprop through this instance!!')
                            optimizer.zero_grad() # grad contains nan so not backprop through this instance
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 1, norm_type=2)
                        optimizer.step()
                        # DF_loss_list.append(-soft_evaluation + loss * regularization)

                    # print('strict performance:', strict_evaluation)
                    # print('soft performance:', soft_evaluation)
                    # print('loglikelihood:', loglikelihood.detach().item())
                    # print('feature:', feature)
                    # print('label:', label)
                    # print('prediction:', prediction.detach())
                    # print('Q:', torch.max(Q.detach(), dim=2)[0])

                    backward_time_list.append(time.time() - start_time)

                    tqdm_loader.set_postfix(
                            loss='{:.3f}'.format(np.mean(loss_list)),
                            strict_eval='{:.3f}'.format(np.mean(strict_evaluation_list)),
                            cwpdis='{:.3f}'.format(np.mean(cwpdis_list)),
                            ess='{:.3f}'.format(np.mean(ess_list)),
                            loglikelihood='{:.3f}'.format(np.mean(loglikelihood_list))
                            )

                if evaluated:
                    print('Epoch {} with average {} loss: {}, strict evaluation: {}, cwpdis: {}, likelihood time: {}, forward time: {}, evaluation time: {}, backward time: {}'.format(
                        epoch, mode, np.mean(loss_list), np.mean(strict_evaluation_list), np.mean(cwpdis_list),
                        np.mean(likelihood_time_list), np.mean(forward_time_list),
                        np.mean(evaluation_time_list), np.mean(backward_time_list)))
                else:
                    print('Epoch {} with average {} loss: {}, likelihood time: {}, forward time: {}, evaluation time: {}, backward time: {}'.format(
                        epoch, mode, np.mean(loss_list),
                        np.mean(likelihood_time_list), np.mean(forward_time_list),
                        np.mean(evaluation_time_list), np.mean(backward_time_list)))

                if mode == 'train':
                    f_result.write('{}, {}, {}, {}, {}, '.format(epoch, np.mean(loss_list), np.mean(strict_evaluation_list), np.mean(cwpdis_list), np.mean(backward_time_list)))
                elif mode == 'validate':
                    f_result.write('{}, {}, {}, '.format(np.mean(loss_list), np.mean(strict_evaluation_list), np.mean(cwpdis_list)))
                else:
                    f_result.write('{}, {}, {}\n'.format(np.mean(loss_list), np.mean(strict_evaluation_list), np.mean(cwpdis_list)))

            # if ((method == 'TS') and (mode == 'train') and (epoch > 0)) or ((method == 'DF') and (mode == 'train') and (epoch < pretrained_epoch) and (epoch > 0)):
            #     optimizer.zero_grad()
            #     TS_loss = sum(TS_loss_list) / len(TS_loss_list)
            #     TS_loss.backward()
            #     torch.nn.utils.clip_grad_norm_(net.parameters(), 10, norm_type=2)
            #     optimizer.step()
            # elif (method == 'DF') and (mode == 'train') and (epoch > 0):
            #     optimizer.zero_grad()
            #     DF_loss = sum(DF_loss_list) / len(DF_loss_list)
            #     DF_loss.backward()
            #     if any([torch.isnan(parameter.grad).any() for parameter in net.parameters()]):
            #         print('Found nan!! Not backprop through this instance!!')
            #         optimizer.zero_grad() # grad contains nan so not backprop through this instance
            #     torch.nn.utils.clip_grad_norm_(net.parameters(), 1, norm_type=2)
            #     optimizer.step()

        f_result.close()
