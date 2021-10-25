import numpy as np
import argparse
import random
import torch
import tqdm
import copy
import time
import pickle
import sklearn

from TBWorld import TBWorld, TBBeliefWorld
from utils import generate_instances
from model import MLP
from pathlib import Path
from functools import partial

import os
import sys
sys.path.insert(1, '../')

from diffdqn_transition import PerformanceDQN, DiffDQN, evaluate_phi_loglikelihood, CWPDIS

from stable_baselines3 import DQN


np.set_printoptions(precision=2)
torch.set_printoptions(precision=2, sci_mode=False)
torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TB Adherence Problem')
    parser.add_argument('--method', default='DF', type=str,
                        help='TS (two-stage learning) or DF (decision-focused learning)')
    parser.add_argument('--rl-method', type=str, default='DQN', help='DQN')
    parser.add_argument('--Q-initial', default=0, type=float,
                        help='Initialization of the Q value')
    parser.add_argument('--discount', default=0.95,
                        type=float, help='Future discount rate')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--epoch', type=int, default=100, help='total epochs')
    parser.add_argument('--generate', type=int, default=0,
                        help='Generate data or not. If 1 then invoke generate_instances function, if 0 then load data from before directly.')
    parser.add_argument('--softness', type=float, default=5,
                        help='softness of the DQN solver')
    parser.add_argument('--demonstrate-softness', type=float, default=0,
                        help='demonstrate softness used to generate trajectories')
    parser.add_argument('--prefix', type=str, default='test',
                        help='prefix of the saved files')
    parser.add_argument('--sample-size', type=int,
                        default=10, help='sample size')
    parser.add_argument('--warmstart', type=int, default=1, help='warm start')
    # default not using the last replay buffer
    parser.add_argument('--recycle', type=int,
                        default=0, help='recycle replay buffer')
    parser.add_argument('--regularization', type=float, default=0.1,
                        help='using two-stage loss as the regularization')
    parser.add_argument('--backprop-method', type=int, default=1,
                        help='back-propagation method: 0 -> full hessian, 1 -> Woodbury approximation, 2 -> ignoring the second term')
    parser.add_argument('--ess-const', type=float, default=10,
                        help='the ess weight used to regularize off-policy evaluation')
    parser.add_argument('--noise', type=float, default=0.25,
                        help='noise std added to the generated features')
    parser.add_argument('--number-trajectories', type=int,
                        default=100, help='number of trajectories')
    parser.add_argument('--effect-size', type=float, default=0.4,
                        help='size of the action effect (0 -> no effect, 1 -> guaranteed adherence)')
    parser.add_argument('--num-patients', type=int,
                        default=5, help='number of patients to intervene on')
    parser.add_argument('--start-adhering', type=int,
                        default=0, help='whether patients start by adhering or not')
    parser.add_argument('--fully-observable', type=int,
                        default=0, help="whether patients' adherences are fully observable or not")
    parser.add_argument('--patient-data', type=str, help="location of the csv file containing patient data")
    args = parser.parse_args()
    print(args)

    rl_method = args.rl_method  # Q-learning or AC
    Q_initial_default = args.Q_initial  # for Q-learning only
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
    NUM_PATIENTS = args.num_patients
    EFFECT_SIZE = args.effect_size
    START_ADHERING = args.start_adhering
    env_type = TBWorld if args.fully_observable else TBBeliefWorld
    data_file = args.patient_data

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = 'cpu'  # 'cpu'

    sample_size = args.sample_size
    test_size = int(sample_size * 0.20)
    validate_size = int(sample_size * 0.10)
    train_size = sample_size - test_size - validate_size

    # Generate trajectories to perform offline RL on
    generate_kwargs = {'env_type': env_type, 'NUM_PATIENTS': NUM_PATIENTS, 'EFFECT_SIZE': EFFECT_SIZE, 'discount': discount, 'sample_size': sample_size, 'num_trajectories': number_trajectories,
                       'seed': seed, 'softness': softness, 'demonstrate_softness': demonstrate_softness, 'noise_fraction': noise, 'START_ADHERING': START_ADHERING, 'data_file': data_file}
    data_path = f'data/{"_".join(f"{k}{v}" for k, v in generate_kwargs.items())}.p'
    if not generate and os.path.exists(data_path):
        full_dataset, info = pickle.load(open(data_path, 'rb'))
        assert len(full_dataset) == sample_size
    else:
        full_dataset, info = generate_instances(**generate_kwargs)
        pickle.dump((full_dataset, info), open(data_path, 'wb'))
    # train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # Split dataset into train, val and test
    # torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    train_loader = full_dataset[:train_size]
    # torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_loader = full_dataset[train_size:train_size + test_size]
    # torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    validate_loader = full_dataset[train_size + test_size:]

    feature_size, label_size = info['feature size'], info['label size']

    # Set up model to predict the reward function
    # channel_size_list = [feature_size, 1]
    channel_size_list = [feature_size, 16, 8]
    net = MLP(channel_size_list=channel_size_list).to(device)

    # Learning rate and optimizer
    lr = 1e-2  # reward neural network learning rate
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    if prefix == '':
        save_path = 'results/{}/{}_{}_seed{}.csv'.format(
            method, method, rl_method, seed)
    else:
        save_path = 'results/{}/{}_{}_{}_seed{}.csv'.format(
            method, prefix, method, rl_method, seed)
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    f_result = open(save_path, 'w')
    f_result.write('epoch, train loss, train strict eval, train soft eval, validate loss, validate strict eval, validate soft eval, test loss, test strict eval, test soft eval\n')
    f_result.close()

    total_epoch = args.epoch
    pretrained_epoch = 0
    model_dict = {}
    replay_buffer_dict = {}
    for epoch in range(-1, total_epoch):
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
            # and (mode == 'train'):
            elif (method == 'TS') and (epoch < total_epoch - 1):
                evaluated = True
            else:
                evaluated = True

            with tqdm.tqdm(data_loader) as tqdm_loader:
                for index, (data_id, NUM_PATIENTS, EFFECT_SIZE, feature, label, real_trajectories) in enumerate(tqdm_loader):
                    feature, label = feature.to(device), label.to(device)
                    # label = torch.clip(label + torch.normal(0, 0.1, size=label.shape), min=0, max=1) # adding random noise everytime
                    start_time = time.time()
                    prediction_raw = net(feature.reshape(-1, feature_size)).view(label.shape)
                    prediction = prediction_raw / prediction_raw.sum(dim=-1).unsqueeze(-1)  # ensure predicted probabilities sum to 1
                    prediction_detach = prediction.detach()
                    if epoch < 0:
                        prediction = label.detach().clone()

                    # print(label.shape)
                    # print(prediction.shape)

                    # Environment wrapper for later use
                    env_wrapper = partial(env_type, NUM_PATIENTS=NUM_PATIENTS, START_ADHERING=START_ADHERING)

                    loss = env_type.loss_fn(real_trajectories, prediction) - \
                        env_type.loss_fn(real_trajectories, label)
                    loss_list.append(loss.detach().item())

                    likelihood_time_list.append(time.time() - start_time)
                    start_time = time.time()

                    if evaluated:
                        # new state version: reward_fn(new_s)
                        real_env = env_wrapper(transition_prob=prediction, true_transition_prob=label)

                        # DQN implementation
                        if epoch < 0:  # or epoch == total_epoch - 1:
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
                            dynamic_softness = softness  # 1 + epoch / total_epoch * softness
                            verbose = 0
                            baseline = model_dict[data_id]['baseline']
                        else:
                            model_parameters = None
                            learning_starts = 1000
                            min_num_iters = 10000
                            load_replay_buffer = False
                            save_replay_buffer = True
                            dynamic_softness = softness  # 1 + epoch / total_epoch * softness
                            verbose = 0
                            baseline = 0

                        net_arch = [64, 64]
                        policy_kwargs = {'softness': dynamic_softness, 'net_arch': net_arch,
                                         'activation_fn': torch.nn.ReLU, 'squash_output': False, 'scale': 1}
                        strict_policy_kwargs = {'softness': 100, 'net_arch': net_arch,
                                                'activation_fn': torch.nn.ReLU, 'squash_output': False, 'scale': 1}

                        diffdqn_solver = DiffDQN(
                            env_wrapper=env_wrapper,
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
                        performance_eval = CWPDIS(env=real_env, policy_kwargs=policy_kwargs,
                                                  trajectories=real_trajectories, discount=discount, device=device)
                        cwpdis, ess = performance_eval(model_parameters)
                        soft_evaluation = cwpdis - \
                            ess_const / torch.sqrt(ess + 0.001)

                        # -------------------- simulation ------------------
                        model = DQN("MlpPolicy", real_env, policy_kwargs=strict_policy_kwargs,
                                    verbose=verbose, gamma=discount, buffer_size=0, seed=seed)
                        model.policy.eval()
                        model.policy.load_from_vector(
                            model_parameters.detach())
                        model.policy.to(device)
                        _, _, _, total_reward_tensor = evaluate_phi_loglikelihood(
                            real_env, model, num_episodes=100, discount=discount, deterministic=True)
                        strict_evaluation = np.mean(total_reward_tensor)
                        del model

                        # ------------------- strict OPE -------------------
                        # strict_policy_kwargs = {'softness': 100, 'net_arch': net_arch, 'activation_fn': torch.nn.ReLU, 'squash_output': False, 'scale': 1}
                        # strict_performance_eval = CWPDIS(env=real_env, policy_kwargs=strict_policy_kwargs, trajectories=real_trajectories, discount=discount, device=device)
                        # strict_cwpdis, _ = strict_performance_eval(model_parameters.detach())

                        strict_evaluation_list.append(strict_evaluation)
                        soft_evaluation_list.append(
                            soft_evaluation.detach().item())
                        ess_list.append(ess.detach().item())
                        cwpdis_list.append(cwpdis.detach().item())

                        evaluation_time_list.append(time.time() - start_time)
                        if epoch >= 0:
                            model_dict[data_id] = {
                                'model': model_parameters.detach(), 'baseline': cwpdis.detach()}

                    start_time = time.time()
                    # ================== backprop ====================
                    if ((method == 'TS') and (mode == 'train') and (epoch > 0)) or ((method == 'DF') and (mode == 'train') and (epoch < pretrained_epoch) and (epoch > 0)):
                        optimizer.zero_grad()
                        loss.backward()
                        for parameter in net.parameters():
                            print('norm:', torch.norm(parameter.grad))
                        torch.nn.utils.clip_grad_norm_(
                            net.parameters(), 1, norm_type=2)
                        optimizer.step()
                        # TS_loss_list.append(loss)

                    elif (method == 'DF') and (mode == 'train') and (epoch > 0):
                        # pdb.set_trace()
                        optimizer.zero_grad()
                        (-soft_evaluation + loss * regularization).backward()
                        for parameter in net.parameters():
                            print('norm:', torch.norm(parameter.grad))
                        if any([torch.isnan(parameter.grad).any() for parameter in net.parameters()]):
                            print('Found nan!! Not backprop through this instance!!')
                            optimizer.zero_grad()  # grad contains nan so not backprop through this instance
                        torch.nn.utils.clip_grad_norm_(
                            net.parameters(), 1, norm_type=2)
                        optimizer.step()

                    backward_time_list.append(time.time() - start_time)

                    tqdm_loader.set_postfix(
                        loss='{:.3f}'.format(np.mean(loss_list)),
                        strict_eval='{:.3f}'.format(
                            np.mean(strict_evaluation_list)),
                        soft_eval='{:.3f}'.format(
                            np.mean(soft_evaluation_list)),
                        cwpdis='{:.3f}'.format(np.mean(cwpdis_list)),
                        ess='{:.3f}'.format(np.mean(ess_list)),
                    )

                if evaluated:
                    print('Epoch {} with average {} loss: {}, strict evaluation: {}, soft eval: {}, likelihood time: {}, forward time: {}, evaluation time: {}, backward time: {}'.format(
                        epoch, mode, np.mean(loss_list), np.mean(
                            strict_evaluation_list), np.mean(soft_evaluation_list),
                        np.mean(likelihood_time_list), np.mean(
                            forward_time_list),
                        np.mean(evaluation_time_list), np.mean(
                            backward_time_list)
                    ))
                else:
                    print('Epoch {} with average {} loss: {}, likelihood time: {}, forward time: {}, evaluation time: {}, backward time: {}'.format(
                        epoch, mode, np.mean(loss_list),
                        np.mean(likelihood_time_list), np.mean(
                            forward_time_list),
                        np.mean(evaluation_time_list), np.mean(
                            backward_time_list)
                    ))

                if mode == 'train':
                    f_result.write('{}, {}, {}, {}, '.format(epoch, np.mean(loss_list), np.mean(
                        strict_evaluation_list), np.mean(soft_evaluation_list)))
                elif mode == 'validate':
                    f_result.write('{}, {}, {}, '.format(np.mean(loss_list), np.mean(
                        strict_evaluation_list), np.mean(soft_evaluation_list)))
                else:
                    f_result.write('{}, {}, {}\n'.format(np.mean(loss_list), np.mean(
                        strict_evaluation_list), np.mean(soft_evaluation_list)))

        f_result.close()
