import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import torch
import tqdm
import copy
import time
# from memory_profiler import profile

from model import MLP
from stable_baselines3 import DQN

MAX_STEPS_PER_EPISODE = 100 # Prevent infinitely long episodes
softmax = torch.nn.Softmax(dim=0)

# ============================= Policy Gradient Version =================================
def evaluate_phi_loglikelihood(env, model, num_episodes=50, discount=0.99, deterministic=False,  device='cpu', baseline=0):
    # print('Policy gradient version')
    total_reward_list, phi_list, total_log_likelihood_list, total_transition_log_likelihood_list = [], [], [], [] # torch.zeros(num_episodes), torch.zeros(num_episodes), torch.zeros(num_episodes)
    count = 0
    step_time = 0
    rewind_time = 0
    while count < num_episodes:
        episode_iter = 0
        start_time = time.time()
        env.reset()
        s1 = env.observe()
        rewind_time += time.time() - start_time
        state_list, reward_list, future_reward_list = [], [], []
        log_likelihood_list = []
        transition_log_likelihood_list = []
        while not env.is_terminal() and episode_iter < MAX_STEPS_PER_EPISODE:
            state_list.append(s1.view(1, -1))
            a, logp = model.policy.predict_with_logprob(s1.view(1, -1), deterministic=deterministic)  # logp is the probability of executing action a, i.e., prob(action | state)
            start_time = time.time()
            s1, r, _, info = env.step(a.detach())
            step_time += time.time() - start_time
            transition_logprob = info['logprob']
            reward_list.append(r)
            log_likelihood_list.append(logp)
            transition_log_likelihood_list.append(transition_logprob)
            episode_iter += 1
        G = 0
        total_log_likelihood_list.append(sum(log_likelihood_list))
        total_transition_log_likelihood_list.append(sum(transition_log_likelihood_list))
        q_values, probs = model.q_net_target(torch.cat(state_list, dim=0))
        # baselines = torch.sum(q_values * probs, dim=1)
        baselines = torch.sum(q_values * probs, dim=1).detach()
        phi_individual_list = []
        for i in range(len(reward_list) - 1, -1, -1):
            G = G + reward_list[i] * (discount ** i)
            future_reward_list.append(G)
            baseline = baselines[i]
            phi_individual_list.append(log_likelihood_list[i] * (G - baseline * discount ** i))
            # phi_individual_list.append(log_likelihood_list[i] * transition_log_likelihood_list[i] * (G - baseline * discount ** i))
        phi = sum(phi_individual_list)
        phi_list.append(phi)
        # total_reward_list.append(G)
        total_reward_list.append(sum(reward_list))
        count += 1
    # print('env time {}, rewind time {}'.format(step_time, rewind_time))
    return phi_list, total_log_likelihood_list, total_transition_log_likelihood_list, total_reward_list

# =========================== Bellman equation version ================================
# Bellman equation based approach
def compute_bellman_error(env, model, num_episodes=50, discount=0.99, deterministic=False, device='cpu'):
    # print('Bellman version')
    count = 0
    phi_list, total_log_likelihood_list, total_transition_log_likelihood_list, total_reward_list = [], [], [], []
    while count < num_episodes:
        episode_iter = 0
        log_likelihood_list, transition_log_likelihood_list = [], []
        state_list, action_list, next_state_list, reward_list = [], [], [], []

        env.reset()
        s1 = env.observe()
        while not env.is_terminal() and episode_iter < MAX_STEPS_PER_EPISODE:
            a, logp = model.policy.predict_with_logprob(s1.view(1, -1), deterministic=deterministic)  # logp is the probability of executing action a, i.e., prob(action | state)
            s2, r, _, info = env.step(a.detach().item())
            state_list.append(s1.view(1, -1))
            next_state_list.append(s2.view(1, -1))
            s1 = s2
            transition_logprob = info['logprob']
            log_likelihood_list.append(logp)
            transition_log_likelihood_list.append(transition_logprob)
            action_list.append(a.view(1, -1))
            reward_list.append(r)
            episode_iter += 1
        count += 1
        total_log_likelihood_list.append(sum(log_likelihood_list))
        total_transition_log_likelihood_list.append(sum(transition_log_likelihood_list))

        state_list = torch.cat(state_list, dim=0)
        action_list = torch.cat(action_list, dim=0)
        next_state_list = torch.cat(next_state_list, dim=0)
        reward_list = torch.tensor(reward_list).view(-1, 1)

        # current q values
        current_q_values, current_probs = model.q_net(state_list)
        current_q_values = torch.gather(current_q_values, dim=1, index=action_list.long())

        next_q_values, next_probs = model.q_net(next_state_list)
        next_q_values = torch.sum(next_q_values * next_probs, dim=1).reshape(-1, 1)
        target_q_values = reward_list + discount * next_q_values

        bellman_error_list = torch.abs(target_q_values - current_q_values)
        # bellman_error_list = (target_q_values - current_q_values)**2
        phi = torch.sum(bellman_error_list) ** 2

        # G = 0
        # phi_individual_list = []
        # for i in range(len(bellman_error_list)-1, -1, -1):
        #     G = G + bellman_error_list[i] * (discount ** i)
        #     phi_individual_list.append(log_likelihood_list[i] * G)
        # phi = sum(phi_individual_list)
        phi_list.append(phi)

    return phi_list, total_log_likelihood_list, total_transition_log_likelihood_list, total_reward_list

# ================= Estimated version of the policy hessian inverse vector product =================
# This version doesn't explicitly write down the Hessian itself.
# The specialty of not writing down the Hessian allows us to compute high dimensional Hessian without crashing the memory
def compute_policy_hessian_inverse_vector_product(x_raw, env_parameter, env, model, num_episodes=50, discount=0.99, baseline=0, verbose=0, backprop_method=1):
    start_time = time.time()
    if backprop_method in [0,1,2,3]:
        phi_list, total_log_likelihood_list, total_transition_log_likelihood_list, _ = evaluate_phi_loglikelihood(env, model, num_episodes=num_episodes, discount=discount, baseline=baseline)
        c = -1
    elif backprop_method in [4,5,6,7]:
        phi_list, total_log_likelihood_list, total_transition_log_likelihood_list, _ = compute_bellman_error(env, model, num_episodes=num_episodes, discount=discount)
        c = 1

    simulation_time = time.time() - start_time
    # print('simulation time', simulation_time)
    weights = 1 / num_episodes * torch.ones(num_episodes)
    U, V, W = [], [], []
    # U: phi
    # V: \nabla_\theta log_likelihood
    # W: \nabla_w log_likelihood
    # Hessian: UV 
    # \nalba_w theta J: UW

    if backprop_method in [1,2,3,6]:
        U, V = [], []
        for episode_id, (phi, total_log_likelihood, total_transition_log_likelihood, weight) in enumerate(zip(phi_list, total_log_likelihood_list, total_transition_log_likelihood_list, weights)):
            model.policy.optimizer.zero_grad()
            phi.backward(retain_graph=True, create_graph=False)
            tmp_gradient = torch.cat([param.grad.flatten().detach() for param in model.policy.q_net.parameters()]).view(1,-1)
            U.append(tmp_gradient)

            model.policy.optimizer.zero_grad()
            total_log_likelihood.backward(retain_graph=True, create_graph=False)
            tmp_log_likelihood_gradient = torch.cat([param.grad.flatten().detach() for param in model.policy.q_net.parameters()]).view(1,-1)
            V.append(tmp_log_likelihood_gradient)

            if env_parameter.grad is not None:
                env_parameter.grad.data.zero_() # manually clean gradient
                del env_parameter.grad

            total_transition_log_likelihood.backward(retain_graph=True, create_graph=False)
            tmp_log_likelihood_transition_gradient = env_parameter.grad.view(1,-1)
            W.append(tmp_log_likelihood_transition_gradient)
        U = torch.cat(U, dim=0).t() / np.sqrt(num_episodes)  # n by k matrix (n >> k)
        V = torch.cat(V, dim=0)     / np.sqrt(num_episodes)  # k by n matrix
        W = torch.cat(W, dim=0)     / np.sqrt(num_episodes)  # k by n matrix

    elif backprop_method in [0]:
        start_time = time.time()
        all_param = torch.cat([param.flatten() for param in model.policy.q_net.parameters()]).flatten()
        f = open('computation/{}_full_size{}.csv'.format(backprop_method, all_param.numel()), 'a')
        # print(all_param.numel())
        hessian_first_part  = torch.zeros(all_param.numel(), all_param.numel())
        hessian_second_part = torch.zeros(all_param.numel(), all_param.numel())

        for phi, total_log_likelihood, weight in zip(phi_list, total_log_likelihood_list, weights):
            model.policy.optimizer.zero_grad()
            phi.backward(retain_graph=True, create_graph=True)
            tmp_gradient = torch.cat([param.grad.flatten() for param in model.policy.q_net.parameters()]).view(1,-1)
            U.append(tmp_gradient)

        gradient = sum(U) / num_episodes
        gradient = gradient.flatten()

        for episode_id, (phi, total_log_likelihood, total_transition_log_likelihood, weight) in enumerate(zip(phi_list, total_log_likelihood_list, total_transition_log_likelihood_list, weights)):
            model.policy.optimizer.zero_grad()
            total_log_likelihood.backward(retain_graph=True, create_graph=False)
            tmp_log_likelihood_gradient = torch.cat([param.grad.flatten().detach() for param in model.policy.q_net.parameters()]).view(1,-1)
            V.append(tmp_log_likelihood_gradient)

            tmp_hessian_first_part = U[episode_id].view(-1,1) @ tmp_log_likelihood_gradient.view(1,-1)
            hessian_first_part += tmp_hessian_first_part

        for i, gradient_entry in enumerate(gradient):
            model.policy.optimizer.zero_grad()
            gradient_entry.backward(retain_graph=True, create_graph=False)
            tmp_gradient = torch.cat([param.grad.flatten().detach() for param in model.policy.q_net.parameters()]).flatten()
            hessian_second_part[i] = tmp_gradient

        hessian = hessian_first_part + hessian_second_part + torch.eye(all_param.numel())
        hessian_inv = torch.inverse(hessian)
        elapsed_time = time.time() - start_time
        # print('Hessian time', elapsed_time)
        f.write('{}, {} \n'.format(all_param.numel(), elapsed_time))
        f.close()

    elif backprop_method in [4]:
        start_time = time.time()
        all_param = torch.cat([param.flatten() for param in model.policy.q_net.parameters()]).flatten()
        f = open('computation/{}_full_size{}.csv'.format(backprop_method, all_param.numel()), 'a')
        # print(all_param.numel())
        hessian_first_part  = torch.zeros(all_param.numel(), all_param.numel())
        hessian_second_part = torch.zeros(all_param.numel(), all_param.numel())

        for phi, total_log_likelihood, weight in zip(phi_list, total_log_likelihood_list, weights):
            model.policy.optimizer.zero_grad()
            phi.backward(retain_graph=True, create_graph=True)
            tmp_gradient = torch.cat([param.grad.flatten() for param in model.policy.q_net.parameters()]).view(1,-1)
            U.append(tmp_gradient / torch.sqrt(phi.detach()))

        for episode_id, (phi, total_log_likelihood, total_transition_log_likelihood, weight) in enumerate(zip(phi_list, total_log_likelihood_list, total_transition_log_likelihood_list, weights)):
            model.policy.optimizer.zero_grad()
            total_log_likelihood.backward(retain_graph=True, create_graph=True)
            tmp_log_likelihood_gradient = torch.cat([param.grad.flatten() for param in model.policy.q_net.parameters()]).view(1,-1)
            V.append(tmp_log_likelihood_gradient)

            tmp_hessian_first_part = U[episode_id].view(-1,1) @ tmp_log_likelihood_gradient.view(1,-1) * torch.sqrt(phi.detach()) + U[episode_id].view(-1,1) @ U[episode_id].view(1,-1)
            hessian_first_part += tmp_hessian_first_part

        gradient = sum(U) / num_episodes
        gradient = gradient.flatten()

        likelihood_gradient = sum(V) / num_episodes
        likelihood_gradient = likelihood_gradient.flatten()

        for i, gradient_entry in enumerate(gradient):
            model.policy.optimizer.zero_grad()
            gradient_entry.backward(retain_graph=True, create_graph=False)
            tmp_gradient = torch.cat([param.grad.flatten().detach() for param in model.policy.q_net.parameters()]).flatten()
            hessian_second_part[i] += tmp_gradient * torch.sqrt(phi.detach())

        for i, gradient_entry in enumerate(likelihood_gradient):
            model.policy.optimizer.zero_grad()
            gradient_entry.backward(retain_graph=True, create_graph=False)
            tmp_gradient = torch.cat([param.grad.flatten().detach() for param in model.policy.q_net.parameters()]).flatten()
            hessian_second_part[i] += tmp_gradient * phi.detach()

        hessian = hessian_first_part + hessian_second_part + torch.eye(all_param.numel())
        hessian_inv = torch.inverse(hessian)
        elapsed_time = time.time() - start_time
        # print('Hessian time', elapsed_time)
        f.write('{}, {} \n'.format(all_param.numel(), elapsed_time))
        f.close()

    elif backprop_method in [5,7]:
        for episode_id, (phi, total_log_likelihood, total_transition_log_likelihood, weight) in enumerate(zip(phi_list, total_log_likelihood_list, total_transition_log_likelihood_list, weights)):
            model.policy.optimizer.zero_grad()
            phi.backward(retain_graph=True, create_graph=False)
            tmp_gradient = torch.cat([param.grad.flatten().detach() for param in model.policy.q_net.parameters()]).view(1,-1)
            U.append(tmp_gradient)
            V.append(tmp_gradient / phi.detach())

            if env_parameter.grad is not None:
                env_parameter.grad.data.zero_() # manually clean gradient
                del env_parameter.grad
            total_transition_log_likelihood.backward(retain_graph=True, create_graph=False)
            tmp_log_likelihood_transition_gradient = env_parameter.grad.view(1,-1)
            W.append(tmp_log_likelihood_transition_gradient)
        U = torch.cat(U, dim=0).t() / np.sqrt(num_episodes)  # n by k matrix (n >> k)
        V = torch.cat(V, dim=0)     / np.sqrt(num_episodes)  # k by n matrix
        W = torch.cat(W, dim=0)     / np.sqrt(num_episodes)  # k by n matrix

    # print('U', U, torch.norm(U))
    # print('V', V, torch.norm(V))
    # print('W', W, torch.norm(W))
    # We need to compute:
    # x^T (UV + another Hessian)^-1 (UW)
    x = x_raw[:x_raw.numel()//2] # remove the target Q net part

    if backprop_method in [0,4]:
        raise NotImplementedError('Full Hessian Method is not supported here.')
    elif backprop_method in [1,5]:
        # ========================= Woodbury matrix implementation =============================
        # Woodbury matrix identity (https://en.wikipedia.org/wiki/Woodbury_matrix_identity)
        # H = c * I + U V
        # H^-1 = 1/c * I - 1/c * U (c I + V U)^-1 V
    
        xU = x.view(1,-1) @ U
        xU = xU.view(1, -1, 1)
        VU = V @ U
        # VU = (VU + VU.t()) / 2 # although this might not be symmetric, theoretically the hessian should be symmetric. The reason this expression is not symmetric is because it is a sampled Hessian only.
        VU_t = VU.t()
        # print('w norm: {}'.format(torch.norm(w)))
        # print(torch.eig(VU)[0])
    
        B = torch.eye(U.shape[1]) * c + VU_t   # B = c I + VU
        B = B.view(1, *B.shape)
        try:
            Binv_xU, _ = torch.solve(xU, B)
            Binv_xU = Binv_xU[0]
        except:
            B = torch.eye(U.shape[1]) * c
            Binv_xU = xU / c
            print('Found singular matrix B. Use an identity matrix * c as B instead.')
    
        # print(x.shape, Binv_xU.shape, V.shape)
        y = (x.view(1,-1) - Binv_xU.view(1,-1) @ V) / c
        # print(y.shape, U.shape, W.shape)
        z = (y @ U) @ W
        z = z.flatten()
        result = z # torch.cat([z, torch.zeros_like(z)])

    elif backprop_method in [2,6]:
        # ============================= ignoring the second term ==================================
        # H = UV
        # compute: x^t (UV)^-1 (UW)
        result = (((x.flatten() @ torch.pinverse(V)) @ torch.pinverse(U)) @ U) @ W
        result = result.flatten()
        # result = torch.cat([result, torch.zeros_like(result)])
        return result
        # raise NotImplementedError('Direct inversion is not supported here')
    elif backprop_method in [3,7]:
        # Assuming the Hessian is cI
        # so we just need to compute x^t (cI)^-1 (UW)
        z = (x.view(1,-1) @ U) @ W / c
        z = z.flatten()
        result = z # torch.cat([z, torch.zeros_like(z)])

    # print(torch.norm(result))
    return result.view(env_parameter.shape)

# def compute_bellman_hessian_inverse_vector_product(x_raw, env_parameter, env, model, num_episodes=50, discount=0.99, reg_const=100, baseline=0, verbose=0, backprop_method=4):
#     phi_list, total_log_likelihood_list = compute_bellman_error(env, model, num_episodes=num_episodes, discount=discount)
# 
#     weights = 1 / num_episodes * torch.ones(num_episodes)
#     U, V = [], []
#     for episode_id, (phi, total_log_likelihood, weight) in enumerate(zip(phi_list, total_log_likelihood_list, weights)):
#         model.policy.optimizer.zero_grad()
#         phi.backward(retain_graph=True, create_graph=False)
#         tmp_gradient = torch.cat([param.grad.flatten().detach() for param in model.policy.q_net.parameters()]).view(1,-1)
#         U.append(tmp_gradient)
# 
#         model.policy.optimizer.zero_grad()
#         del env_parameter.grad # manually clean gradient
# 
#         total_log_likelihood.backward(retain_graph=True, create_graph=False)
#         tmp_log_likelihood_gradient = torch.cat([param.grad.flatten().detach() for param in model.policy.q_net.parameters()]).view(1,-1)
#         tmp_log_likelihood_transition_gradient = env_parameter.grad.view(1,-1)
#         V.append(tmp_log_likelihood_gradient)
#         W.append(tmp_log_likelihood_transition_gradient)
#     U = torch.cat(U, dim=0).t() / np.sqrt(num_episodes)  # n by k matrix (n >> k)
#     V = torch.cat(V, dim=0)     / np.sqrt(num_episodes)  # k by n matrix
#     W = torch.cat(W, dim=0)     / np.sqrt(num_episodes)  # k by n matrix
# 
#     if backprop_method == 4: # Woodbury
#         U = []
#         for i in range(len(bellman_error_list)):
#             model.policy.optimizer.zero_grad()
#             bellman_error_list[i].backward(retain_graph=True, create_graph=False)
#             tmp_grad = torch.cat([param.grad.flatten() for param in model.policy.q_net.parameters()])
#             U.append(tmp_grad.view(1,-1))
# 
#         # H = U^T U + cI
#         # H^-1 = 1/c * I - 1/c * U^T (c I + U U^T)^-1 U
#         # compute H^-1 w
#         c = reg_const
#         U = torch.cat(U, dim=0).detach() / np.sqrt(len(U)) # k by n matrix
#         UUT = U @ U.t()
#         x = U @ w.view(-1,1)
#         x = x.view(1,-1,1)
#         B = c * torch.eye(len(U)) + UUT
#         B = B.view(1, *B.shape)
#         Binv_x, _ = torch.solve(x, B)
#         Binv_x = Binv_x[0]
#         result = (w - U.t() @ Binv_x) / c
#         result = result.flatten()
#         result = torch.cat([result, torch.zeros_like(result)])
#     # elif backprop_method == 5:
#     #     pass
#     elif backprop_method == 6:
#         result = w
#         result = torch.cat([result, torch.zeros_like(result)])
# 
#     return result

# =========================== Off-policy Evaluation ===================================
def CWPDIS(env, policy_kwargs, trajectories, discount=0.95, device='cpu'): # take model as input, not the model parameters
    class CWPDISFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, model_parameters):
            with torch.enable_grad():
                model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=0, gamma=discount, buffer_size=0, device=device)
                model.policy.load_from_vector(model_parameters)
                model.policy.train()

                N = len(trajectories)    # number of trajectories
                T = len(trajectories[0]) # fixed time horizon
                weights = torch.ones((N,T)) # rho_nt in CWPDIS
                states = torch.cat([trajectories[n][t][0].view(1,-1) for n in range(N) for t in range(T)], dim=0)
                actions = torch.cat([trajectories[n][t][1].view(1,-1) for n in range(N) for t in range(T)], dim=0).long()
                rewards = torch.tensor([trajectories[n][t][2] for n in range(N) for t in range(T)]).view(N,T)
                next_states = torch.cat([trajectories[n][t][3].view(1,-1) for n in range(N) for t in range(T)], dim=0)
                demonstrated_probs = torch.tensor([trajectories[n][t][4] for n in range(N) for t in range(T)]).view(N,T)

                q_values, probs = model.policy.q_net(states)
                selected_probs = torch.gather(probs, dim=1, index=actions).view(N,T)
                prob_ratios = selected_probs / demonstrated_probs

                for t in range(T):
                    weights[:,t:] *= prob_ratios[:,t].view(-1,1)
                # weights += 1e-6  # avoid zero weight

                # for n, trajectory in enumerate(trajectories):
                #     total_prob = torch.tensor(1)
                #     for t, (s, a, r, s2, prob) in enumerate(trajectory):
                #         q_values, probs = model.policy.q_net(s.view(1,-1))
                #         total_prob = total_prob * probs[0,a] / prob  # cumulative prob
                #         weights[n,t] += total_prob                   # get and copy the value only
                #         rewards[n,t] = r

                discounts = discount ** torch.arange(T)
                total_weights = torch.sum(weights, dim=0)

                # total_weights += 0.001 # avoid zero weight
                # total_weights[total_weights == 0] = 1 # set zero weight to 1 to avoid nan or inf issue
                cwpdis_list = discounts * torch.sum(weights * rewards, dim=0) / total_weights
                cwpdis = torch.sum(cwpdis_list) # CWPDIS
                total_square_weights = torch.sum(weights**2, dim=0)
                # total_square_weights += 0.001
                total_square_weights[total_square_weights == 0] = 1
                ess = torch.sum(torch.sum(weights, dim=0)**2 / total_square_weights)

                model.policy.optimizer.zero_grad()
                cwpdis.backward(retain_graph=True)
                cwpdis_gradient = [param.grad.flatten() for param in model.policy.q_net.parameters()] # consider the gradient of q_net only
                cwpdis_gradient = torch.cat(cwpdis_gradient).detach()
                cwpdis_gradient = torch.cat([cwpdis_gradient, torch.zeros_like(cwpdis_gradient)])

                model.policy.optimizer.zero_grad()
                ess.backward()
                ess_gradient = [param.grad.flatten() for param in model.policy.q_net.parameters()] # consider the gradient of q_net only
                ess_gradient = torch.cat(ess_gradient).detach()
                ess_gradient = torch.cat([ess_gradient, torch.zeros_like(ess_gradient)])

                # print('OPE gradient sum', torch.sum(cwpdis_gradient), torch.sum(ess_gradient))

                cwpdis_gradient[cwpdis_gradient.isnan()] = 0
                ess_gradient[ess_gradient.isnan()] = 0
                model.policy.optimizer.zero_grad()

            ctx.save_for_backward(cwpdis_gradient, ess_gradient)
            for param in model.policy.parameters(): # manually clearning the model gradient to prevent memory leak
                del param.grad

            return cwpdis.detach(), ess.detach()

        def backward(ctx, dl_dcwpdis, dl_dess):
            cwpdis_gradient, ess_gradient = ctx.saved_tensors
            return dl_dcwpdis * cwpdis_gradient + dl_dess * ess_gradient

    return CWPDISFn.apply

# per decision importance sampling
def PDIS(env, policy_kwargs, trajectories, discount=0.95, device='cpu'): # take model as input, not the model parameters
    class CWPDISFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, model_parameters):
            with torch.enable_grad():
                model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=0, gamma=discount, buffer_size=0, device=device)
                model.policy.load_from_vector(model_parameters)
                model.policy.train()

                N = len(trajectories)    # number of trajectories
                T = len(trajectories[0]) # fixed time horizon
                weights = torch.ones((N,T)) # rho_nt in CWPDIS
                states = torch.cat([trajectories[n][t][0].view(1,-1) for n in range(N) for t in range(T)], dim=0)
                actions = torch.cat([trajectories[n][t][1].view(1,-1) for n in range(N) for t in range(T)], dim=0).long()
                rewards = torch.tensor([trajectories[n][t][2] for n in range(N) for t in range(T)]).view(N,T)
                next_states = torch.cat([trajectories[n][t][3].view(1,-1) for n in range(N) for t in range(T)], dim=0)
                demonstrated_probs = torch.tensor([trajectories[n][t][4] for n in range(N) for t in range(T)]).view(N,T)

                q_values, probs = model.policy.q_net(states)
                selected_probs = torch.gather(probs, dim=1, index=actions).view(N,T)
                prob_ratios = selected_probs / demonstrated_probs

                for t in range(T):
                    weights[:,t:] *= prob_ratios[:,t].view(-1,1)

                discounts = discount ** torch.arange(T)
                pdis_list = discounts * torch.mean(weights * rewards, dim=0)
                pdis = torch.sum(pdis_list) # CWPDIS

                total_square_weights = torch.sum(weights**2, dim=0)
                # total_square_weights += 0.001
                total_square_weights[total_square_weights == 0] = 1
                ess = torch.sum(torch.sum(weights, dim=0)**2 / total_square_weights)

                model.policy.optimizer.zero_grad()
                pdis.backward(retain_graph=True)
                pdis_gradient = [param.grad.flatten() for param in model.policy.q_net.parameters()] # consider the gradient of q_net only
                pdis_gradient = torch.cat(pdis_gradient).detach()
                pdis_gradient = torch.cat([pdis_gradient, torch.zeros_like(pdis_gradient)])

                model.policy.optimizer.zero_grad()
                ess.backward()
                ess_gradient = [param.grad.flatten() for param in model.policy.q_net.parameters()] # consider the gradient of q_net only
                ess_gradient = torch.cat(ess_gradient).detach()
                ess_gradient = torch.cat([ess_gradient, torch.zeros_like(ess_gradient)])

                pdis_gradient[pdis_gradient.isnan()] = 0
                ess_gradient[ess_gradient.isnan()] = 0

                model.policy.optimizer.zero_grad()

            ctx.save_for_backward(pdis_gradient, ess_gradient)
            for param in model.policy.parameters(): # manually clearning the model gradient to prevent memory leak
                del param.grad

            return pdis.detach(), ess.detach()

        def backward(ctx, dl_dcwpdis, dl_dess):
            pdis_gradient, ess_gradient = ctx.saved_tensors
            return dl_dcwpdis * pdis_gradient + dl_dess * ess_gradient

    return CWPDISFn.apply

# Online evaluation
def PerformanceDQN(env, policy_kwargs, num_episodes=50, discount=0.99, device='cpu'):
    class PerformanceDQNFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, model_parameters):
            with torch.enable_grad():
                model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=0, gamma=discount, buffer_size=0, device=device)
                model.policy.load_from_vector(model_parameters)

                model.policy.train()
                model.policy.optimizer.zero_grad()
                phi_list, total_log_likelihood_list, _, total_reward_list = evaluate_phi_loglikelihood(env, model, num_episodes=num_episodes, discount=discount, deterministic=False)
                weights = 1 / num_episodes * torch.ones(num_episodes)
                phi = sum(phi_list) / num_episodes
                phi.backward(retain_graph=True)
                gradient = [param.grad.flatten() for param in model.policy.q_net.parameters()] # consider both the gradient of q_net and q_net_target
                gradient = torch.cat(gradient).detach()
                # gradient = torch.cat([param.grad.flatten() for param in model.policy.parameters()]) # consider both the gradient of q_net and q_net_target
                gradient_full = torch.cat([gradient, torch.zeros_like(gradient)])
                model.policy.optimizer.zero_grad()

            ctx.save_for_backward(gradient_full)
            total_reward = sum(total_reward_list) / num_episodes
            for param in model.policy.parameters():
                del param.grad

            return torch.tensor(total_reward)

        @staticmethod
        def backward(ctx, dl_dJ):
            gradient_full, = ctx.saved_tensors
            # with torch.enable_grad():
            #     weights = 1 / num_episodes * torch.ones(num_episodes)
            #     phi = sum(phi_list * weights)
            #     gradient = torch.autograd.grad(phi, Q_var)[0]
            return dl_dJ * gradient_full

    return PerformanceDQNFn.apply

# ===================================== implementation of the differentiable DDQN ==============================
def DiffDQN(env_wrapper, policy_kwargs, learning_rate=0.001, learning_starts=20000, min_num_iters=100000, model_initial_parameters=None, discount=0.99, target_update_interval=1000, buffer_size=100000, device='cpu', verbose=0, load_replay_buffer=False, save_replay_buffer=False, replay_buffer_dict=None, data_id=None, baseline=0, backprop_method=1, seed=0):
    class DiffDQNFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, env_parameter): # reward
            env = env_wrapper(env_parameter.detach())

            # DQN # TODO: choosing the right parameters for DQN setup
            model = DQN("MlpPolicy", env, learning_rate=learning_rate, learning_starts=learning_starts, policy_kwargs=policy_kwargs, verbose=verbose, gamma=discount, strict=True,
                    target_update_interval=target_update_interval,
                    buffer_size=buffer_size,
                    optimize_memory_usage=False,
                    seed=seed,
                    )

            if model_initial_parameters is not None:
                model.policy.load_from_vector(model_initial_parameters)

            if load_replay_buffer:
                try:
                    model.replay_buffer = replay_buffer_dict[data_id]
                except:
                    print('Failed to load the replay buffer...')

            with torch.enable_grad():
                model.learn(total_timesteps=min_num_iters, log_interval=50)

            model_parameters = torch.from_numpy(model.policy.parameters_to_vector()) # pytorch autograd doesn't support non-tensor output so we have to create one :(
            if save_replay_buffer:
                replay_buffer_dict[data_id] = model.replay_buffer

            ctx.save_for_backward(env_parameter.detach(), model_parameters.detach())
            return model_parameters

        @staticmethod
        def backward(ctx, dl_dmodel):
            env_parameter, model_parameters = ctx.saved_tensors
            env = env_wrapper(env_parameter.detach())

            model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=0, gamma=discount, device=device, buffer_size=0)
            model.policy.load_from_vector(model_parameters.detach()) # loading model parameters

            with torch.enable_grad():
                env_parameter_var = torch.autograd.Variable(env_parameter, requires_grad=True)
                env_var = env_wrapper(env_parameter_var)

                # policy_gradient = compute_policy_gradient(env, model, num_episodes=50, discount=discount)

                # --------------- version 1: full policy hessian --------------------
                # This is not feasible because of the dimensionality

                # --------------- version 2: Partial policy hessian -----------------
                # This is still no feasible because of the dimensionality of the Hessian matrix.
                # Simply the size of the Hessian matrix exceeds the memory limit
                # policy_hessian = compute_policy_hessian(env, model, num_episodes=5, discount=discount).detach()
                # x, _ = torch.solve(policy_gradient.view(-1,1), policy_hessian)
                # surrogate = - dl_dmodel @ x.flatten()

                # --- version 2.5: partial policy hessian without explicity expansion
                # This is an improved version of version 2 but without explicitly writing down the entire Hessian
                # if backprop_method in [4,5,6]: # Bellman equation based
                #     if backprop_method == 6:
                #         dl_denv = compute_bellman_hessian_inverse_vector_product(-dl_dmodel, env_parameter_var, env_var, model, num_episodes=100, discount=discount, baseline=baseline, backprop_method=backprop_method)
                #     else:
                #         dl_denv = compute_bellman_hessian_inverse_vector_product(-dl_dmodel, env_parameter_var, env_var, model, num_episodes=100, discount=discount, baseline=baseline, backprop_method=backprop_method)
                # else: # policy gradient based
                dl_denv = compute_policy_hessian_inverse_vector_product(-dl_dmodel, env_parameter_var, env_var, model, num_episodes=100, discount=discount, baseline=baseline, backprop_method=backprop_method)

                # print('OPE gradient norm:', torch.norm(dl_dmodel))

                # --------------- version 3: scalar estimate of policy hessian ------
                # policy_hessian_estimate = compute_policy_hessian_estimate(env, model, num_episodes=5, discount=discount).detach()
                # surrogate = - dl_dmodel.flatten() @ policy_gradient.flatten() / policy_hessian_estimate

                # --------------- version 4: using -I to approximate policy hessian -
                # surrogate = dl_dmodel.flatten() @ policy_gradient.flatten() # / (torch.norm(policy_gradient.detach().flatten())**2)


                # compute the gradient wrt environment parameter
                # dl_denv = torch.autograd.grad(surrogate, env_parameter_var, allow_unused=False)[0].detach()
                # if dl_denv is None:
                #     dl_denv = torch.zeros_like(env_parameter_var)
                # else:
                #     dl_denv = dl_denv.detach()

            # clearning model parameter gradient
            for param in model.policy.parameters():
                del param.grad
            del env_parameter_var.grad

            return dl_denv
    return DiffDQNFn.apply
