import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import torch
import tqdm
import copy

from gridworld import *
from model import MLP

from utils import softmax_action, run_simulation, policy_from_Q

# pytorch anomaly detection
# torch.autograd.set_detect_anomaly(True)

MAX_STEPS_PER_EPISODE = 100 # Prevent infinitely long episodes
softmax = torch.nn.Softmax(dim=0)
softmax_d1 = torch.nn.Softmax(dim=1)

def evaluate_phi_loglikelihood(env, Q, num_episodes=50, discount=0.95, lamb=5):
    """Evaluate a policy as specified by `Q` without performing any additional
    training, for some `num_episodes` at some action stochasticity"""
    total_reward_list, phi_list, total_log_likelihood_list = [], [], []
    count = 0
    while count < num_episodes:
        episode_iter = 0
        env.reset()
        s1 = env.observe()
        state_list, reward_list, future_reward_list = [], [], []
        log_likelihood_list = []
        while not env.is_terminal(s1) and episode_iter < MAX_STEPS_PER_EPISODE:
            state_list.append(s1)
            a, prob = softmax_action(s1, Q, lamb)
            s1, r, _, _ = env.step(a)
            reward_list.append(r)
            log_likelihood_list.append(torch.log(prob))
            episode_iter += 1
        G = 0
        total_log_likelihood_list.append(sum(log_likelihood_list))
        phi_individual_list = []
        for i in range(len(reward_list)-1, -1, -1):
            G = G + reward_list[i] * (discount ** i)
            future_reward_list.append(G)
            q_values = Q[state_list[i]].detach()
            probs = (q_values * lamb).softmax(dim=0)
            baseline = torch.sum(q_values * probs)

            # phi_individual_list.append(log_likelihood_list[i] * G) # no baseline
            phi_individual_list.append(log_likelihood_list[i] * (G - baseline * discount ** i)) # using baseline to reduce variance
        phi = sum(phi_individual_list)
        phi_list.append(phi)
        total_reward_list.append(G)
        count += 1
    return phi_list, total_log_likelihood_list, total_reward_list

# =========================== Bellman equation version ================================
# Bellman equation based approach
def compute_bellman_error(env, Q, num_episodes=50, discount=0.99, lamb=5):
    count = 0
    phi_list, total_log_likelihood_list, total_reward_list = [], [], []
    while count < num_episodes:
        episode_iter = 0
        log_likelihood_list, bellman_error_list = [], []
        state_list, action_list, next_state_list, reward_list = [], [], [], []

        env.reset()
        s1 = env.observe()
        while not env.is_terminal(s1) and episode_iter < MAX_STEPS_PER_EPISODE:
            a, prob = softmax_action(s1, Q, lamb)
            s2, r, _, info = env.step(a)

            state_list.append(torch.Tensor(s1).view(1,-1))
            next_state_list.append(torch.Tensor(s2).view(1,-1))
            log_likelihood_list.append(torch.log(prob))
            action_list.append(a)
            reward_list.append(r)

            current_q_value = Q[s1[0], s1[1], a]
            next_q_value = torch.sum(Q[s2[0], s2[1]] * (Q[s2[0], s2[1]] * lamb).softmax(dim=0))
            target_q_value = r + discount * next_q_value
            bellman_error_list.append(torch.abs(current_q_value - target_q_value))

            s1 = s2
            episode_iter += 1
        count += 1
        total_log_likelihood_list.append(sum(log_likelihood_list))

        phi = sum(bellman_error_list)
        # G = 0
        # phi_individual_list = []
        # for i in range(len(bellman_error_list)-1, -1, -1):
        #     G = G + bellman_error_list[i] * (discount ** i)
        #     phi_individual_list.append(log_likelihood_list[i] * G)
        # phi = sum(phi_individual_list)

        phi_list.append(phi**2)

    return phi_list, total_log_likelihood_list, total_reward_list

def compute_policy_hessian_gradient(env, Q, num_episodes=5, discount=0.95, lamb=5, device='cpu', backprop_method=0):
    # Full Hessian version
    Q_var = torch.autograd.Variable(Q, requires_grad=True).to(device)
    if backprop_method == 0:
        phi_list, total_log_likelihood_list, _ = evaluate_phi_loglikelihood(env, Q_var, num_episodes=num_episodes, discount=discount, lamb=lamb)
        c = -1
    elif backprop_method == 4:
        phi_list, total_log_likelihood_list, _ = compute_bellman_error(env, Q_var, num_episodes=num_episodes, discount=discount, lamb=lamb)
        c = 1

    weights = 1 / num_episodes * torch.ones(num_episodes)
    # weights = softmax(total_log_likelihood_list).detach()
    weighted_gradient_list = []
    w_list = []
    for phi, total_log_likelihood, weight in zip(phi_list, total_log_likelihood_list, weights):
        if backprop_method == 0:
            tmp_gradient = torch.autograd.grad(phi, Q_var, retain_graph=True, create_graph=True)[0].flatten()
            weighted_gradient_list.append(tmp_gradient * weight)
            w_list.append(tmp_gradient * weight)
        elif backprop_method == 4:
            tmp_gradient = torch.autograd.grad(phi**2, Q_var, retain_graph=True, create_graph=True)[0].flatten()
            weighted_gradient_list.append(tmp_gradient * weight)
            tmp_gradient = torch.autograd.grad(phi, Q_var, retain_graph=True, create_graph=False)[0].flatten()
            w_list.append(tmp_gradient.detach() * weight * phi)

    gradient = sum(weighted_gradient_list)
    w = sum(w_list)
    # nonzero_indices = torch.nonzero(gradient).flatten()
    # print(len(gradient))
    nonzero_indices = torch.range(0,len(gradient)-1).flatten().long()
    nonzero_gradient = gradient[nonzero_indices]
    nonzero_w = w[nonzero_indices]

    hessian_first_part  = torch.zeros(len(nonzero_indices), len(nonzero_indices), device=device)
    hessian_second_part = torch.zeros(len(nonzero_indices), len(nonzero_indices), device=device)

    if backprop_method == 0:
        for i, (phi, total_log_likelihood, weight) in enumerate(zip(phi_list, total_log_likelihood_list, weights)):
            tmp_log_likelihood_gradient = torch.autograd.grad(total_log_likelihood, Q_var, retain_graph=True, create_graph=False)[0].flatten()[nonzero_indices]

            tmp_hessian_first_part = weighted_gradient_list[i][nonzero_indices].view(-1,1) @ tmp_log_likelihood_gradient.view(1,-1)
            hessian_first_part += tmp_hessian_first_part

    for i, gradient_entry in enumerate(nonzero_gradient):
        hessian_second_part[i] += torch.autograd.grad(gradient_entry, Q_var, retain_graph=True, create_graph=False)[0].flatten()[nonzero_indices]

    hessian = hessian_first_part + hessian_second_part
    hessian = hessian.detach()

    # print('Hessian first part eigenvalues:', torch.eig(hessian_first_part))
    # print('Hessian second part eigenvalues:', torch.eig(hessian_second_part))
    # print('Hessin eigenvalues:', torch.eig(hessian))

    # subtracting regularization term to make it negative semi definite
    # print('max eigenvalue:', max([real_part.item() for (real_part, imag_part) in eigenvalues]))
    # hessian -= (max([real_part.item() for (real_part, imag_part) in eigenvalues]) + c) * torch.eye(nonzero_indices.numel())
    hessian += c * torch.eye(nonzero_indices.numel())

    # print('gradient:', gradient)
    # print('hessian:', hessian)
    x, _ = torch.solve(nonzero_w.view(-1,1), hessian) # solve Ax=b

    return x, nonzero_indices

def compute_policy_hessian_inverse_vector_product(dl_dQ, env_parameter_var, env, Q, num_episodes=5, discount=0.95, lamb=5, device='cpu', backprop_method=1):
    Q_var = torch.autograd.Variable(Q, requires_grad=True).to(device)
    if backprop_method in [0,1,2,3]:
        phi_list, total_log_likelihood_list, _ = evaluate_phi_loglikelihood(env, Q_var, num_episodes=num_episodes, discount=discount, lamb=lamb)
        c = -1
    elif backprop_method in [4,5,6,7]:
        phi_list, total_log_likelihood_list, _ = compute_bellman_error(env, Q_var, num_episodes=num_episodes, discount=discount)
        c = 1

    weights = 1 / num_episodes * torch.ones(num_episodes)
    U, V = [], []
    total_phi = sum(phi_list) / num_episodes
    w = torch.autograd.grad(total_phi, Q_var, retain_graph=True, create_graph=True)[0].flatten()
    # else:
    #     w = torch.autograd.grad(total_phi, Q_var, retain_graph=True, create_graph=True)[0].flatten()

    if backprop_method in [1]:
        for phi, total_log_likelihood, weight in zip(phi_list, total_log_likelihood_list, weights):
            tmp_gradient = torch.autograd.grad(phi, Q_var, retain_graph=True, create_graph=False)[0].flatten().view(1,-1)
            U.append(tmp_gradient)

            tmp_likelihood_gradient = torch.autograd.grad(total_log_likelihood, Q_var, retain_graph=True, create_graph=False)[0].flatten().view(1,-1)
            V.append(tmp_likelihood_gradient)

        U = torch.cat(U, dim=0).t() / np.sqrt(num_episodes)  # n by k matrix (n >> k)
        V = torch.cat(V, dim=0)     / np.sqrt(num_episodes)  # k by n matrix

    elif backprop_method in [5,6,7]:
        w_p_list = []
        for phi, total_log_likelihood, weight in zip(phi_list, total_log_likelihood_list, weights):
            tmp_gradient = torch.autograd.grad(torch.sqrt(phi), Q_var, retain_graph=True, create_graph=False)[0].flatten().view(1,-1)
            tmp_likelihood_gradient = torch.autograd.grad(total_log_likelihood, Q_var, retain_graph=True, create_graph=False)[0].flatten().view(1,-1)
            U.append(tmp_gradient)
            V.append(tmp_gradient)
            # w_p_list.append(tmp_gradient.detach().flatten() * total_log_likelihood * torch.sqrt(phi).detach() + tmp_likelihood_gradient.detach().flatten() * torch.sqrt(phi) * torch.sqrt(phi).detach())
            w_p_list.append(tmp_gradient.detach().flatten() * torch.sqrt(phi))

        # w += 2 * sum(w_p_list) / num_episodes
        w = sum(w_p_list) / num_episodes
        U = torch.cat(U, dim=0).t() / np.sqrt(num_episodes)  # n by k matrix (n >> k)
        V = torch.cat(V, dim=0)     / np.sqrt(num_episodes)  # k by n matrix


    if backprop_method in [1,5]:
        # ========================= Woodbury matrix implementation =============================
        # Woodbury matrix identity (https://en.wikipedia.org/wiki/Woodbury_matrix_identity)
        # H = c * I + U V
        # H^-1 = 1/c * I - 1/c * U (c I + V U)^-1 V

        x = V @ w
        x = x.view(1, -1, 1)
        VU = V @ U
        # VU = (VU + VU.t()) / 2 # although this might not be symmetric, theoretically the hessian should be symmetric. The reason this expression is not symmetric is because it is a sampled Hessian only.
        # print('w norm: {}'.format(torch.norm(w)))
        # print('eigenvalues:', torch.eig(VU))

        B = torch.eye(U.shape[1]) * c + VU   # c I + VU
        B = B.view(1, *B.shape)
        try:
            Binv_x, _ = torch.solve(x, B)
            Binv_x = Binv_x[0]
        except:
            B = torch.eye(U.shape[1]) * c
            Binv_x = x / c
            print('Found singular matrix B. Use an identity matrix * c as B instead.')

        y = (w - U @ Binv_x.flatten()) / c
        surrogate = y.flatten() @ dl_dQ
        dl_denv = torch.autograd.grad(surrogate, env_parameter_var)[0]

    # elif backprop_method in [2,6]:
    #     # raise NotImplementedError('Direct inverse method is not implemented')
    #     # # ============================= ignoring the second term ==================================
    #     result = torch.pinverse(V) @ (torch.pinverse(U) @ w)
    #     result = result.flatten()
    elif backprop_method in [3,7]:
        y = w / c
        surrogate = y.flatten() @ dl_dQ
        dl_denv = torch.autograd.grad(surrogate, env_parameter_var)[0]

    # print(torch.norm(dl_dQ), torch.norm(dl_denv))
    return dl_denv

def plot_policy(env, Q, filename='default', evaluation=0):
    """Visualize a policy for debugging."""
    row_count, col_count = env.maze_dimensions
    maze_dims = (row_count, col_count)
    value_function = np.reshape(np.max(Q.detach().to('cpu').numpy().reshape(-1, env.num_actions), 1), maze_dims)
    policy_function = np.reshape(np.argmax(Q.detach().to('cpu').numpy().reshape(-1, env.num_actions), 1), maze_dims)
    wall_info = .5 + np.zeros(maze_dims)
    wall_mask = np.zeros(maze_dims)
    for row in range(row_count):
        for col in range(col_count):
            if env.maze.topology[row][col] == '#':
                wall_mask[row,col] = 1
    wall_info = np.ma.masked_where(wall_mask==0, wall_info)
    value_function *= (1-wall_mask)**2
    plt.imshow(value_function, interpolation='none', cmap='jet')
    plt.colorbar(label='Value Function')
    plt.imshow(wall_info, interpolation='none' , cmap='gray')
    for y,x in env.maze.start_coords:
        plt.text(x,y,'start', color='gray', fontsize=14, va='center', ha='center', fontweight='bold')
    for y,x in env.maze.goal_coords:
        plt.text(x,y,'goal', color='yellow', fontsize=14, va='center', ha='center', fontweight='bold')
    for row in range( row_count ):
        for col in range( col_count ):
            if wall_mask[row][col] == 1:
                continue
            if policy_function[row,col] == 0:
                dx = 0; dy = -.5
            if policy_function[row,col] == 1:
                dx = 0; dy = .5
            if policy_function[row,col] == 2:
                dx = .5; dy = 0
            if policy_function[row,col] == 3:
                dx = -.5; dy = 0
            if policy_function[row,col] == 4:
                dx = 0; dy = 0
            plt.arrow(col, row, dx, dy,
                shape='full', fc='w' , ec='w' , lw=3, length_includes_head=True, head_width=.2)
    plt.xlabel("X-Coordinate")
    plt.ylabel("Y-Coordinate")
    plt.title('Reward: {}'.format(evaluation))
    plt.savefig('{}.png'.format(filename))
    plt.close()

def CWPDIS(trajectories, Q, discount=0.95, lamb=5, ess_const=4):
    N = len(trajectories)    # number of trajectories
    T = len(trajectories[0]) # fixed time horizon
    weights = torch.zeros((N,T)) # rho_nt in CWPDIS
    rewards = torch.zeros((N,T))
    for n, trajectory in enumerate(trajectories):
        total_prob = torch.tensor(1)
        for t, (s, a, r, s2, prob) in enumerate(trajectory):
            prob_e = (Q[s] * lamb).softmax(dim=0)[a] # + 0.001
            total_prob = total_prob * prob_e / prob  # cumulative prob
            weights[n,t] += total_prob               # get and copy the value only
            rewards[n,t] = r

    discounts = discount ** torch.arange(T)
    cwpdis = torch.sum(discounts * torch.sum(weights * rewards, dim=0) / (torch.sum(weights, dim=0) + 0.001)) # CWPDIS
    ess = torch.sum((torch.sum(weights, dim=0) ** 2) / (torch.sum(weights**2, dim=0) + 0.001))
    return cwpdis, ess

def PerformanceQ(env, num_episodes=500, discount=0.95, lamb=5, device='cpu'):
    class PerformanceQFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, Q):
            with torch.enable_grad():
                Q_var = torch.autograd.Variable(Q, requires_grad=True).to(device)
                phi_list, total_log_likelihood_list, total_reward_list = evaluate_phi_loglikelihood(env, Q_var, num_episodes=num_episodes, discount=discount, lamb=lamb)
                weights = 1 / num_episodes * torch.ones(num_episodes)
                phi = sum(phi_list) / num_episodes
                gradient = torch.autograd.grad(phi, Q_var)[0]

            ctx.save_for_backward(gradient, Q_var, phi_list, total_log_likelihood_list)
            total_reward = sum(total_reward_list) / num_episodes
            return total_reward

        @staticmethod
        def backward(ctx, dl_dJ):
            gradient, Q_var, phi_list, total_log_likelihood_list = ctx.saved_tensors
            # with torch.enable_grad():
            #     weights = 1 / num_episodes * torch.ones(num_episodes)
            #     phi = sum(phi_list * weights)
            #     gradient = torch.autograd.grad(phi, Q_var)[0]
            return dl_dJ * gradient

    return PerformanceQFn.apply

def DiffQ(env_wrapper, min_num_episodes=1000, min_num_iters=10000, Q_initial=0, discount=0.95, lamb=5, device='cpu', backprop_method=0):
    class DiffQFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, env_parameter): # reward
            env = env_wrapper(env_parameter)

            # Q learning
            with torch.enable_grad():
                res = run_simulation(env, method='Q-learning', min_num_episodes=min_num_episodes, min_num_iters=min_num_iters, Q_initial=Q_initial, discount=discount, lamb=lamb, device=device)
                Q, num_cliff_falls, episode_rewards = res['Q'], res['num_cliff_falls'], res['episode_rewards']

            ctx.save_for_backward(Q, env_parameter)
            return Q

        @staticmethod
        def backward(ctx, dl_dQ):
            Q, env_parameter = ctx.saved_tensors

            with torch.enable_grad():
                env_parameter_var = torch.autograd.Variable(env_parameter, requires_grad=True)
                env_var = env_wrapper(env_parameter_var)

                if backprop_method in [0,4]:
                    x, nonzero_indices = compute_policy_hessian_gradient(env_var, Q.detach(), num_episodes=100, discount=discount, lamb=lamb, device=device, backprop_method=backprop_method)
                    surrogate = - dl_dQ.flatten()[nonzero_indices] @ x.flatten()
                    dl_denv = torch.autograd.grad(surrogate, env_parameter_var)[0]
                elif backprop_method in [1,2,3,5,6,7]:
                    dl_denv = compute_policy_hessian_inverse_vector_product(-dl_dQ.flatten(), env_parameter_var, env_var, Q.detach(), num_episodes=100, discount=discount, lamb=lamb, device=device, backprop_method=backprop_method)
                    # surrogate = - dl_dQ.flatten() @ x.flatten()
                    # dl_denv = torch.autograd.grad(surrogate, env_parameter_var)[0]

                # print('gradient norm (pre): {}, norm (post): {}'.format(torch.norm(dl_dpolicy.flatten()), torch.norm(dl_denv.flatten())))

            return dl_denv
    return DiffQFn.apply


