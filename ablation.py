import numpy as np
import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pdb

colors = {
        'TS': 'blue',
        'DF-PG-Full': 'magenta',
        'DF-PG-Woodbury': 'red',
        'DF-PG-Identity': 'goldenrod',
        'DF-Bellman-Full': 'cyan',
        'DF-Bellman-Woodbury': 'green',
        'DF-Bellman-Identity': 'purple',
        }

linestyles = {
        'TS': 'dashdot',
        'DF-PG-Full': 'dotted',
        'DF-PG-Woodbury': 'solid',
        'DF-PG-Identity': 'dashed',
        'DF-Bellman-Full': 'dotted',
        'DF-Bellman-Woodbury': 'solid',
        'DF-Bellman-Identity': 'dashed',
        }

def plot_algo(ax, algo_name, algo_label, algo_mean, algo_std, lw=1, alpha=0.2):
    ax.plot(range(len(algo_mean)), algo_mean, label=algo_label, lw=lw, color=colors[algo_name], linestyle=linestyles[algo_name])
    ax.fill_between(range(len(algo_mean)), 
                    algo_mean+algo_std, 
                    algo_mean-algo_std,
                    facecolor=colors[algo_name], 
                    alpha=alpha)

def plot_figures(plot_dict, algorithm_list, item, filename, plot_legend=True):
    fig = plt.figure()
    lw = 2
    # fig.subplots_adjust(hspace=0.4, wspace=0.4)
    col = len(plot_dict)   # graph name
    nrows = 1
    alpha_shade = 0.18
    fig, axes = plt.subplots(nrows=nrows, ncols=col, figsize=(10*col,6))
    for i in range(nrows):
        for j, plot_key in enumerate(plot_dict):
            item, partition = plot_key.split()
            if col == 1:
                ax = axes
            else:
                if nrows == 1:
                    ax = axes[j]
                else:
                    ax = axes[i][j]
            print('plotting', i, j, domain)
            for (algo_name, algo_label, mean, sem) in plot_dict[plot_key]:
                plot_algo(ax, algo_name=algo_name, algo_label=algo_label, algo_mean=mean, algo_std=sem, lw=lw, alpha=0.2)
            idx = i * nrows + j

            #ax.set_ylim([0, 1.05])
            if i == 0:
                ax.set_xlabel('Training Epochs', fontsize=24)
            if True: #j == 0:
                if item == 'loss':
                    if partition == 'train':
                        ax.set_ylabel('Training predictive loss', fontsize=30)
                    else:
                        ax.set_ylabel('Loss', fontsize=30)

                    # ax.set_yticks([0, 5, 10, 15, 20])
                    ax.set_xlim([0, 100])
                    # ax.set_ylim([0, 50])
                else:
                    if partition == 'train':
                        ax.set_ylabel('Training evaluation', fontsize=30)
                    else:
                        ax.set_ylabel('Testing evaluation', fontsize=30)
                    # ax.set_yticks([0])
                    ax.set_xlim([0, 100])
                    # ax.set_ylim([0, 20])
            #ax.set_xlabel('Training Steps', fontsize=7)
            #ax.set_ylabel('Mean Test Won Rate', fontsize=7)
            #ax.legend(fontsize=10, frameon=True, loc='upper left', facecolor='white', framealpha=0.9, edgecolor='white')
            ax.spines['right'].set_visible(True)
            ax.spines['top'].set_visible(True)
            ax.grid(color='grey', ls = ':', lw=0.5)
            #ax.spines['right'].set_color((.5,.5,.5))
            #ax.spines['top'].set_color((.9,.9,.9))
            default_xticks_range = 50
            pivot_len = len(mean) + 5
            if pivot_len > 200:
                default_xticks_range = 50
                xlabels = ['{}'.format(x if int(x)!= x else int(x)) + ('M' if x !=0 else '') 
                       for x in np.array(range(0, pivot_len, default_xticks_range))/100.]
            else:
                default_xticks_range = 50
                if 50 <= pivot_len <= 110:
                    default_xticks_range = 20
                    if pivot_len > 100:
                        default_xticks_range = 20
                    xlabels = ['{}'.format(x if int(x)!= x else int(x)) + ('0K' if x !=0 else '') 
                           for x in np.array(range(0, pivot_len, default_xticks_range))]     
                else:
                    xlabels = ['{}'.format(x if int(x)!= x else int(x)) + ('M' if x !=0 else '') 
                           for x in np.array(range(0, pivot_len, default_xticks_range))/100.]
            #print(xlabels)
            #ax.set_xticks(np.array(range(0, pivot_len, default_xticks_range)))
            #ax.set_xticklabels(convert_labels(xlabels))
            # if idx == 0:
            #     # ax.set_yticks([0, 5, 10, 15, 20])
            #     ax.set_ylim([20, 40])
            # else:
            #     # ax.set_yticks([0])
            #     ax.set_ylim([3, 7])

            #print(ax.get_xticks())
            ax.xaxis.set_tick_params(labelsize=24)
            ax.yaxis.set_tick_params(labelsize=24)
            if plot_legend and idx == 0:
                legend = fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fontsize=22, ncol=7, frameon=False)
                # legend = ax.legend(fontsize=6, frameon=False)
                for line in legend.get_lines():
                    line.set_linewidth(1)
                for legobj in legend.legendHandles:
                    legobj.set_linewidth(4.0)
            # if domain == 'gridworld':
            #     title_name = 'Grid world'
            # elif domain == 'snare-finding':
            #     title_name = 'Snare finding'
            # elif domain == 'TB':
            #     title_name = 'Tuberculosis'
            # ax.set_title(title_name, fontsize=24)

    os.makedirs('./images', exist_ok=True)
    plt.savefig('./images/{}.pdf'.format(filename), bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close()
    # plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregating reports')

    method_fullname = {'DF': 'decision-focused', 'TS': 'two-stage'}

    domain_list = ['gridworld', 'snare-finding', 'TB']

    # 0514-1500: no network effect
    # 0515-0100: with network effect 0.2 probability Erdos random graph
    algorithm_dict = {
            'snare-finding': {
                'TS':                  'DF/0808-ablation-noise0.75-demo20-reg2_backprop1_DF_DQN',
                'DF-PG-Woodbury':      'DF/0808-ablation-noise0.75-demo20-reg2_backprop1_DF_DQN',
                'DF-PG-Identity':      'DF/0808-ablation-noise0.75-demo20-reg2_backprop3_DF_DQN',
                'DF-Bellman-Woodbury': 'DF/0808-ablation-noise0.75-demo20-reg2_backprop5_DF_DQN',
                'DF-Bellman-Identity': 'DF/0808-ablation-noise0.75-demo20-reg2_backprop7_DF_DQN',
                },
            'gridworld': {
                'TS':                  'TS/0527-1400-noise0.75-demo0_backprop1_TS_Q-learning',
                'DF-PG-Full':          'DF/0527-1400-noise0.75-demo0_backprop0_DF_Q-learning',
                'DF-PG-Woodbury':      'DF/0527-1400-noise0.75-demo0_backprop1_DF_Q-learning',
                'DF-PG-Identity':      'DF/0527-1400-noise0.75-demo0_backprop3_DF_Q-learning',
                'DF-Bellman-Full':     'DF/0527-1400-noise0.75-demo0_backprop4_DF_Q-learning',
                'DF-Bellman-Woodbury': 'DF/0527-1400-noise0.75-demo0_backprop5_DF_Q-learning',
                'DF-Bellman-Identity': 'DF/0527-1400-noise0.75-demo0_backprop7_DF_Q-learning',
                },
            # 'TB': {
            #     'TS':                  'TS/0525-demo0-actioneffect0.4_TS_DQN',
            #     'DF-PG-Woodbury':      'DF/0525-demo0-actioneffect0.4-backprop1_DF_DQN',
            #     'DF-PG-Identity':      'DF/0525-demo0-actioneffect0.4-backprop3_DF_DQN',
            #     'DF-Bellman-Woodbury': 'DF/0525-demo0-actioneffect0.4-backprop5_DF_DQN',
            #     'DF-Bellman-Identity': 'DF/0525-demo0-actioneffect0.4-backprop7_DF_DQN',
            #     }
            'TB': {
                'TS':                  'TS/0524-demo5-observable1_TS_DQN',
                'DF-PG-Woodbury':      'DF/0524-demo5-observable1-backprop1_DF_DQN',
                'DF-PG-Identity':      'DF/0524-demo5-observable1-backprop3_DF_DQN',
                'DF-Bellman-Woodbury': 'DF/0524-demo5-observable1-backprop5_DF_DQN',
                'DF-Bellman-Identity': 'DF/0524-demo5-observable1-backprop7_DF_DQN',
                }
            }

    plot = True

    for domain in domain_list:
        large_plot_dict = {}
        for item in ['loss', 'soft_eval']:
            for partition in ['train', 'validate', 'test']:
                plot_dict = {}
                plot_key = item + ' ' + partition
                plot_dict[plot_key] = []
                large_plot_dict[plot_key] = []
                # algorithm_list = algorithm_dict[domain]
                if domain == 'gridworld':
                    # algorithm_list = ['TS', 'DF-PG-Woodbury', 'DF-Bellman-Woodbury', 'DF-PG-Identity', 'DF-Bellman-Identity', 'DF-PG-Full', 'DF-Bellman-Full']
                    algorithm_list = ['TS', 'DF-PG-Identity', 'DF-Bellman-Identity', 'DF-PG-Woodbury', 'DF-Bellman-Woodbury']
                else:
                    algorithm_list = ['TS', 'DF-PG-Identity', 'DF-Bellman-Identity', 'DF-PG-Woodbury', 'DF-Bellman-Woodbury']

                df_dict = {}
                all_df_list = []
                for method in algorithm_list:
                    prefix = algorithm_dict[domain][method]
                    df_list = []
                    # reading files
                    number_seed = 10 if domain == 'snare-finding' else 10
                    size = None
                    test_performance = []
                    optimal_performance = []
                    initial_performance = []
                    for seed in range(1, number_seed+1):
                        if domain == 'gridworld':
                            if seed in []:
                                continue
                        if domain == 'TB':
                            if seed in []:
                                continue
                        if domain == 'snare-finding':
                            if seed in []:
                                continue
                        file_path = '{}/results/{}_seed{}.csv'.format(domain, prefix, seed)
                        df = pd.read_csv(file_path, delimiter=',', na_values=[' nan', ' inf'], skiprows=1, header=None).dropna()
                        # nan_indices = df.isna().any(axis=1)
                        # df.dropna(axis=0, inplace=True)
                        df_list.append(df)
                        if size is None:
                            size = df.shape[0]
                        # elif size != df.shape[0]:
                        #     print(file_path)
                        #     print('size mismatch!!', size, df.shape[0], seed)

                        # compute the testing performance based on validation choice
                        if method == 'TS' and partition == 'test':
                            optimal_epoch = np.argmin(df[4].values[1:]) + 1 # loss
                        elif 'DF' in method and partition == 'test':
                            optimal_epoch = np.argmax(df[6].values[1:]) + 1 # soft eval

                        if partition == 'test':
                            # print(optimal_epoch)
                            # optimal_epoch = df.shape[0] - 1
                            test_loss           = df[7].iloc[optimal_epoch]
                            test_strict_eval    = df[8].iloc[optimal_epoch] 
                            test_soft_eval      = df[9].iloc[optimal_epoch]
                            optimal_loss        = df[7].iloc[0]
                            optimal_strict_eval = df[8].iloc[0] 
                            optimal_soft_eval   = df[9].iloc[0]
                            initial_loss        = df[7].iloc[1]
                            initial_strict_eval = df[8].iloc[1] 
                            initial_soft_eval   = df[9].iloc[1]
                            if np.isnan(test_soft_eval):
                                print(seed)
                            test_performance.append([test_loss, test_soft_eval, test_strict_eval])
                            optimal_performance.append([optimal_loss, optimal_soft_eval, optimal_strict_eval])
                            initial_performance.append([initial_loss, initial_soft_eval, initial_strict_eval])

                    if partition == 'test' and item == 'loss':
                        performance_mean = np.mean(test_performance, axis=0)[1]
                        performance_sem  = np.std(test_performance, axis=0)[1] / np.sqrt(number_seed)
                        print('Domain {} \t method {} \t mean {:.2f} \t sem {:.2f}'.format(domain, method, performance_mean, performance_sem))
                        # if method == 'TS':
                        #     optimal_performance_mean = np.mean(optimal_performance, axis=0)[1]
                        #     optimal_performance_sem  = np.std(optimal_performance, axis=0)[1] / np.sqrt(number_seed)
                        #     print('Domain {} \t method optimal \t mean {:.2f} \t sem {:.2f}'.format(domain, optimal_performance_mean, optimal_performance_sem))
                        #     initial_performance_mean = np.mean(initial_performance, axis=0)[1]
                        #     initial_performance_sem  = np.std(initial_performance, axis=0)[1] / np.sqrt(number_seed)
                        #     print('Domain {} \t method initial \t mean {:.2f} \t sem {:.2f}'.format(domain, initial_performance_mean, initial_performance_sem))


                    save_path = '{}/results/{}.csv'.format(domain, method)
                    # df_dict[method] = pd.concat(df_list).groupby(level=0).mean()
                    # df_dict[method].to_csv(save_path, index=False)

                    full_df = pd.concat(df_list)
                    full_df_train    = full_df[[0,1,2,3]].copy().rename(columns={0: 'epoch', 1: 'loss', 2: 'strict_eval', 3: 'soft_eval'})
                    full_df_validate = full_df[[0,4,5,6]].copy().rename(columns={0: 'epoch', 4: 'loss', 5: 'strict_eval', 6: 'soft_eval'})
                    full_df_test     = full_df[[0,7,8,9]].copy().rename(columns={0: 'epoch', 7: 'loss', 8: 'strict_eval', 9: 'soft_eval'})

                    if partition == 'train':
                        all_df = full_df_train
                    elif partition == 'validate':
                        all_df = full_df_validate
                    elif partition == 'test':
                        all_df = full_df_test

                    all_df = all_df[all_df['epoch'] != -1]
    
                    epoch_list = list(set(all_df['epoch']))
                    epoch_list.sort()
    
                    tmp_mean = all_df.groupby(by=['epoch'], level=0).mean()[item]
                    tmp_sem  = all_df.groupby(by=['epoch'], level=0).sem()[item]
    
                    plot_dict[plot_key].append((method, method, tmp_mean, tmp_sem))
                    large_plot_dict[plot_key].append((method, method, tmp_mean, tmp_sem))

                plot_figures(plot_dict, algorithm_list, item, "{}/{}_{}".format(domain, partition, item), plot_legend=False)
        plot_figures(large_plot_dict, algorithm_list, item, "{}".format(domain), plot_legend=True)
