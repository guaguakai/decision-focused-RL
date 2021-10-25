import pandas as pd
from statistics import mean, stdev
import pdb


problem_sizes = [5, 10, 15, 20, 25]
backprop_methods = [0, 1, 3, 4, 5, 7]

results_mean = {}
results_stdev = {}

for size in problem_sizes:
    results_mean[size] = {}
    results_stdev[size] = {}
    for method in backprop_methods:
        backprop_times = []
        for seed in range(1, 31):
            filename = f"DF/0805-scaling--backprop{method}-problemsize{size}_DF_Q-learning_seed{seed}.csv"
            contents = pd.read_csv(filename)
            backprop_times.extend(contents[' backward time'].to_list())
        results_mean[size][method] = mean(backprop_times)
        results_stdev[size][method] = stdev(backprop_times)

pd.DataFrame(results_mean).to_csv('results_mean.csv')
pd.DataFrame(results_stdev).to_csv('results_stdev.csv')
