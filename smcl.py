import numpy as np
import scipy.io
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import time
from pns import pns
from show_result import show_result
from sgms import sgms

def smcl(dataset_name):
    # set random seed
    np.random.seed(123)

    # dataset path and name
    dataset_path = 'dataset/'
    data = scipy.io.loadmat(f"{dataset_path}{dataset_name}.mat")['data']

    # data normalization
    data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

    # precompute
    data_dist = cdist(data, data)
    data_num = data.shape[0]
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    mean_d = np.zeros(data_num)
    
    for i in range(data_num):
        sort_d = np.sort(data_dist[i, :])
        mean_d[i] = np.mean(sort_d[:round(0.02 * data_num)])

    ball_graph = np.zeros((data_num, data_num))
    ball_graph[data_dist < np.mean(mean_d)] = 1

    # parameter setting
    smcl_options = {
        'E': 1000,
        'K0': 2,
        'alpha_c': 0.005,
        'gamma': 0.005,
        'xi': 0.001
    }

    # plot setting
    plot_options = {
        'mod_num': 100,
        'fix_plot': [data_min[0], data_max[0], data_min[1], data_max[1]]
    }

    # main
    print(f'SMCL starts on dataset {dataset_name}...')
    start_time = time.time()

    # PNS: Prototype Number Selection
    m = pns(data, ball_graph, data_dist, smcl_options, plot_options)

    # SGMS: Subcluster Grouping with Model Selection
    pred_all, cluster_num, measure_sep, measure_com, sep_com = sgms(data, m)

    print(f'Finish SMCL in {time.time() - start_time:.4f} seconds')

    # show results and plots
    label = [1] * data_num
    show_result(data, label, pred_all, cluster_num, measure_sep, measure_com, sep_com)
    
if __name__ == '__main__':
    smcl('gaussian')
    # smcl('banana')
    # smcl('ids2')

