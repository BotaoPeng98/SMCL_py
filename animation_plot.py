import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def animation_plot(i, epoch, m, data, options):
    # ANIMATION_PLOT   Plot the data and seed points with animation.
    #   i: the ith data in the current epoch
    #   epoch: current epoch
    #   m: vectors of seed points
    #   options: plot options

    data_num = data.shape[0]        
    d = cdist(m, data)
    min_idx = np.argmin(d, axis=0)
    pred = min_idx

    if (data_num * (epoch - 1) + i) % options['mod_num'] == 0:
        
        if pred.size == 0:
            d = cdist(m, data)
            min_idx = np.argmin(d, axis=0)
            pred = min_idx
        
        plt.cla()
        for k in range(m.shape[0]):
            plt.plot(data[pred == k, 0], data[pred == k, 1], '.', markersize=10)
            plt.plot(m[k, 0], m[k, 1], 'pk', markerfacecolor='k', markersize=20)
        plt.xlabel(f'epoch={epoch}')
        plt.axis(options['fix_plot'])
        plt.draw()
        plt.savefig(f"./results/animation_plot_{epoch}_{i}.png")
        plt.pause(0.001)

