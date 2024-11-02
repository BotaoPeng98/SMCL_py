import numpy as np
from scipy.spatial.distance import cdist

def density_gap(data, m, n, rho, data_dist):
    # MAX_DENSITY_GAP_IDX   Find the index of seed point with max density gap.
    #   data: input data
    #   m: vectors of seed points
    #   n: cumulative number of winning times
    #   rho: sample local density
    #   data_dist: data pairwise distance matrix
    #
    #   max_density_gap_idx: the index of seed point with max density gap

    K = m.shape[0]
    min_idx = np.argmin(cdist(m, data), axis=0)
    pred = min_idx
    mean_dist = np.mean(data_dist[data_dist != 0])

    delta = np.zeros(K)

    for i in range(K):
        min_dist = []
        idx = np.where(pred == i)[0]
        rho_i = rho[idx]
        for j in range(len(idx)):
            if rho_i[j] > np.mean(rho_i):
                larger_idx = np.where(rho_i[j] < rho_i)[0]
                if larger_idx.size == 0:
                    min_dist.append(0)
                else:
                    min_dist.append(np.min(cdist(data[idx[j], :].reshape(1, -1), data[idx[larger_idx], :])))

            else:
                min_dist.append(0)
        delta[i] = np.max(min_dist)
        delta[i] = delta[i] / mean_dist

    max_density_gap_idx = np.argmax(n * delta)
    return max_density_gap_idx

