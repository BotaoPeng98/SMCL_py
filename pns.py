import numpy as np
from scipy.spatial.distance import cdist
from animation_plot import animation_plot
from density_gap import density_gap
import tqdm

def pns(data, ball_graph, data_dist, smcl_options, plot_options):
    # PNS Prototype Number Selection
    # data: input data
    # ball_graph: precomputed epsilon ball graph
    # data_dist: data pairwise distance matrix
    # smcl_options: parameters of smcl
    # plot_options: plot options

    # m: final vectors of seed points

    vis_flag = 1
    K = smcl_options['K0']
    data_num = data.shape[0]
    rand_idx = np.random.permutation(data.shape[0])
    m = data[rand_idx[:K], :]
    rho = np.sum(ball_graph, axis=0)

    m_old = m.copy()

    for epoch in tqdm.tqdm(range(smcl_options['E']), desc="Prototype number selection:"):
        # print(f'epoch {epoch + 1}')
        rand_idx = np.random.permutation(data_num)
        rand_data = data[rand_idx, :]
        n = np.ones(K)

        for data_i in range(data_num):
            xt = rand_data[data_i, :]
            new_m = m.copy()
            new_nc = n.copy()

            dist = np.sqrt(np.sum((xt - m) ** 2, axis=1))
            sort_idx = np.argsort(dist)
            winner_idx = sort_idx[0] # 以上三行代码对应论文公式(7)
            competitor_idx = sort_idx[1:]

            # update winner
            new_m[winner_idx, :] = m[winner_idx, :] + K * smcl_options['alpha_c'] * (xt - m[winner_idx, :])
            new_nc[winner_idx] += 1

            # update competitors
            for k in range(K - 1):
                beta = np.exp(-(np.linalg.norm(xt - m[competitor_idx[k], :]) ** 2 - 
                                 np.linalg.norm(xt - m[winner_idx, :]) ** 2) / 
                               np.linalg.norm(m[winner_idx, :] - m[competitor_idx[k], :]) ** 2)
                new_m[competitor_idx[k], :] = m[competitor_idx[k], :] - \
                    K * smcl_options['alpha_c'] * smcl_options['gamma'] * beta * (xt - m[competitor_idx[k], :])

            m = new_m
            n = new_nc
            if vis_flag:
                animation_plot(data_i, epoch, m, data, plot_options)
                vis_flag = 0

        max_norm = np.max(np.sum((m[n > 1, :] - m_old[n > 1, :]) ** 2, axis=0))
        if max_norm < smcl_options['xi'] and epoch < smcl_options['E']:
            d = cdist(m, data)
            min_idx = np.argmin(d, axis=0)
            pred = min_idx
            test_a = pred == k
            test_b = np.sum(test_a)
            count = np.array([np.sum(pred == k) for k in range(K)])
            theta = data_num * 0.02
            delete_idx = np.where(count <= theta)[0]
            if delete_idx.size > 0:
                m = np.delete(m, delete_idx, axis=0)
                n = np.delete(n, delete_idx)
                K -= len(delete_idx)
                delete_idx_str = ', '.join(map(str, delete_idx))
                print(f': drive seed point {delete_idx_str} out')
                break

            max_density_gap_idx = density_gap(data, m, n, rho, data_dist)
            m = np.vstack([m, m[max_density_gap_idx, :] + np.ones((1, m.shape[1])) * 1e-4])
            print(f': duplicate seed point {max_density_gap_idx}')
            vis_flag = 1
            
            K += 1
        m_old = m.copy()

    return m

