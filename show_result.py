import numpy as np
import matplotlib.pyplot as plt
from label_correction import label_correction
from clustering_evaluate import clustering_evaluate

def show_result(data, label, pred_all, cluster_num, measure_sep, measure_com, sep_com):
    # Show the numerical results and plot figures
    # data: input data
    # label: ground truth
    # pred_all: all prediction of intermediate clustering result
    # cluster_num: number of clusters
    # measure_sep: global separability
    # measure_com: global compactness
    # sep_com: global separability + global compactness

    _, ia = np.unique(cluster_num, return_index=True)

    ia = np.array(ia)
    cluster_num = np.array(cluster_num)
    
    plt.figure()
    plt.plot(cluster_num[ia], sep_com, '-s', markersize=10)
    plt.plot(cluster_num[ia], measure_sep, '-s', markersize=10)
    plt.plot(cluster_num[ia], measure_com, '-s', markersize=10)
    plt.legend(['sep+com', 'sep', 'com'])
    plt.xlabel('Global measures')
    plt.savefig(f"./results/sep.png")

    # label = label_correction(label, label, 2)
    u_pred = np.unique(label)
    
    plt.figure()
    for i in range(len(u_pred)):
        plt.plot(data[label == u_pred[i], 0], data[label == u_pred[i], 1], '.', markersize=10)
    plt.xlabel('Ground truth')
    plt.savefig(f"./results/gt.png")

    sep_com[0] = float('inf')
    median_pos = np.argmin(sep_com)
    pred = pred_all[median_pos, :]
    # pred = label_correction(label, pred.T, 1)
    u_pred = np.unique(pred)
    plt.figure()
    for i in range(len(u_pred)):
        plt.plot(data[pred == u_pred[i], 0], data[pred == u_pred[i], 1], '.', markersize=10)
    plt.xlabel('SMCL clustering result')
    plt.savefig(f"./results/SMCL_rslt.png")
    # smcl_result = clustering_evaluate(label, pred)

