import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.neighbors import NearestNeighbors

def global_measures(data, pairwise_sep, pred, K):
    """
    Calculate the global separability and global compactness.
    
    Parameters:
    data: input data
    pairwise_sep: pairwise separation measure returned by one_dim_gm_pdf
    pred: prediction result
    K: number of seed points
    
    Returns:
    pred_all: all prediction of intermediate clustering result
    global_sep: global separability
    global_com: global compactness
    cluster_num: number of clusters
    """
    group_set = [{i} for i in range(K)]
    
    pred_all = []
    global_com = []
    
    for k in range(K, 1, -1):
        s = np.full((k, k), np.inf)
        for i in range(1, k):
            for j in range(i):
                ss = [pairwise_sep[ii, jj] for ii in group_set[i] for jj in group_set[j]]
                s[i, j] = min(ss)
        
        min_s = np.min(s)
        i, j = np.unravel_index(np.argmin(s), s.shape)
        
        group_set.append(group_set[i].union(group_set[j]))
        group_set = [group_set[x] for x in range(len(group_set)) if x not in [i, j]]
        global_com.append(min_s)
        
        new_pred = np.zeros_like(pred)
        for i, group in enumerate(group_set):
            new_pred[np.isin(pred, list(group))] = i
        pred_all.append(new_pred)
    
    pred_all = np.array(pred_all)
    cluster_num = [len(np.unique(p)) for p in pred_all]
    _, ia = np.unique(cluster_num, return_index=True)
    
    k = 10
    nn_idx = my_knn(data, data, [], k)
    global_sep = []
    for i in range(len(ia)):
        sep = calculate_global_sep(pred_all[ia[i]], nn_idx)
        global_sep.append(max(global_sep + [sep]))
    
    return pred_all, global_sep, global_com, cluster_num

def calculate_global_sep(pred, nn_idx):
    """
    Calculate global separability
    """
    u_label = np.unique(pred)
    cluster_num = len(u_label)
    
    sep = []
    for i in range(cluster_num):
        mask = pred == u_label[i]
        sep.append(np.sum(np.isin(nn_idx[:, mask], np.where(~mask)[0])) / nn_idx.shape[0])
    
    return max(sep)

def my_knn(sel_data, data, catidx, k):
    """
    Custom k-nearest neighbors implementation
    """
    sel_num, fea_num = sel_data.shape
    k = min(k, sel_num - 1)
    data_num = data.shape[0]
    
    if not catidx:
        catidx = []
    cat_num = len(catidx)
    num_idx = [i for i in range(fea_num) if i not in catidx]
    med_std = np.median(np.std(data[:, num_idx], axis=0))
    
    cat_mat = np.zeros((sel_num, data_num))
    for i in range(sel_num):
        for j in range(data_num):
            for l in catidx:
                cat_mat[i, j] += float(sel_data[i, l] != data[j, l])

    num_pd = cdist(sel_data[:, num_idx], data[:, num_idx])
    sq_dist = np.sqrt(num_pd**2 + med_std**2 * cat_mat)
    
    nn = NearestNeighbors(n_neighbors=k+1, metric='precomputed')
    nn.fit(sq_dist)
    
    _, nn_idx = nn.kneighbors()
    return nn_idx[:, 1:].T

