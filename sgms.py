import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import norm
from one_dim_gm_pdf import one_dim_gm_pdf
from global_measures import global_measures

def sgms(data, m):
    """
    SGMS Subcluster Grouping with Model Selection
    
    Parameters:
    data: input data
    m: vectors of seed points
    
    Returns:
    pred_all: all prediction of intermediate clustering result
    cluster_num: number of clusters
    global_sep: global separability
    global_com: global compactness
    sep_com: global separability + global compactness
    """
    
    K = m.shape[0]
    d = cdist(m, data)
    min_idx = np.argmin(d, axis=0)
    pred = min_idx

    pairwise_sep = one_dim_gm_pdf(data, m, pred)
    pred_all, global_sep, global_com, cluster_num = global_measures(data, pairwise_sep, pred, K)

    global_sep = global_sep / np.max(global_sep)
    global_com = global_com / np.max(global_com)
    sep_com = global_sep + global_com

    return pred_all, cluster_num, global_sep, global_com, sep_com

