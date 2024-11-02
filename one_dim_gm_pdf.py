import numpy as np
from scipy.stats import multivariate_normal

def one_dim_gm_pdf(data, m, pred):
    """
    1-d Gaussian Mixture Probability Density Function
    
    Parameters:
    data: input data
    m: vectors of seed points
    pred: prediction result
    
    Returns:
    pairwise_sep: pairwise separation measure between subclusters
    """
    
    K = m.shape[0]
    pairwise_sep = np.zeros((K, K))

    for i in range(K):
        for j in range(i+1, K):
            c_i = np.mean(data[pred==i, :], axis=0)
            c_j = np.mean(data[pred==j, :], axis=0)
            c_0 = (c_i + c_j) / 2
            
            data_i = data[pred==i,:]
            data_j = data[pred==j,:]
            data_i_proj = np.dot(data_i - c_0, (c_i - c_j).T) / np.linalg.norm(c_i - c_j)**2
            data_j_proj = np.dot(data_j - c_0, (c_i - c_j).T) / np.linalg.norm(c_i - c_j)**2
            
            sigma_mat = np.array([np.std(data_j_proj)**2, np.std(data_i_proj)**2])
            
            GMModel_mu = np.array([0.5, -0.5])
            weights = np.array([np.sum(pred == i), np.sum(pred == j)]) / (np.sum(pred == i) + np.sum(pred == j))
            
            x = np.arange(-0.5, 0.51, 0.01)
            y = weights[0] * multivariate_normal.pdf(x, mean=GMModel_mu[0], cov=sigma_mat[0]) + \
                weights[1] * multivariate_normal.pdf(x, mean=GMModel_mu[1], cov=sigma_mat[1])
            
            pairwise_sep[i,j] = 1 / np.min(y)
    
    pairwise_sep = np.maximum(pairwise_sep, pairwise_sep.T)
    
    return pairwise_sep

