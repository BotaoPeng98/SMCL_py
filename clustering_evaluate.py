import numpy as np
from scipy.special import comb
from scipy.sparse import csr_matrix

def clustering_evaluate(target, result):
    """
    Evaluate the clustering result
    
    Parameters:
    target: ground truth
    result: clustering result
    
    Returns:
    output: dictionary containing evaluation metrics
    """
    
    def label_correction(target, result, num_classes):
        # Placeholder for label_correction function
        # Implement this function if needed
        return result
    
    result = label_correction(target, result, 2)
    target = label_correction(target, target, 2)
    
    data_num = len(target)
    target_length = len(np.unique(target))
    result_length = len(np.unique(result))
    
    b = np.zeros(target_length)
    cb = 0
    cd = 0
    for i in range(target_length):
        b[i] = np.sum(target == i+1)
        if b[i] >= 2:
            cb += comb(b[i], 2)
        else:
            cd += 0
    
    d = np.zeros(result_length)
    for i in range(result_length):
        d[i] = np.sum(result == i+1)
        if d[i] >= 2:
            cd += comb(d[i], 2)
        else:
            cd += 0
    
    n = np.zeros((target_length, result_length))
    fval = np.zeros((target_length, result_length))
    cn = 0
    for i in range(target_length):
        for j in range(result_length):
            n[i,j] = np.sum((target == i+1) & (result == j+1))
            if n[i,j] >= 2:
                cn += comb(n[i,j], 2)
            else:
                cn += 0
            rec = n[i,j] / b[i]
            pre = n[i,j] / d[j]
            fval[i,j] = 2 * rec * pre / (rec + pre) if (rec + pre) != 0 else 0
    
    n_max = np.max(n, axis=1)
    max_idx = np.argmax(n, axis=1)
    
    output = {}
    output['acc'] = 1 / data_num * np.sum(n_max)
    output['pre'] = 1 / target_length * np.sum(n_max / b)
    output['rec'] = 1 / target_length * np.sum(n_max / d[max_idx])
    output['fval'] = np.sum(b / data_num * np.max(fval, axis=1))
    
    temp = (cb * cd) / comb(data_num, 2)
    output['ari'] = (cn - temp) / (0.5 * (cb + cd) - temp)
    
    temp = 0
    for l in range(target_length):
        for h in range(result_length):
            temp += 2 * n[l,h] / data_num * np.log(n[l,h] * data_num / (b[l] * d[h]) + np.finfo(float).eps)
    
    output['nmi'] = nmi(target, result)
    
    if result_length == 1:
        output['dcv'] = abs(np.sqrt(np.sum((b-np.mean(b))**2)/(target_length-1))/np.mean(b))
    else:
        output['dcv'] = abs(np.sqrt(np.sum((b-np.mean(b))**2)/(target_length-1))/np.mean(b) - 
                            np.sqrt(np.sum((d-np.mean(d))**2)/(result_length-1))/np.mean(d))
    
    output['cluster_num'] = len(np.unique(result))
    
    return output

def nmi(x, y):
    """
    Compute normalized mutual information I(x,y)/sqrt(H(x)*H(y)) of two discrete variables x and y.
    
    Parameters:
    x, y: two integer vectors of the same length 
    
    Returns:
    z: normalized mutual information z=I(x,y)/sqrt(H(x)*H(y))
    """
    assert len(x) == len(y)
    n = len(x)
    x = np.reshape(x, (1, n))
    y = np.reshape(y, (1, n))
    
    l = min(np.min(x), np.min(y))
    x = x - l + 1
    y = y - l + 1
    k = max(np.max(x), np.max(y))
    
    idx = np.arange(n)
    Mx = csr_matrix((np.ones(n), (idx, x[0]-1)), shape=(n, k))
    My = csr_matrix((np.ones(n), (idx, y[0]-1)), shape=(n, k))
    Pxy = np.array(Mx.T * My / n).ravel()
    Pxy = Pxy[Pxy > 0]
    Hxy = -np.dot(Pxy, np.log2(Pxy))
    
    Px = np.array(Mx.mean(axis=0)).ravel()
    Py = np.array(My.mean(axis=0)).ravel()
    
    Px = Px[Px > 0]
    Py = Py[Py > 0]
    
    Hx = -np.dot(Px, np.log2(Px))
    Hy = -np.dot(Py, np.log2(Py))
    
    MI = Hx + Hy - Hxy
    
    z = np.sqrt((MI/Hx)*(MI/Hy))
    z = max(0, z)
    
    return z

