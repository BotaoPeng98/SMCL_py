import numpy as np

def label_correction(class_arr, label, mode):
    """
    Make class and label consistent for clustering.
    
    Parameters:
    class_arr (array-like): Array of class labels
    label (array-like): Array of labels to be corrected
    mode (int): Mode of correction (1 for ratio-based, other for order-based)
    
    Returns:
    new_label (numpy.ndarray): Corrected labels
    """
    
    new_label = np.zeros(len(class_arr), dtype=int)
    uc = np.unique(class_arr)
    ul = np.unique(label)
    num_uc = len(uc)
    num_ul = len(ul)

    if mode == 1:
        # assign by ratio
        for i in range(num_ul):
            nij = np.zeros(num_uc)
            for j in range(num_uc):
                nij[j] = np.sum((label == ul[i]) & (class_arr == uc[j])) / np.sum(class_arr == uc[j])
            max_idx = np.argmax(nij)
            new_label[label == ul[i]] = uc[max_idx]
    
    else:
        # assign by order
        label_num = np.array([np.sum(label == l) for l in ul])
        sort_idx = np.argsort(label_num)[::-1]
        for i in range(num_ul):
            new_label[label == ul[sort_idx[i]]] = i + 1

    return new_label

