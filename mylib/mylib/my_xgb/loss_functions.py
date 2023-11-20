import numpy as np


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))


def softprob_obj(data, labels, prev_pred, weights=None):
    '''Loss function.  Computing the gradient and approximated hessian (diagonal).
    Reimplements the `multi:softprob` inside XGBoost.
    '''

    # The prediction is of shape (rows, classes), each element in a row
    # represents a raw prediction (leaf weight, hasn't gone through softmax
    # yet).  In XGBoost 1.0.0, the prediction is transformed by a softmax
    # function, fixed in later versions.
    
    kRows    = prev_pred.shape[0]
    kClasses = prev_pred.shape[1]
    
    grad = np.zeros((kRows, kClasses), dtype=float)
    hess = np.zeros((kRows, kClasses), dtype=float)

    eps = 1e-6

    # compute the gradient and hessian, slow iterations in Python, only
    # suitable for demo.  Also the one in native XGBoost core is more robust to
    # numeric overflow as we don't do anything to mitigate the `exp` in
    # `softmax` here.
    for r in range(kRows):
        target = labels[r]
        p = softmax(prev_pred[r, :])
        for c in range(kClasses):
            #assert target >= 0 or target <= kClasses
            g = p[c] - 1.0 if c == target else p[c]
            h = max((2.0 * p[c] * (1.0 - p[c])).item(), eps)
            grad[r, c] = g
            hess[r, c] = h
    
    if weights is not None:
        grad = grad*weights
        hess = hess*weights
        
    # Right now (XGBoost 1.0.0), reshaping is necessary
    # grad = grad.reshape((kRows * kClasses, 1))
    # hess = hess.reshape((kRows * kClasses, 1))
    return grad, hess