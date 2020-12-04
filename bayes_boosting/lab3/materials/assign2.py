import numpy as np
import numpy.linalg
import labfuns
from assign1 import mlParams

def computePrior(labels, W=None):
    """
    NOTE: you do not need to handle the W argument for this part!
    in: labels - N vector of class labels
    out: prior - C x 1 vector of class priors
    """
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts,1))/Npts
    else:
        assert(W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    prior = np.zeros((Nclasses,1))

    # TODO: compute the values of prior for each class!
    # ==========================
    for l in labels:
        prior[l] += 1

    prior = prior / len(labels)
    # ==========================

    return prior

def classifyBayes(X, prior, mu, sigma):
    """
    in:      X - N x d matrix of M data points
         prior - C x 1 matrix of class priors
            mu - C x d matrix of class means (mu[i] - class i mean)
         sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
    out:     h - N vector of class predictions for test points
    """
    Npts = X.shape[0]
    Nclasses,Ndims = np.shape(mu)
    logProb = np.zeros((Nclasses, Npts))

    # TODO: fill in the code to compute the log posterior logProb!
    # ==========================
    for c in range(Nclasses):
        log_prior = np.log(prior[c])
        log_det_sigma = np.log(np.linalg.det(sigma[c]))
        # calculate the inverse of the diagonal sigma
        inv_sigma = np.diag(1.0 / np.diag(sigma[c]))
        dev = X - mu[c]  # deviation
        # Mahalanobis distance
        mah_dis_sq = np.diag(dev.dot(inv_sigma).dot(dev.T))

        logProb[c, :] = -0.5*log_det_sigma - 0.5*mah_dis_sq + log_prior


    # ==========================
    # one possible way of finding max a-posteriori once
    # you have computed the log posterior
    h = np.argmax(logProb, axis=0)
    return h

if __name__ == '__main__':
    # testing
    X, labels = labfuns.genBlobs(centers=5)

    prior = computePrior(labels)

    # estimate the mu and sigma from maximum libklihood method
    mu, sigma = mlParams(X, labels)

    ret = classifyBayes(X, prior, mu, sigma)

    print(ret)
    print("--------------------------")
    print(labels)
    print(ret == labels)


