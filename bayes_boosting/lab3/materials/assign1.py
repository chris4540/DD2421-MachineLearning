import numpy as np
import labfuns

def mlParams(X, labels, W=None):
    """
    NOTE: you do not need to handle the W argument for this part!
    in:      X - N x d matrix of N data points
        labels - N vector of class labels
    out:    mu - C x d matrix of class means (mu[i] - class i mean)
         sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
    """
    assert(X.shape[0]==labels.shape[0])
    Npts, Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    if W is None:
        W = np.ones((Npts,1))/float(Npts)

    mu = np.zeros((Nclasses,Ndims))
    sigma = np.zeros((Nclasses,Ndims,Ndims))

    # ==================================================================
    # calucalte the mu and sigma maximum likehood estimator
    # ==================================================================
    for cidx, c in enumerate(classes):  # c means class
        idxs = np.where(labels == c)   # the indices of this class
        mu[cidx, :] = np.mean(X[idxs], axis=0)  # avg over the axis 0, keep axis 1

        # We make the sigma matrix as diagonal matrix, which is a naive assumption
        # dvi: deviation (x - mu)
        dvi = X[idxs] - mu[cidx, :]
        diag = np.full(Ndims, np.mean(dvi**2, axis=0))
        sigma[cidx, :] = np.diag(diag)

    return mu, sigma

# ======================================================================
if __name__ == '__main__':
    # ## Test the Maximum Likelihood estimates
    #
    # Call `genBlobs` and `plotGaussian` to verify your estimates.
    X, labels = labfuns.genBlobs(centers=5)
    mu, sigma = mlParams(X,labels)
    # here we use 68-95-99.7 rule
    # the default of plotGaussian plotting 2 sigma, which is 95% Confident interval
    labfuns.plotGaussian(X, labels, mu, sigma)
