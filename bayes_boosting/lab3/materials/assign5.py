import numpy as np
import numpy.linalg
import labfuns
from assign4 import mlParams
import matplotlib.pyplot as plt

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
    for cidx, c in enumerate(classes):  # c means class
        idxs = np.where(labels == c)
        prior[cidx, 0] = np.sum(W[idxs])
    # ==========================
    return prior


# NOTE: you do not need to handle the W argument for this part!
# in:      X - N x d matrix of N data points
#     labels - N vector of class labels
# out:    mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
def mlParams(X, labels, W=None):
    from assign4 import mlParams as _mlParams
    mu, sigma = _mlParams(X, labels, W)
    return mu, sigma

# in:      X - N x d matrix of M data points
#      prior - C x 1 matrix of class priors
#         mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
# out:     h - N vector of class predictions for test points
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


# The implemented functions can now be summarized into the `BayesClassifier` class, which we will use later to test the classifier, no need to add anything else here:


# NOTE: no need to touch this
class BayesClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, X, labels, W=None):
        rtn = BayesClassifier()
        rtn.prior = computePrior(labels, W)
        rtn.mu, rtn.sigma = mlParams(X, labels, W)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBayes(X, self.prior, self.mu, self.sigma)


# ## Boosting functions to implement
#
# The lab descriptions state what each function should do.

# in: base_classifier - a classifier of the type that we will boost, e.g. BayesClassifier
#                   X - N x d matrix of N data points
#              labels - N vector of class labels
#                   T - number of boosting iterations
# out:    classifiers - (maximum) length T Python list of trained classifiers
#              alphas - (maximum) length T Python list of vote weights
def trainBoost(base_classifier, X, labels, T=10):
    # these will come in handy later on
    Npts,Ndims = np.shape(X)

    classifiers = [] # append new classifiers to this list
    alphas = [] # append the vote weight of the classifiers to this list

    # The weights for the first iteration
    wCur = np.ones((Npts,1))/float(Npts)

    for i_iter in range(0, T):
        # a new classifier can be trained like this, given the current weights
        classifiers.append(base_classifier.trainClassifier(X, labels, wCur))

        # do classification for each point
        vote = classifiers[-1].classify(X)

        # TODO: Fill in the rest, construct the alphas etc.
        # ==========================
        err = wCur.T.dot(1 - (vote == labels))
        alpha = (np.log(1 - err) - np.log(err)) * 0.5
        alphas.append(alpha)
        if err >= 0.5:
            break

        # update wCur
        # if classified correctly, reduce the weight by np.exp(-alpha)
        idxs = np.where(vote == labels)
        wCur[idxs] = wCur[idxs] * np.exp(-alpha)

        # if classified wrongly, increase the weight by np.exp(alpha)
        idxs = np.where(vote != labels)
        wCur[idxs] = wCur[idxs] * np.exp(alpha)

        # normalize the weighting
        wCur = wCur / np.sum(wCur)
        # ==========================

    return classifiers, alphas

# in:       X - N x d matrix of N data points
# classifiers - (maximum) length T Python list of trained classifiers as above
#      alphas - (maximum) length T Python list of vote weights
#    Nclasses - the number of different classes
# out:  yPred - N vector of class predictions for test points
def classifyBoost(X, classifiers, alphas, Nclasses):
    Npts = X.shape[0]
    Ncomps = len(classifiers)

    # if we only have one classifier, we may just classify directly
    if Ncomps == 1:
        return classifiers[0].classify(X)
    else:
        votes = np.zeros((Npts,Nclasses))

        # TODO: implement classificiation when we have trained several classifiers!
        # here we can do it by filling in the votes vector with weighted votes
        # ==========================
        for i in range(Ncomps):
            alpha = alphas[i]
            classifier = classifiers[i]
            vote = classifier.classify(X)
            votes[np.arange(Npts), vote] += alpha

        # ==========================

        # one way to compute yPred after accumulating the votes
        return np.argmax(votes,axis=1)


# NOTE: no need to touch this
class BoostClassifier(object):
    def __init__(self, base_classifier, T=10):
        self.base_classifier = base_classifier
        self.T = T
        self.trained = False

    def trainClassifier(self, X, labels):
        rtn = BoostClassifier(self.base_classifier, self.T)
        rtn.nbr_classes = np.size(np.unique(labels))
        rtn.classifiers, rtn.alphas = trainBoost(self.base_classifier, X, labels, self.T)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBoost(X, self.classifiers, self.alphas, self.nbr_classes)

if __name__ == '__main__':

    # ==================================
    #  MAKE PLOTS
    # ==================================
    # print("Without Boosting:")
    # labfuns.plotBoundary(BayesClassifier(), dataset='vowel',split=0.7)
    # # ax = plt.gca()
    # # ax.set_xlim([-4, 4])
    # # ax.set_ylim([-1.5, 1.5])
    # plt.title("Without boosting: NB on Vowel")
    # plt.savefig("woboost_vowel_70.png", bbox_inches='tight')
    # plt.clf()
    # print("With Boosting:")
    # labfuns.plotBoundary(BoostClassifier(BayesClassifier(), T=10), dataset='vowel',split=0.7)
    # # ax = plt.gca()
    # # ax.set_xlim([-4, 4])
    # # ax.set_ylim([-1.5, 1.5])
    # plt.title("With boosting: NB on Vowel")
    # plt.savefig("wboost_vowel_70.png", bbox_inches='tight')
    # plt.clf()
    # =========================================================================
    print("Iris Data set:")
    print("Without Boosting:")
    labfuns.testClassifier(BayesClassifier(), dataset='iris',split=0.7)
    print("With Boosting:")
    labfuns.testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='iris',split=0.7)
