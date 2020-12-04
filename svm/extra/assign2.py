"""
References
------------
1. https://www.baeldung.com/cs/svm-multiclass-classification
2. https://shomy.top/2017/02/20/svm04-soft-margin/
3. http://people.csail.mit.edu/dsontag/courses/ml13/slides/lecture6.pdf
"""

import pandas as pd
import numpy as np


class MulticlassSVM:
    """
    Simply use one-vs-rest
    """
    def __init__(self, n_classes=3, ploy_dim=2):
        self.alpha_times_y = None
        self.ploy_dim = ploy_dim
        self.n_classes = n_classes

    def kernel_fn(self, p_vec, q_vec):
        return (p_vec.dot(q_vec))**self.ploy_dim

    def fit(self, X, y):
        n_samples, n_dim = X.shape

        # record down data
        self.data = X
        self.n_dim = n_dim
        self.n_samples = n_samples
        # this is tricky as usually we use alpha to weight samples
        # but they use \alpha_i \y_i to represent the fake weighting w_i
        self.alpha_times_y = np.zeros((self.n_classes, n_samples))

        # ------------------------------------------------------
        # How to optimize this loop? Levae Douglas as exercise
        for i in range(self.n_samples):
            inner_prod = self._inner_prod(X[i, :])
            preds = self.sign(inner_prod)
            truth = self._label_to_custom_onehot(y[i])
            for j in range(self.n_classes):
                if inner_prod[j]*truth[j] <= 0:
                    # incorrect prediction, apply delta rule
                    self.alpha_times_y[j, i] -= preds[j]

    def _inner_prod(self, x):
        assert x.shape == (self.n_dim,)

        ret = np.zeros((self.n_classes,))
        for i in range(self.n_samples):
            kernel_val = self.kernel_fn(x, self.data[i, :])
            weights = self.alpha_times_y[:, i]
            ret += kernel_val*weights
        assert ret.shape == (self.n_classes,)
        return ret

    def predict(self, x):
        inner_prod = self._inner_prod(x)
        return np.argmax(inner_prod)

    @staticmethod
    def sign(val):
        ret = np.where(val <= 0.0, -1, 1)
        return ret

    def _label_to_custom_onehot(self, lable: int):
        """
        Similar to one-hot encoding, we mark the truth one as 1, otherwise -1

        Example:
        >>> svm = MulticlassSVM(n_classes=5)
        >>> svm._label_to_custom_onehot(2)
        [-1, -1, 1, -1, -1]
        >>> svm._label_to_custom_onehot(4)
        [-1, -1, -1, -1, 1]
        """
        ret = np.full((self.n_classes,), -1)
        ret[lable] = 1
        return ret


def load_dat(fname):
    dat = pd.read_csv(fname, sep='\s+', header=None).to_numpy()
    y = dat[:, 0].astype(np.int) - 1  # we are zero-based indexing
    X = dat[:, 1:].astype(np.float32)
    return X, y


if __name__ == "__main__":
    X_train, y_train = load_dat("dtrain123.dat")
    svm = MulticlassSVM(n_classes=3)
    svm.fit(X_train, y_train)

    nsamples = X_train.shape[0]
    correct = 0
    for i in range(nsamples):
        x = X_train[i, :]
        y = y_train[i]
        if y == svm.predict(x):
            correct += 1
    # this is different from the mathematica impl., we eval the svm after
    # training
    print("train correct = ", correct)
    print("train acc. = ", correct / nsamples)
    print("train misstake = ", nsamples - correct)
    # -------------------------------------------------
    # pd.read_csv(filename, sep='\s+',header=None)
    X_test, y_test = load_dat("dtest123.dat")
    nsamples = X_test.shape[0]
    correct = 0
    for i in range(nsamples):
        x = X_test[i, :]
        y = y_test[i]
        if y == svm.predict(x):
            correct += 1
    print("test correct = ", correct)
    print("test acc. = ", correct / nsamples)
    print("test misstake = ", nsamples - correct)
