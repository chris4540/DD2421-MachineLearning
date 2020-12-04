import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class Kernal:
    """ Kernal functions
    """

    @staticmethod
    def linear(x, y):
        """ linear kernels
        """
        return x.dot(y)

    @staticmethod
    def polynomail(x, y, p):
        """ polynomail kernels
        """
        return (x.dot(y) + 1)**p

    @staticmethod
    def rbf(x, y, gamma):
        """ Radial basic function

        See also:
        https://chrisalbon.com/machine_learning/support_vector_machines/svc_parameters_using_rbf_kernel/
        """
        return np.exp(-gamma*np.sum((x - y.T)**2))


class SupportVectorMachine:

    # the precompute matrix: P_{ij} = t_i * t_j * K(x_i, x_j)
    precom_mat = None

    targets = None
    nSamples = None

    penalty = None

    THRESHOLD = 1e-7

    def __init__(self, x_arr, y, kernel="linear", p=1, gamma=1, penalty=None):
        """
        Args:
            x_arr (np.array): x_arr.shape[0] = # of samples
                              x_arr.shape[1] = # of features
            y     (np.array): y.shape[0] = # of samples
        """
        assert(x_arr.shape[0] == y.shape[0])

        self.nSamples = y.shape[0]

        # x vectors
        self.data = x_arr
        # t vector
        self.targets = y

        # penalty
        self.penalty = penalty

        if (kernel == "linear"):
            self.kernel_func = Kernal.linear
        elif (kernel == "polynomail"):
            self.kernel_func = lambda x, y: Kernal.polynomail(x, y, p)
        elif (kernel == "rbf"):
            self.kernel_func = lambda x, y: Kernal.rbf(x, y, gamma)
        else:
            raise ValueError("Invalid input")

        self.precom_mat = self._get_precom_matrix()


    def _get_precom_matrix(self):
        # compute the P_{ij}
        y = self.targets
        x_arr = self.data

        ret = y[:, np.newaxis] * self.kernel_func(x_arr, x_arr.T) * y
        return ret


    def _func(self, alpha):
        """
        The objective function we would like to minimize.

        This objective function is the Lagrangian dual of the
        orignial problem


        Args:
            alpha (np.array): the alpha in the eqn.
        """
        return (0.5*self.precom_mat.dot(alpha).dot(alpha)) - np.sum(alpha)

    def _zerofunc(self, alpha):
        """
        The constrain of $$sum_{i} alpha_i t_i = 0$$

        optimize(..., constraints={'type':"eq", 'fun', self._zerofunc}
        """
        return alpha.dot(self.targets)

    def _minimize(self):
        # fg: first guess
        fg = np.zeros(self.nSamples)

        # minimize the dual objective
        self._min_result = minimize(
            self._func, fg,
            bounds=[(0, self.penalty) for _ in range(self.nSamples)],
            constraints={'type':"eq", 'fun': self._zerofunc})

        if not self._min_result['success']:
            raise RuntimeError("Unable to minimize the objective function")

        # save down the alphas
        self._alphas = self._min_result['x']
        # print(self._alphas)

    def _get_bias(self):
        # pick the first non-zero alpha value to calculate b
        sv_idx = np.where(self._alphas > self.THRESHOLD)[0][0]

        sv = self.data[sv_idx]
        sv_t = self.targets[sv_idx]

        # K(x_i, sv)
        k_prod = self.kernel_func(self.data, sv)

        prod = np.sum(self._alphas * self.targets * k_prod)

        self.bias = prod - sv_t

    def train(self):
        # find the alphas
        self._minimize()

        # find the bias b
        self._get_bias()


    def indicator(self, x):

        # TODO: avoid to use for loop
        prod = 0   # w_t *phi(x)
        for i, alpha in enumerate(self._alphas):
            if alpha < self.THRESHOLD:
                continue
            prod += alpha * self.targets[i] * self.kernel_func(x, self.data[i])

        # return np.sign(prod - self.bias)
        return prod - self.bias

    def classify(self, x):
        return np.sign(self.indicator(x))

