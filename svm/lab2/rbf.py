import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
import svm


if __name__ == '__main__':
    # ##########################
    # prepare data
    # ##########################

    N = 50
    gamma = .1

    # set seed
    np.random.seed(100)

    X_xor = np.random.randn(N, 2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0,
                       X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, -1)


    permute = list(range(N))

    np.random.shuffle(permute)
    inputs = X_xor[permute, :]
    targets = y_xor[permute]
    nSample = inputs.shape[0]

    # ====================================================================

    model = svm.SupportVectorMachine(inputs, targets, kernel="rbf", gamma=gamma)
    plt.plot(X_xor[y_xor == 1, 0],
             X_xor[y_xor == 1, 1],
             'b.')
    plt.plot(X_xor[y_xor == -1, 0],
             X_xor[y_xor == -1, 1],
             'rx')

    plt.xlim([-3, 3])
    plt.ylim([-3, 3])

    model.train()

    xgrid = np.linspace(-5, 5)
    ygrid = np.linspace(-4, 4)
    grid = np.array([[model.indicator(np.array([x, y]))
                       for x in xgrid]
                       for y in ygrid])

    plt.contour(xgrid, ygrid, grid,
                (-1.0, 0.0, 1.0),
                colors=('red', 'black', 'blue'),
                linewidths=(1 , 3 , 1))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("SVM with {} samples; gamma = {}".format(nSample, gamma))


    plt.axis("equal")
    plt.savefig('rbfplot{}_{}.png'.format(nSample, gamma))
    plt.close()

