import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
import svm


if __name__ == '__main__':
    # ##########################
    # prepare data
    # ##########################

    poly_degree = 10
    N = 30
    factor = 0.4

    y_loc = 0.2
    x_loc = 1.5

    # set seed
    np.random.seed(100)

    # class A feature vectors
    clsA = np.concatenate((randn(N//2, 2) * factor + [x_loc, y_loc],
                           randn(N//2, 2) * factor + [-x_loc, y_loc]))
    # class B feature vectors
    clsB = randn(N, 2) * factor + [0.0, -y_loc]


    # concat class A and class B together
    inputs = np.concatenate([clsA, clsB])
    targets = np.concatenate((np.ones(clsA.shape[0]), -np.ones(clsB.shape[0])))

    # number of samples
    nSample = inputs.shape[0]

    permute = list(range(nSample))

    np.random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]

    # ====================================================================

    model = svm.SupportVectorMachine(inputs, targets, kernel="polynomail", p=poly_degree)

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
    plt.title("SVM with {} sample data; p = {}".format(nSample, poly_degree))

    # plot the graph
    plt.plot(
        [p[0] for p in clsA],
        [p[1] for p in clsA], 'b.')

    plt.plot(
        [p[0] for p in clsB],
        [p[1] for p in clsB], 'rx')

    plt.axis("equal")
    plt.savefig('ploysvm{}_{}.png'.format(nSample, poly_degree))
    plt.close()

