from assign1 import mlParams
from assign2 import computePrior
from assign2 import classifyBayes
import labfuns
from labfuns import testClassifier
import numpy as np
from sklearn import decomposition
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import ColorConverter

class BayesClassifier(object):
    """
    NOTE: no need to touch this
    """
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

def plotBoundary(classifier, dataset='iris', split=0.7):

    X,y,pcadim = labfuns.fetchDataset(dataset)
    xTr,yTr,xTe,yTe,trIdx,teIdx = labfuns.trteSplitEven(X,y,split,1)
    classes = np.unique(y)

    pca = decomposition.PCA(n_components=2)
    pca.fit(xTr)

    xTr = pca.transform(xTr)
    xTe = pca.transform(xTe)

    pX = np.vstack((xTr, xTe))
    py = np.hstack((yTr, yTe))

    # Train
    trained_classifier = classifier.trainClassifier(xTr, yTr)

    xRange = np.arange(np.min(pX[:,0]),np.max(pX[:,0]),np.abs(np.max(pX[:,0])-np.min(pX[:,0]))/100.0)
    yRange = np.arange(np.min(pX[:,1]),np.max(pX[:,1]),np.abs(np.max(pX[:,1])-np.min(pX[:,1]))/100.0)

    grid = np.zeros((yRange.size, xRange.size))

    for (xi, xx) in enumerate(xRange):
        for (yi, yy) in enumerate(yRange):
            # Predict
            grid[yi,xi] = trained_classifier.classify(np.array([[xx, yy]]))


    ys = [i+xx+(i*xx)**2 for i in range(len(classes))]
    colormap = cm.rainbow(np.linspace(0, 1, len(ys)))

    fig = plt.figure()
    # plt.hold(True)
    conv = ColorConverter()
    for (color, c) in zip(colormap, classes):
        try:
            CS = plt.contour(xRange,yRange,(grid==c).astype(float),15,linewidths=0.25,colors=conv.to_rgba_array(color))
        except ValueError:
            pass
        trClIdx = np.where(y[trIdx] == c)[0]
        teClIdx = np.where(y[teIdx] == c)[0]
        plt.scatter(xTr[trClIdx,0],xTr[trClIdx,1],marker='o',c=color,s=40,alpha=0.5, label="Class "+str(c)+" Train")
        plt.scatter(xTe[teClIdx,0],xTe[teClIdx,1],marker='*',c=color,s=50,alpha=0.8, label="Class "+str(c)+" Test")
    plt.legend(bbox_to_anchor=(1., 1), loc=2, borderaxespad=0.)
    fig.subplots_adjust(right=0.7)

if __name__ == '__main__':
    # Call the `testClassifier` and `plotBoundary` functions for this part.
    for s in [.5, .6, .7, .8, .9]:
        print("split = %f" % s)
        testClassifier(BayesClassifier(), dataset='iris', split=s)
        plotBoundary(BayesClassifier(), dataset='iris',split=s)

        # get current axes
        ax = plt.gca()
        ax.set_xlim([-4, 4])
        ax.set_ylim([-1.5, 1.5])
        plt.title("NB Classifier for Iris data. %d%%" % (s*100))
        plt.savefig("iris_%d.png" % (s*100), bbox_inches='tight')
    # plotBoundary(BayesClassifier(), dataset='iris',split=0.9)

    # testClassifier(BayesClassifier(), dataset='vowel', split=0.7)
    # plotBoundary(BayesClassifier(), dataset='vowel', split=0.7)
