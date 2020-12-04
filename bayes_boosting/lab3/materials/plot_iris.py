import labfuns
from assign1 import mlParams

if __name__ == '__main__':

    X, labels, pcadim = labfuns.fetchDataset("iris")
    mu, sigma = mlParams(X,labels)
