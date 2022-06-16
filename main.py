from util import LoadDataset
from numpy import load

if __name__ == '__main__':
    # load training and validation set
    data = load('5-celebrity-faces-dataset.npz')
    