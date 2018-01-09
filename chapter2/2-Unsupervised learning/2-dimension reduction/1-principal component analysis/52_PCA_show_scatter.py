import pandas as pd
import numpy as np

digits_train=pd.read_csv('https://archive.ics.uci.edu/ml/\
machine-learning-databases/optdigits/optdigits.tra',header=None)
digits_test=pd.read_csv('https://archive.ics.uci.edu/ml/\
machine-learning-databases/optdigits/optdigits.tes',header=None)

X_digits=digits_train[np.arange(64)]
y_digits=digits_train[64]

from sklearn.decomposition import PCA
estimator=PCA(n_components=2)
X_pca=estimator.fit_transform(X_digits)
from matplotlib import pyplot as plt

def plot_pca_scatter():
    colors=['black','blue','purple','yellow','brown','red','lime','cyan','orange','gray']
    for i in xrange(len(colors)):
        px=X_pca[:,0][y_digits.as_matrix()==i]
        py=X_pca[:,1][y_digits.as_matrix()==i]
        plt.scatter(px,py,c=colors[i])
    plt.legend(np.arange(0,10).astype(str))
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()

plot_pca_scatter()