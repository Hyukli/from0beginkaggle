import pandas as pd
import numpy as np

digits_train=pd.read_csv('https://archive.ics.uci.edu/ml/\
machine-learning-databases/optdigits/optdigits.tra',header=None)
digits_test=pd.read_csv('https://archive.ics.uci.edu/ml/\
machine-learning-databases/optdigits/optdigits.tes',header=None)

X_train=digits_train[np.arange(64)]
y_train=digits_train[64]

X_test=digits_test[np.arange(64)]
y_test=digits_test[64]

from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

svc=LinearSVC()
svc.fit(X_train,y_train)
y_predict=svc.predict(X_test)

estimator=PCA(n_components=20)

pca_X_train=estimator.fit_transform(X_train)
pca_X_test=estimator.transform(X_test)

pca_svc=LinearSVC()
pca_svc.fit(pca_X_train,y_train)
pca_y_predict=pca_svc.predict(pca_X_test)

from sklearn.metrics import classification_report
print svc.score(X_test,y_test)
print classification_report(y_test,y_predict,target_names=np.arange(10).astype(str))
print pca_svc.score(pca_X_test,y_test)

print classification_report(y_test,pca_y_predict,target_names=np.arange(10).astype(str))


