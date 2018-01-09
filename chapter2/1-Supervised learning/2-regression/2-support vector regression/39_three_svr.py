from sklearn.datasets import load_boston
boston=load_boston()
from sklearn.cross_validation import train_test_split
import numpy as np
X=boston.data
y=boston.target

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=33,test_size=0.25)


from sklearn.preprocessing import StandardScaler

ss_X=StandardScaler()
ss_y=StandardScaler()

X_train=ss_X.fit_transform(X_train)
X_test=ss_X.transform(X_test)
y_train=ss_y.fit_transform(y_train.reshape(-1, 1))
y_test=ss_y.transform(y_test.reshape(-1, 1))

from sklearn.svm import SVR

linear_svr=SVR(kernel='linear')
linear_svr.fit(X_train,y_train)
linear_svr_y_predict=linear_svr.predict(X_test)

poly_svr=SVR(kernel='poly')
poly_svr.fit(X_train,y_train)
poly_svr_y_predict=poly_svr.predict(X_test)

rbf_svr=SVR(kernel='rbf')
rbf_svr.fit(X_train,y_train)
rbf_svr_y_predict=rbf_svr.predict(X_test)

