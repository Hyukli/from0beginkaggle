from sklearn.datasets import load_boston
boston=load_boston()
from sklearn.cross_validation import train_test_split
import numpy as np
X=boston.data
y=boston.target

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=33,test_size=0.25)

print "The max target value is",np.max(boston.target)
print "The min target value is",np.min(boston.target)
print "The average taregt value is",np.mean(boston.target)

from sklearn.preprocessing import StandardScaler

ss_X=StandardScaler()
ss_y=StandardScaler()

X_train=ss_X.fit_transform(X_train)
X_test=ss_X.transform(X_test)
y_train=ss_y.fit_transform(y_train.reshape(-1, 1))
y_test=ss_y.transform(y_test.reshape(-1, 1))

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)
lr_y_predict=lr.predict(X_test)

from sklearn.linear_model import SGDRegressor

sgdr=SGDRegressor(max_iter=5)

sgdr.fit(X_train,y_train.ravel())
sgdr_y_predict=sgdr.predict(X_test)