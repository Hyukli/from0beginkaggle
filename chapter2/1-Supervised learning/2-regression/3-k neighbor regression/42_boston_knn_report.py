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

from sklearn.neighbors import KNeighborsRegressor

uni_knr=KNeighborsRegressor(weights='uniform')
uni_knr.fit(X_train,y_train)
uni_knr_y_predict=uni_knr.predict(X_test)

dis_knr=KNeighborsRegressor(weights='distance')
dis_knr.fit(X_train,y_train)
dis_knr_y_predict=dis_knr.predict(X_test)

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

print 'R-squared value of uniform-weighted KNeiborRegression:',uni_knr.score(X_test,y_test)
print 'The mean squared error of uniform-weighted KNeighorRegression:',mean_squared_error\
    (ss_y.inverse_transform(y_test),ss_y.inverse_transform(uni_knr_y_predict))
print 'The mean absoluate error of uniform-weighted KNeighorRegression:',mean_absolute_error\
    (ss_y.inverse_transform(y_test),ss_y.inverse_transform(uni_knr_y_predict))

print 'R-squared value of distance-weighted KNeighborRegssion:',dis_knr.score(X_test,y_test)
print 'The mean squared error of distance-weighted KNeighborRegression:',mean_squared_error\
    (ss_y.inverse_transform(y_test),ss_y.inverse_transform(dis_knr_y_predict))
print 'The mean absolute error of distance-weighted KNeighborRegression:',mean_absolute_error\
    (ss_y.inverse_transform(y_test),ss_y.inverse_transform(dis_knr_y_predict))