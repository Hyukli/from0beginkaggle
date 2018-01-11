X_train=[[6],[8],[10],[14],[18]]
y_train=[[7],[9],[13],[17.5],[18]]
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly2=PolynomialFeatures(degree=2)
poly4=PolynomialFeatures(degree=4)
X_train_poly2=poly2.fit_transform(X_train)
X_train_poly4=poly4.fit_transform(X_train)

import numpy as np
xx=np.linspace(0,26,100)
xx=xx.reshape(xx.shape[0],1)

regressor_ploy2=LinearRegression()
regressor_ploy4=LinearRegression()
regressor_ploy2.fit(X_train_poly2,y_train)
regressor_ploy4.fit(X_train_poly4,y_train)
xx_poly2=poly2.transform(xx)
xx_poly4=poly4.transform(xx)

import matplotlib.pyplot as plt
yy_poly2=regressor_ploy2.predict(xx_poly2)
yy_poly4=regressor_ploy4.predict(xx_poly4)
plt.scatter(X_train,y_train)
regressor=LinearRegression()
regressor.fit(X_train,y_train)
yy=regressor.predict(xx)

X_test=[[6],[8],[11],[16]]
y_test=[[8],[12],[15],[18]]

from sklearn.linear_model import Lasso
lasso_poly4=Lasso()
lasso_poly4.fit(X_train_poly4,y_train)
X_test_poly2=poly2.transform(X_test)
X_test_poly4=poly4.transform(X_test)
print lasso_poly4.score(X_test_poly4,y_test)
print lasso_poly4.coef_
print regressor_ploy4.score(X_test_poly4,y_test)
print regressor_ploy4.coef_