X_train=[[6],[8],[10],[14],[18]]
y_train=[[7],[9],[13],[17.5],[18]]
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly2=PolynomialFeatures(degree=2)
X_train_poly2=poly2.fit_transform(X_train)

import numpy as np
xx=np.linspace(0,26,100)
xx=xx.reshape(xx.shape[0],1)

regressor_ploy2=LinearRegression()
regressor_ploy2.fit(X_train_poly2,y_train)
xx_poly2=poly2.transform(xx)

import matplotlib.pyplot as plt
yy_poly2=regressor_ploy2.predict(xx_poly2)
plt.scatter(X_train,y_train)
regressor=LinearRegression()
regressor.fit(X_train,y_train)
yy=regressor.predict(xx)

plt1,=plt.plot(xx,yy,label='Degree=1')
plt2,=plt.plot(xx,yy_poly2,label='Degree=2')

plt.axis([0,25,0,25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles=[plt1,plt2])
plt.show()

print 'The R-squared value of Polynominal Regressor(Degree=2)\
performing on the training data is',regressor_ploy2.score(X_train_poly2,y_train)

