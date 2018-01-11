X_train=[[6],[8],[10],[14],[18]]
y_train=[[7],[9],[13],[17.5],[18]]
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

import numpy as np
xx=np.linspace(0,26,100)
xx=xx.reshape(xx.shape[0],1)
yy=regressor.predict(xx)

import matplotlib.pyplot as plt
plt.scatter(X_train,y_train)

plt1,=plt.plot(xx,yy,label="Degree=1")

plt.axis([0,25,0,25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles=[plt1])
plt.show()

print 'The R-squared value of Linear Regressor performing\
 on the training data is',regressor.score(X_train,y_train)