from sklearn.datasets import load_boston
import numpy as np
boston=load_boston()
from sklearn.cross_validation import train_test_split
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

from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()
dtr.fit(X_train,y_train)
dtr_y_predict=dtr.predict(X_test)

from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
rfr=RandomForestRegressor()
rfr.fit(X_train,y_train)
rfr_y_predict=rfr.predict(X_test)

etr=ExtraTreesRegressor()
etr.fit(X_train,y_train)
etr_y_predict=etr.predict(X_test)

gbr=GradientBoostingRegressor()
gbr.fit(X_train,y_train)
gbr_y_predict=gbr.predict(X_test)

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

print 'R-squared value of RandomForestRegressor:',rfr.score(X_test,y_test)
print 'The mean squared error of RandomForestRegressor:',mean_squared_error\
    (ss_y.inverse_transform(y_test),ss_y.inverse_transform(rfr_y_predict))
print 'The mean absoluate error of RandomForestRegressor:',mean_absolute_error\
    (ss_y.inverse_transform(y_test),ss_y.inverse_transform(rfr_y_predict))

print np.sort(zip(etr.feature_importances_,boston.feature_names),axis=0)

print 'R-squared value of GradientBoostingRegressor:',gbr.score(X_test,y_test)
print 'The mean squared error of GradientBoostingRegressor:',mean_squared_error\
    (ss_y.inverse_transform(y_test),ss_y.inverse_transform(gbr_y_predict))
print 'The mean absoluate error of GradientBoostingRegressor:',mean_absolute_error\
    (ss_y.inverse_transform(y_test),ss_y.inverse_transform(gbr_y_predict))