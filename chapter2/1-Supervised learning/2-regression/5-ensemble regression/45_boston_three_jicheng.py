from sklearn.datasets import load_boston
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