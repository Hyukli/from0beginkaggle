from sklearn.datasets import load_iris
iris=load_iris()
#print iris.data.shape
#print iris.DESCR
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(iris.data,iris.target,\
                                               test_size=0.25,random_state=33)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)

knc=KNeighborsClassifier()
knc.fit(X_train,y_train)
y_predict=knc.predict(X_test)