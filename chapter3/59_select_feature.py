import pandas as pd
titanic=pd.read_csv('http://biostat.mc.vanderbilt.\
edu/wiki/pub/Main/DataSets/titanic.txt')

y=titanic['survived']
X=titanic.drop(['row.names','name','survived'],axis=1)
X['age'].fillna(X['age'].mean,inplace=True)
X.fillna('UNKNOWN',inplace=True)

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)
from sklearn.feature_extraction import DictVectorizer
vec=DictVectorizer()
X_train=vec.fit_transform(X_train.to_dict(orient='record'))
X_test=vec.transform(X_test.to_dict(orient='record'))

print len(vec.feature_names_)
