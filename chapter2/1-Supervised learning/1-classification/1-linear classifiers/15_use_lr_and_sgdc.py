#encoding:utf-8
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
column_names=['Sample code number','Clump Thickness','Uniformity of Cell Size',\
    'Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size',\
              'Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
data=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',\
                 names=column_names)

data=data.replace(to_replace='?',value=np.nan)
data=data.dropna(how='any')
#print data.shape

X_train,X_test,y_train,y_test=train_test_split\
    (data[column_names[1:10]],data[column_names[10]],test_size=0.25,random_state=33)

#标准化数据，保证每个维度的特征数据方差为1，均值为0，使得预测结果不会被某些维度过大的特征而主导

ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)

lr=LogisticRegression()
sgdc=SGDClassifier()
lr.fit(X_train,y_train)
lr_y_predict=lr.predict(X_test)
sgdc.fit(X_train,y_train)
sgcd_y_predict=sgdc.predict(X_test)