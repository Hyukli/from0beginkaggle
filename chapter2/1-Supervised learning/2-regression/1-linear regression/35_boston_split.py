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

