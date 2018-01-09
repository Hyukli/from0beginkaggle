from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits
digits=load_digits()

X_train,X_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.25,random_state=33)

print y_train.shape
print y_test.shape
