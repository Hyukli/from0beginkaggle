from sklearn.datasets import  fetch_20newsgroups
from sklearn.cross_validation import train_test_split
news=fetch_20newsgroups(subset='all')
#print len(news.data)
#print news.data[0]
X_train,X_test,y_train,t_test=train_test_split(news.data,news.target,test_size=0.25,random_state=33)