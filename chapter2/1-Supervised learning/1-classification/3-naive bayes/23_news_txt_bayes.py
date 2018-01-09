from sklearn.datasets import  fetch_20newsgroups
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
news=fetch_20newsgroups(subset='all')
#print len(news.data)
#print news.data[0]
X_train,X_test,y_train,t_test=train_test_split(news.data,news.target,\
                                               test_size=0.25,random_state=33)
vec=CountVectorizer()
X_train=vec.fit_transform(X_train)
X_test=vec.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
mnb=MultinomialNB()
mnb.fit(X_train,y_train)
y_predict=mnb.predict(X_test)