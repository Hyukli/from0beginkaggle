from sklearn.datasets import fetch_20newsgroups
news=fetch_20newsgroups(subset='all')

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(\
    news.data,news.target,test_size=0.25,random_state=33)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec=TfidfVectorizer()

X_tfidf_train=tfidf_vec.fit_transform(X_train)
X_tfidf_test=tfidf_vec.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
mnb_tfidf=MultinomialNB()
mnb_tfidf.fit(X_tfidf_train,y_train)
print 'The accuracy of classifying 20newsgroups with Naive Bayes\
(TfidfVectorizer without filtering stopwords):',\
    mnb_tfidf.score(X_tfidf_test,y_test)

y_tfidf_predict=mnb_tfidf.predict(X_tfidf_test)
from sklearn.metrics import classification_report
print classification_report(y_test,y_tfidf_predict,target_names=news.target_names)