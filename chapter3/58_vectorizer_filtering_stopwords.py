from sklearn.datasets import fetch_20newsgroups
news=fetch_20newsgroups(subset='all')

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(\
    news.data,news.target,test_size=0.25,random_state=33)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
tfidf_vec=TfidfVectorizer()

X_tfidf_train=tfidf_vec.fit_transform(X_train)
X_tfidf_test=tfidf_vec.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
mnb_tfidf=MultinomialNB()
mnb_tfidf.fit(X_tfidf_train,y_train)

y_tfidf_predict=mnb_tfidf.predict(X_tfidf_test)
from sklearn.metrics import classification_report
count_filter_vec,tfidf_filter_vec=\
    CountVectorizer(analyzer='word',stop_words='english'),\
    TfidfVectorizer(analyzer='word',stop_words='english')

X_count_filter_train=count_filter_vec.fit_transform(X_train)
X_count_filter_test=count_filter_vec.transform(X_test)

X_tfidf_filter_train=tfidf_filter_vec.fit_transform(X_train)
X_tfidf_filter_test=tfidf_filter_vec.transform(X_test)

mnb_count_filter=MultinomialNB()
mnb_count_filter.fit(X_count_filter_train,y_train)
print 'The accuracy of classifying 20newsgroups using Naive Bayes\
(CountVectorizer by filtering stopwords):',mnb_count_filter.score\
    (X_count_filter_test,y_test)
y_count_filter_predict=mnb_count_filter.predict(X_count_filter_test)

mnb_tfidf_filter=MultinomialNB()
mnb_tfidf_filter.fit(X_tfidf_filter_train,y_train)
print 'The accuracy of classifying 20newsgroups with Naive Bayes(TfidfVectorizer by filtering stopwords):',mnb_tfidf_filter.score(X_tfidf_filter_test,y_test)
y_tfidf_filter_predict=mnb_tfidf_filter.predict(X_tfidf_filter_test)

from sklearn.metrics import classification_report
print classification_report(y_test,y_count_filter_predict,\
                            target_names=news.target_names)
print classification_report(y_test,y_tfidf_filter_predict,\
                            target_names=news.target_names)