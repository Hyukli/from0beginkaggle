sent1='The cat is walking in the bedroom.'
sent2='A dog was running across the kitchen.'
from sklearn.feature_extraction.text import CountVectorizer
count_vec=CountVectorizer()

sentences=[sent1,sent2]
print count_vec.fit_transform(sentences).toarray()
print count_vec.get_feature_names()