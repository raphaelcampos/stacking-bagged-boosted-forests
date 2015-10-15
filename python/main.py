from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np


from LazyNN_RF import *

X, y = load_svmlight_file("../release/datasets/20ng.svm")

tf_transformer = TfidfTransformer(use_idf=True)

X = tf_transformer.fit_transform(X)

X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.1, random_state=42)



#knn = kNN(n_neighbors=30, weights='distance', algorithm='brute', metric='cosine',n_jobs=8)
#knn.fit(X_train, y_train)

#print "kNN : ", knn.score(X_test, y_test)

#lazy = LazyNNRF(n_jobs=8, n_estimators=200, n_neighbors=60)
#lazy.fit(X_train, y_train)
#print "lazy : ", lazy.score(X_test, y_test)


rf = ForestClassifier(n_estimators=200, n_jobs=8, oob_score=True)
rf.fit(X_train, y_train)
print rf.oob_score_
print rf.oob_decision_function_
print len(rf.oob_decision_function_),len(rf.oob_decision_function_[0])
print "RF : ", rf.score(X_test, y_test)
print rf.predict_proba(X_test)
