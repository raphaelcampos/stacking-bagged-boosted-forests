from sklearn.datasets import fetch_20newsgroups, load_svmlight_file
from sklearn.cross_validation import train_test_split, cross_val_predict, cross_val_score, KFold, StratifiedKFold
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.base import clone
from sklearn.metrics import f1_score

import numpy as np

from LazyNN_RF import *
from Broof import *

import argparse
import time

parser = argparse.ArgumentParser(description="Random Forest based classifiers.")

parser.add_argument("dataset", type=str,
                    help="SVM light format dataset. If \'toy\' is given then it is used 20ng as a toy example.", default='toy')

parser.add_argument("-m", "--method", choices=['rf','lazy', 'broof', 'lazy_xt'], default='rf')

parser.add_argument("-H", "--height", type=int, help='trees maximum height. If 0 is given then the trees grow to their maximum depth (default:0)', default=0)

parser.add_argument("-j", "--jobs", type=int, help='Number of CPUs available to parallelize the execution (Default:1). If -1 is given then it gets all CPUs available', default=1)

parser.add_argument("-i", "--ibroof", type=int, help='Number of iteration for broof to perform (Default:100).', default=100)

parser.add_argument("-t", "--trees", type=int, help='Number of trees (Default:100).', default=100)

parser.add_argument("-k", "--kneighbors", type=int, help='Number of nearrest neirghbors (Default:30).', default=30)


parser.add_argument("--trials", type=int, help='Number of trials (Default:10).', default=10)

parser.add_argument("--test", action='store_true', help='Executes only one trial. It is used when you want to make a rapid test.')


args = parser.parse_args()

start = time.time()

if args.dataset == 'toy':
	twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
	count_vect = CountVectorizer(min_df=1, stop_words='english')
	X = count_vect.fit_transform(twenty_train.data)
	y = twenty_train.target
else:
	X, y = load_svmlight_file(args.dataset)

tf_transformer = TfidfTransformer(use_idf=True)

X = tf_transformer.fit_transform(X)

end = time.time()

datasetLoadingTime = end - start;

estimator = None
if args.method == 'lazy':
	# TODO: fix bug - lazy is not working properly with 20ng dataset using 5-fold cross validation. k > 5 works - investigate why.
	estimator = LazyNNRF(n_neighbors=args.kneighbors, n_estimators=args.trees, n_jobs=args.jobs, max_features=0.3)
elif args.method == 'lazy_xt':
	estimator = LazyNNExtraTrees(n_neighbors=args.kneighbors, n_estimators=args.trees, n_jobs=args.jobs)
elif args.method == 'adarf':
	estimator = AdaBoostClassifier(base_estimator=ForestClassifier(n_estimators=args.trees, n_jobs=args.jobs, oob_score=True),n_estimators=args.ibroof, n_jobs=args.jobs)
elif args.method == 'broof':
	estimator = Broof(n_estimators=args.ibroof, n_jobs=args.jobs, n_trees=args.trees)
else:
	estimator = ForestClassifier(n_estimators=args.trees, n_jobs=args.jobs)

kf = StratifiedKFold(y, n_folds=args.trials, shuffle=True, random_state=42)

folds_time = []
folds_predict = []

f=open('results_20ng_lazy0.30','ab')
k = 1
for train_index, test_index in kf:
	# split dataset
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	
	e = clone(estimator)
	# fit and predict
	start = time.time()
	e.fit(X_train, y_train)
	pred = e.predict(X_test) 
	end = time.time()
	
	# stores fold results
	folds_time = folds_time + [end - start]
	result = np.array([range(len(y_test)), pred, y_test])
	np.savetxt(f, result.transpose(), fmt='%d', header=str(k))
	k = k + 1
	print "\tMicro: ", f1_score(y_true=y_test, y_pred=pred, average='micro')
	print "\tMacro: ", f1_score(y_true=y_test, y_pred=pred, average='macro')
	if args.test:
		break

f.close()
print 'loading time : ', 
print 'times : ', np.average(folds_time), np.std(folds_time) 
