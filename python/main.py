from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import train_test_split, cross_val_predict, cross_val_score, KFold, StratifiedKFold
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.base import clone

from sklearn.metrics import f1_score

import numpy as np

from LazyNN_RF import *
from Broof import *

import argparse
import time

parser = argparse.ArgumentParser(description="Random Forest based classifiers.")

parser.add_argument("dataset", type=str,
                    help="SVM light format dataset")

parser.add_argument("-m", "--method", choices=['rf','lazy', 'broof', 'lazy_xt'], default='rf')

parser.add_argument("-H", "--height", type=int, help='trees maximum height. If 0 is given then the trees grow to their maximum depth (default:0)', default=0)

parser.add_argument("-j", "--jobs", type=int, help='Number of CPUs available to parallelize the execution (Default:1). If -1 is given then it gets all CPUs available', default=1)

parser.add_argument("-i", "--ibroof", type=int, help='Number of iteration for broof to perform (Default:100).', default=100)

parser.add_argument("-t", "--trees", type=int, help='Number of trees (Default:100).', default=100)

parser.add_argument("-k", "--kneighbors", type=int, help='Number of nearrest neirghbors (Default:30).', default=30)


parser.add_argument("--trials", type=int, help='Number of trials (Default:10).', default=10)


args = parser.parse_args()

start = time.time()

X, y = load_svmlight_file(args.dataset)
n_train = 7404
X_t, y_t = X[n_train:],y[n_train:]
X, y = X[0:n_train],y[0:n_train]
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
	X_train, X_test = X, X_t
	y_train, y_test = y, y_t
	print len(y_train), len(y_test)
	e = clone(estimator)
	# fit and predict
	start = time.time()
	e.fit(X_train, y_train)
	pred = e.predict(X_test) 
	end = time.time()
	
	# stores fold results
	folds_time = folds_time + [end - start]
	print pred
	result = np.array([range(len(y_test)), pred, y_test])
	print result
	np.savetxt(f, result.transpose(), fmt='%d', header=str(k))
	k = k + 1
	print "\tMicro: ", f1_score(y_true=y_test, y_pred=pred, average='micro')
	print "\tMacro: ", f1_score(y_true=y_test, y_pred=pred, average='macro')
	break

f.close()

print 'times : ', np.average(folds_time), np.std(folds_time) 
