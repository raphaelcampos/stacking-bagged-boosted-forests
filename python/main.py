from sklearn.datasets import fetch_20newsgroups, load_svmlight_file
from sklearn.cross_validation import train_test_split, cross_val_predict, cross_val_score, KFold, StratifiedKFold
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer

from sklearn.neighbors import NearestNeighbors as kNN

from sklearn.grid_search import GridSearchCV

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

parser.add_argument("-g","--gpus", type=int, help='Number of GPUs available to execute kNN-based classifiers (Default:1).', default=1)

parser.add_argument("-i", "--ibroof", type=int, help='Number of iteration for broof to perform (Default:100).', default=100)

parser.add_argument("-t", "--trees", type=int, help='Number of trees (Default:100).', default=100)

parser.add_argument("-k", "--kneighbors", type=int, help='Number of nearrest neirghbors (Default:30).', default=30)

parser.add_argument("--learning_rate", type=float, help='Algorithm learning rate. It controls algorithm\'s convergence.', default=1.0)

parser.add_argument("--trials", type=int, help='Number of trials (Default:10).', default=10)

parser.add_argument("--cv", type=int, help='Search for best parameters using cross-validation (Default:1). When 1 is given there will not be search at all.', default=1)

parser.add_argument("--test", action='store_true', help='Executes only one trial. It is used when you want to make a rapid test.')

parser.add_argument("-o", "--output", type=str, help="Output file for results.", default="")

args = parser.parse_args()

start = time.time()

if args.dataset == 'toy':
	twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
	count_vect = CountVectorizer(min_df=6, stop_words='english')
	X = count_vect.fit_transform(twenty_train.data)
	y = twenty_train.target
else:
	X, y = load_svmlight_file(args.dataset)


print X.nnz, (X.nnz*4*4)/(2.0**20)


tf_transformer = TfidfTransformer(use_idf=True)

#X = tf_transformer.fit_transform(X)

end = time.time()

datasetLoadingTime = end - start;

estimator = None
if args.method == 'lazy':
	# TODO: fix bug - lazy is not working properly with 20ng dataset using 5-fold cross validation. k > 5 works - investigate why.
	estimator = LazyNNRF(n_neighbors=args.kneighbors, n_estimators=args.trees, n_jobs=args.jobs, max_features='auto', criterion='gini', cuda=True, n_gpus=args.gpus)
	tuned_parameters = [{'n_neighbors': [30,100,500], 'n_estimators': [50, 100, 200, 400]}]

elif args.method == 'lazy_xt':
	estimator = LazyNNExtraTrees(n_neighbors=args.kneighbors, n_estimators=args.trees, n_jobs=args.jobs,
		max_features='auto', criterion='gini', cuda=True, n_gpus=args.gpus)
	tuned_parameters = [{'n_neighbors': [30,100,500], 'n_estimators': [50, 100, 200, 400]}]
elif args.method == 'adarf':
	estimator = AdaBoostClassifier(base_estimator=ForestClassifier(n_estimators=args.trees, n_jobs=args.jobs),n_estimators=args.ibroof, n_jobs=args.jobs)
	tuned_parameters = [{'n_estimators': [50, 100, 200, 400, 600]}]
elif args.method == 'broof':
	estimator = Broof(n_estimators=args.ibroof, n_jobs=args.jobs, n_trees=args.trees, learning_rate=args.learning_rate)
	tuned_parameters = [{'n_trees': [5], 'n_estimators': [10, 20, 50, 100, 200], 'learning_rate': [0.1, 0.5, 1.0]},{'n_trees': [10, 30, 50], 'n_estimators': [10, 20, 30], 'learning_rate': [0.1, 0.5, 1.0]}]
else:
	estimator = ForestClassifier(n_estimators=args.trees, n_jobs=args.jobs, criterion='gini')
	tuned_parameters = [{'n_estimators': [50, 100, 200, 400], 'criterion':['gini', 'entropy']}]

kf = StratifiedKFold(y, n_folds=args.trials, shuffle=True, random_state=42)

folds_time = []
folds_predict = []
folds_macro = []
folds_micro = []

print estimator.get_params()

if not (args.output == "") :
	f=open(args.output,'ab')

k = 1
for train_index, test_index in kf:
	# split dataset
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	

	if(args.cv > 1):
		gs = GridSearchCV(estimator, tuned_parameters, cv=args.cv, verbose=10, scoring='f1_micro')
		gs.fit(X_train, y_train)
		print gs.best_score_, gs.best_params_
		estimator = gs.best_estimator_
	
	

	#e = cuKNeighborsSparseClassifier(n_neighbors=30)

	#e.fit(X_train, y_train)	

	#pred = e.predict(X_test)
	#print (pred.T[0])
	#print np.mean(pred.T[0] == y_test)
	#exit()

	e = clone(estimator)
	
	# fit and predict
	start = time.time()
	e.fit(X_train, y_train)
	pred = e.predict(X_test) 
	end = time.time()
	
	# stores fold results
	folds_time = folds_time + [end - start]
	result = np.array([range(len(y_test)), pred, y_test])
	if not (args.output == "") :
		np.savetxt(f, result.transpose(), fmt='%d', header=str(k))
	k = k + 1
	folds_micro = folds_micro + [f1_score(y_true=y_test, y_pred=pred, average='micro')]
	folds_macro = folds_macro + [f1_score(y_true=y_test, y_pred=pred, average='macro')]

	print "F1-Score"
	print "\tMicro: ", folds_micro[k-2]
	print "\tMacro: ", folds_macro[k-2]
	if args.test:
		break

if not (args.output == "") :
	f.close()

print "F1-Score"
print "\tMicro: ", np.average(folds_micro), np.std(folds_micro)
print "\tMacro: ", np.average(folds_macro), np.std(folds_macro)

print 'loading time : ', datasetLoadingTime
print 'times : ', np.average(folds_time), np.std(folds_time) 
