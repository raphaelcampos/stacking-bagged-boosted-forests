from sklearn.datasets import fetch_20newsgroups, load_svmlight_file
from sklearn.cross_validation import train_test_split, cross_val_predict, cross_val_score, KFold, StratifiedKFold
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

from sklearn.cluster import  KMeans as kmeans

from sklearn.linear_model import *

from sklearn.grid_search import GridSearchCV

from sklearn.base import clone
from sklearn.metrics import f1_score

import numpy as np

from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC, LinearSVC

from LazyNN_RF import *
from Broof import *

from stacking import *

from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

import argparse
import time

def instantiate_estimator(method, args):
	random_state = 123
	if method == 'lazy':
		return LazyNNRF(n_neighbors=args.kneighbors, n_estimators=args.trees,
		 			n_jobs=args.jobs, max_features='auto', criterion='gini',
			 		n_gpus=args.gpus, random_state=random_state)
	elif method == 'lazy_xt':
		return LazyNNExtraTrees(n_neighbors=args.kneighbors, n_estimators=args.trees, n_jobs=args.jobs,
			max_features=args.max_features, criterion='gini', n_gpus=args.gpus)
	elif method == 'lazy_broof':
		return LazyNNBroof(n_iterations=args.ibroof, learning_rate=args.learning_rate,
			n_neighbors=args.kneighbors, n_estimators=args.trees, n_jobs=args.jobs,
			max_features=args.max_features, criterion='gini', n_gpus=args.gpus)
	elif method == 'adarf':
		return AdaBoostClassifier(base_estimator=ForestClassifier(n_estimators=args.trees, n_jobs=args.jobs,
		 max_features=args.max_features),n_estimators=args.ibroof)
	elif method == 'broof':
		return Broof(n_iterations=args.ibroof, n_jobs=args.jobs, n_trees=args.trees, learning_rate=args.learning_rate, max_features=args.max_features)
	elif method == 'bert':
		return Bert(n_iterations=args.ibroof, n_jobs=args.jobs, n_trees=args.trees, learning_rate=args.learning_rate, max_features=args.max_features)
	elif method == 'xt':
		return ExtraTreesClassifier(n_estimators=args.trees, n_jobs=args.jobs, criterion='gini', max_features=args.max_features, verbose=0)
	elif method == 'svm':
		return LinearSVC(C=args.c, dual=True, tol=1e-03)
	elif method == 'nb':
		return MultinomialNB(alpha=args.alpha)
	elif method == 'knn':
		return KNeighborsClassifier(n_neighbors=args.kneighbors, algorithm='brute', weights='distance', metric='cosine', n_jobs=args.jobs)
	else:
		return RandomForestClassifier(n_estimators=args.trees, n_jobs=args.jobs, criterion='gini', max_features=args.max_features, verbose=0)

models = ['rf','lazy', 'adarf', 'broof', 'bert', 'lazy_xt', 'xt',
				 'comb1', 'comb2', 'comb3', 'comb4', 'lazy_broof']

parser = argparse.ArgumentParser(description="Random Forest based classifiers.")

parser.add_argument("dataset", type=str,
                    help="SVM light format dataset. If \'toy\' is given then it is used 20ng as a toy example.", default='toy')

parser.add_argument("-m", "--method", choices=models, default=models[0])

parser.add_argument("-H", "--height", type=int,
 help='trees maximum height. If 0 is given then the trees grow to their maximum depth (default:0)', default=0)

parser.add_argument("-j", "--jobs", type=int, help='Number of CPUs available to parallelize the execution (Default:1). If -1 is given then it gets all CPUs available', default=1)

parser.add_argument("-g","--gpus", type=int, help='Number of GPUs available to execute kNN-based classifiers (Default:1).', default=1)

parser.add_argument("-i", "--ibroof", type=int, help='Number of iteration for broof to perform (Default:100).', default=100)

parser.add_argument("-t", "--trees", type=int, help='Number of trees (Default:100).', default=100)

parser.add_argument("-k", "--kneighbors", type=int, help='Number of nearest neirghbors (Default:30).', default=30)

parser.add_argument("-f", "--max_features", help='Number of nearrest neirghbors (Default:sqrt).', default='sqrt')


parser.add_argument("--learning_rate", type=float, help='Algorithm learning rate. It controls algorithm\'s convergence.', default=1.0)

parser.add_argument("--trials", type=int, help='Number of trials (Default:10).', default=10)

parser.add_argument("--cv", type=int, help='Search for best parameters using cross-validation (Default:1). When 1 is given there will not be search at all.', default=1)

parser.add_argument("--start_fold", type=int, default=1)

parser.add_argument("--test", action='store_true', help='Executes only one trial. It is used when you want to make a rapid test.')

parser.add_argument("-o", "--output", type=str, help="Output file for results.", default="")

args = parser.parse_args()

start = time.time()

try:
	args.max_features = float(args.max_features)
except Exception, e:
	pass
	
if args.dataset == 'toy':
	twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
	count_vect = CountVectorizer(min_df=6, stop_words='english')
	X = count_vect.fit_transform(twenty_train.data)
	y = twenty_train.target
else:
	X, y = load_svmlight_file(args.dataset)


print X.nnz, (X.nnz*4*4)/(2.0**20)

end = time.time()

datasetLoadingTime = end - start;

estimator = None
if args.method == 'lazy':
	estimator = instantiate_estimator(args.method, args)
	tuned_parameters = [{'n_neighbors': [30,100,500],
							 'n_estimators': [50, 100, 200, 400]}]
elif args.method == 'lazy_xt':
	estimator = instantiate_estimator(args.method, args)
	tuned_parameters = [{'n_neighbors': [30,100,500], 'n_estimators': [50, 100, 200, 400]}]
elif args.method == 'lazy_broof':
	estimator = instantiate_estimator(args.method, args)
	tuned_parameters = [{'n_neighbors': [30,100,500], 'n_estimators': [50, 100, 200, 400]}]
elif args.method == 'adarf':
	estimator = instantiate_estimator(args.method, args)
	tuned_parameters = [{'n_estimators': [50, 100, 200, 400, 600]}]
elif args.method == 'broof':
	estimator = instantiate_estimator(args.method, args)
	tuned_parameters = [{'n_trees': [5], 'n_estimators': [10, 20, 50, 100, 200], 'learning_rate': [0.1, 0.5, 1.0]},{'n_trees': [10, 30, 50], 'n_estimators': [10, 20, 30], 'learning_rate': [0.1, 0.5, 1.0]}]
elif args.method == 'bert':
	estimator = instantiate_estimator(args.method, args)
	tuned_parameters = [{'n_trees': [5], 'n_estimators': [10, 20, 50, 100, 200], 'learning_rate': [0.1, 0.5, 1.0]},{'n_trees': [10, 30, 50], 'n_estimators': [10, 20, 30], 'learning_rate': [0.1, 0.5, 1.0]}]
elif args.method == 'xt':
	estimator = instantiate_estimator(args.method, args)
	tuned_parameters = [{'n_estimators': [200], 'criterion':['gini', 'entropy'], 'max_features': ['sqrt', 'log2', 0.03, 0.08, 0.15, 0.3]}]
elif args.method == 'comb1':
	estimators_stack = list()
	estimators_stack.append(
		[Broof(n_iterations=args.ibroof, n_jobs=args.jobs, n_trees=5,
		 learning_rate=args.learning_rate, max_features=args.max_features),
		 LazyNNRF(n_neighbors=args.kneighbors, n_estimators=args.trees, n_jobs=args.jobs,
		  max_features='auto', criterion='gini', n_gpus=args.gpus)])
	estimators_stack.append(ForestClassifier(n_estimators=args.trees, n_jobs=args.jobs, criterion='gini', max_features=args.max_features, verbose=0))
	#estimators_stack.append(RidgeClassifierCV(cv=5))
	estimator = StackingClassifier(estimators_stack)
elif args.method == 'comb2':
	estimators_stack = list()
	estimators_stack.append(
		[Bert(n_iterations=args.ibroof, n_jobs=args.jobs, n_trees=5,
		 learning_rate=args.learning_rate, max_features=args.max_features),
		 LazyNNExtraTrees(n_neighbors=args.kneighbors, n_estimators=args.trees, n_jobs=args.jobs,
		  max_features='auto', criterion='gini', n_gpus=args.gpus)])
	estimators_stack.append(ForestClassifier(n_estimators=args.trees, n_jobs=args.jobs, criterion='gini', max_features=args.max_features, verbose=0))
	#estimators_stack.append(RidgeClassifierCV(cv=5))
	estimator = StackingClassifier(estimators_stack)
elif args.method == 'comb3':
	estimators_stack = list()
	# Level 0 classifiers
	estimators_stack.append(
		[Broof(n_iterations=args.ibroof, n_jobs=args.jobs, n_trees=5,
			 learning_rate=args.learning_rate, max_features=args.max_features),
		 LazyNNRF(n_neighbors=args.kneighbors, n_estimators=args.trees,
		 							 n_jobs=args.jobs, max_features='auto', 
		 							 criterion='gini', n_gpus=args.gpus),
		 Bert(n_iterations=args.ibroof, n_jobs=args.jobs, n_trees=5,
		 								 learning_rate=args.learning_rate,
		 								 max_features=args.max_features),
		 LazyNNExtraTrees(n_neighbors=args.kneighbors, n_estimators=args.trees,
		 							 n_jobs=args.jobs, max_features='auto',
		 							 criterion='gini', n_gpus=args.gpus)])
	# Level 1 classifier (Aggregator)
	estimators_stack.append(ForestClassifier(n_estimators=args.trees, n_jobs=args.jobs, criterion='gini', max_features=args.max_features, verbose=0))
	#estimators_stack.append(RidgeClassifierCV(cv=5))
	estimator = StackingClassifier(estimators_stack)
elif args.method == 'comb4':
	estimators_stack = list()
	# Level 0 classifiers
	estimators_stack.append(
		[SVC(C=1, kernel='linear', probability=True),
		 KNeighborsClassifier(n_neighbors=args.kneighbors, algorithm='brute',
							 weights='distance', metric='cosine', n_jobs=args.jobs),
		 RandomForestClassifier(n_estimators=args.trees, n_jobs=args.jobs,
		 			 criterion='gini', max_features=args.max_features, verbose=10),
		 Broof(n_iterations=args.ibroof, n_jobs=args.jobs, n_trees=5,
			 learning_rate=args.learning_rate, max_features=args.max_features),
		 LazyNNRF(n_neighbors=args.kneighbors, n_estimators=args.trees,
		 							 n_jobs=args.jobs, max_features='auto', 
		 							 criterion='gini', n_gpus=args.gpus),
		 Bert(n_iterations=args.ibroof, n_jobs=args.jobs, n_trees=5,
		 								 learning_rate=args.learning_rate,
		 								 max_features=args.max_features),
		 LazyNNExtraTrees(n_neighbors=args.kneighbors, n_estimators=args.trees,
		 							 n_jobs=args.jobs, max_features='auto',
		 							 criterion='gini', n_gpus=args.gpus)])
	# Level 1 classifier (Aggregator)
	estimators_stack.append(ForestClassifier(n_estimators=args.trees, n_jobs=args.jobs, criterion='gini', max_features=args.max_features, verbose=0))
	#estimators_stack.append(RidgeClassifierCV(cv=5))
	estimator = StackingClassifier(estimators_stack)
else:
	estimator = ForestClassifier(n_estimators=args.trees, n_jobs=args.jobs, criterion='gini', max_features=args.max_features, verbose=10)
	tuned_parameters = [{'n_estimators': [200], 'criterion':['gini', 'entropy'], 'max_features': ['sqrt', 'log2', 0.03, 0.08, 0.15, 0.3]}]

kf = StratifiedKFold(y, n_folds=args.trials, shuffle=True, random_state=42)

folds_time = []
folds_predict = []
folds_macro = []
folds_micro = []

print estimator.get_params(deep=False)

k = 1
for train_index, test_index in kf:
	
	if(k < args.start_fold):
		k = k + 1
		continue 

	# split dataset
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	
	tf_transformer = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True)
	if args.gpus <= 0:
		# Learn the idf vector from training set
		tf_transformer.fit(X_train)
		# Transform test and training frequency matrix
		# based on training idf vector		
		X_train = tf_transformer.transform(X_train)
		X_test = tf_transformer.transform(X_test)

	if(args.cv > 1):
		gs = GridSearchCV(estimator, tuned_parameters,
									 cv=args.cv, verbose=10, scoring='f1_macro')
		gs.fit(X_train, y_train)
		print gs.best_score_, gs.best_params_
		estimator = gs.best_estimator_

	e = clone(estimator)
	
	# fit and predict
	start = time.time()
	e.fit(X_train, y_train)
	pred = e.predict(X_test) 
	end = time.time()
	
	# force to free memory
	del e

	# stores fold results
	folds_time = folds_time + [end - start]
	result = np.array([range(len(y_test)), y_test, pred])
	
	if not (args.output == "") :
		f=open(args.output,'ab')
		np.savetxt(f, result.transpose(), fmt='%d', header=str(k))
		f.close()

	k = k + 1
	folds_micro = folds_micro + [f1_score(y_true=y_test, y_pred=pred, average='micro')]
	folds_macro = folds_macro + [f1_score(y_true=y_test, y_pred=pred, average='macro')]

	print "F1-Score"
	print "\tMicro: ", folds_micro[-1]
	print "\tMacro: ", folds_macro[-1]
	
	if args.test:
		break

print "F1-Score"
print "\tMicro: ", np.average(folds_micro), np.std(folds_micro)
print "\tMacro: ", np.average(folds_macro), np.std(folds_macro)

print 'loading time : ', datasetLoadingTime
print 'times : ', np.average(folds_time), np.std(folds_time) 
