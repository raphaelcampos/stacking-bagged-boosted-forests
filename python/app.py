from sklearn.datasets import fetch_20newsgroups, load_svmlight_file, dump_svmlight_file
from sklearn.cross_validation import KFold, StratifiedKFold

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.utils import check_random_state

from sklearn.grid_search import GridSearchCV

from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from instantiator import EstimatorInstantiator
from config import default_tuning_params, default_transformers

from xsklearn.ensemble import StackingClassifier

import numpy as np
import argparse
import time

MAX_INT = np.iinfo(np.int32).max

class BaseApp(object):
	"""Application base class

    Parameters
    ----------
    description : str, optional (default="Application description")
        Application description.
    
    Attributes
    ----------
    description : str
        Application description.

    parser : argparse
    	Command line argument parser object
    """
	def __init__(self, description="Application description"):
		super(BaseApp, self).__init__()

		self.description = description
		self._init_parser()

	def _init_parser(self):
		self.parser = argparse.ArgumentParser(description=self.description) 

	def get_argparse(self):
		return self.parser
		
class ClassificationApp(BaseApp):
	"""Classification application base class

    Parameters
    ----------
    description : str, optional (default="Classification application")
        Application description.
    
    Attributes
    ----------
    instantiator : EstimatorInstantiator
    	The estimator instantiator responsable to create specific estimator object

    description : str
        Application description.

    parser : argparse
    	Command line argument parser object.
    """
	def __init__(self, description="Classification application"):
		self.instantiator = EstimatorInstantiator()
		
		super(ClassificationApp, self).__init__(description=description)
		

	def _init_parser(self):
		super(ClassificationApp, self)._init_parser()
		
		models = self.instantiator.get_estimators()
		
		self.parser.add_argument("dataset", type=str,
                    help="SVM light format dataset. If \'toy\' is given then it is used 20ng as a toy example.", default='toy')

		self.parser.add_argument("-m", "--method", choices=models, default=models[0])

		self.parser.add_argument("-j", "--n_jobs", type=int, help='Number of CPUs available to parallelize the execution (Default:1). If -1 is given then it gets all CPUs available', default=1)

		self.parser.add_argument("--trials", type=int,
							 help='Number of trials (Default:10).', default=10)

		self.parser.add_argument("--cv", type=int, help='Search for best parameters using grid search cross-validation (Default:1). When 1 is given there will not be search at all.', default=1)

		self.parser.add_argument("--start_fold", type=int, help='Starting trial (default: 1).',
			default=1)

		self.parser.add_argument("-s", "--seed", type=int, help='Random seed initializing the pseudo-random number generator (default: 42).', default=42)

		self.parser.add_argument("--test", action='store_true', help='Executes only one trial. It is used when you want to make a rapid test.')


		self.parser.add_argument("-o", "--output", type=str, help="Output file for results.", default="")

	def parse_arguments(self):
		return self.parser.parse_args()

	def _load_dataset(self, args):
		start = time.time()

		if args.dataset == 'toy':
			twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
			count_vect = CountVectorizer(min_df=6, stop_words='english')
			X = count_vect.fit_transform(twenty_train.data)
			y = twenty_train.target
		else:
			X, y = load_svmlight_file(args.dataset)


		print X.nnz, (X.nnz*4*4)/(2.0**20)

		end = time.time()

		self.datasetLoadingTime = end - start;

		return X, y

	def _setup_instantiator(self, args):
		random_instance = check_random_state(args.seed)
		self.instantiator.set_general_params(vars(args))
		self.instantiator.set_general_params(
						{'random_state': random_instance.randint(0,MAX_INT,1)})

		self.instantiator.set_params(estimators_params)

		return (self.instantiator.get_instance(args.method),
				 default_tuning_params[args.method])

	def _tfidf(self, args):
		return False

	def run(self, args):
		X, y = self._load_dataset(args)

		kf = StratifiedKFold(y, n_folds=args.trials, shuffle=True,
												 random_state=args.seed)

		estimator, tuned_parameters = self._setup_instantiator(args)

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
			
			tf_transformer = TfidfTransformer(norm='max', use_idf=True, smooth_idf=True, sublinear_tf=True)
			if self._tfidf(args):
				# Learn the idf vector from training set
				tf_transformer.fit(X_train)
				# Transform test and training frequency matrix
				# based on training idf vector		
				X_train = tf_transformer.transform(X_train)
				X_test = tf_transformer.transform(X_test)

			#dump_svmlight_file(X_train,y_train,"4uni/stratified/treino%d_orig" % (k - 1), zero_based=False)
			#dump_svmlight_file(X_test,y_test,"4uni/stratified/teste%d_orig" % (k - 1), zero_based=False)
			#k = k + 1
			#continue

			if(args.cv > 1):
				n_jobs = 1 if hasattr(estimator,"n_jobs") else args.n_jobs
				gs = GridSearchCV(estimator, tuned_parameters,
							 n_jobs=n_jobs, refit=False,
							 cv=args.cv, verbose=1, scoring='f1_micro')
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

		print 'loading time : ', self.datasetLoadingTime
		print 'times : ', np.average(folds_time), np.std(folds_time)

class TextClassificationApp(ClassificationApp):
	"""Application for text classification
    
    Attributes
    ----------
    instantiator : EstimatorInstantiator
    	The estimator instantiator responsable to create specific estimator object

    description : str
        Application description.

    parser : argparse
    	Command line argument parser object.
    """
	def __init__(self):
		super(TextClassificationApp, self).__init__(
								description="Text classification classifiers")
		
		self.instantiator = EstimatorInstantiator()
		self._init_parser()

	def _init_parser(self):
		super(TextClassificationApp, self)._init_parser()

		self.parser.add_argument("-g","--n_gpus", type=int, help='Number of GPUs available to execute kNN-based classifiers (Default:1).', default=1)

		self.parser.add_argument("-i", "--n_iterations", type=int, help='Number of iteration for broof to perform (Default:200).', default=200)

		self.parser.add_argument("-t", "--trees", type=int, help='Number of trees (Default:200).', default=200)

		self.parser.add_argument("-k", "--n_neighbors", type=int, help='Number of nearest neirghbors (Default:30).', default=30)

		self.parser.add_argument("-f", "--max_features", help='The number of features to consider when looking for the best split (Default:sqrt).', default='sqrt')

		self.parser.add_argument("--learning_rate", type=float, help='Algorithm learning rate. It controls algorithm\'s convergence.', default=1.0)

		self.parser.add_argument("-a","--alpha", type=float, help='Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).(Default: 1)', default=1)

		self.parser.add_argument("-c", "--C", type=float, help='Penalty parameter C of the error term. For SVM training. (Default: 1)', default=1)

	def _setup_instantiator(self, args):
		random_instance = check_random_state(args.seed)

		self.instantiator.set_general_params(vars(args))
		self.instantiator.set_general_params(
						{'random_state': random_instance.randint(MAX_INT)})

		estimators_params = {
			'rf':	{'n_estimators': args.trees},
			'xt': 	{'n_estimators': args.trees},
			'lazy':	{'n_estimators': args.trees},
			'lxt': 	{'n_estimators': args.trees},
			'broof':{'n_trees': args.trees},
			'bert': {'n_trees': args.trees}
		}

		self.instantiator.set_params(estimators_params)

		return (self.instantiator.get_instance(args.method),
				 default_tuning_params[args.method])

	def _tfidf(self, args):
		return args.n_gpus <= 0

	def parse_arguments(self):
		args = super(TextClassificationApp, self).parse_arguments()

		try:
			args.max_features = float(args.max_features)
		except Exception, e:
			pass

		return args 
import json
class StackerApp(TextClassificationApp):
	"""Application for text classification using stacking of classifiers
    
    Attributes
    ----------
    instantiator : EstimatorInstantiator
    	The estimator instantiator responsable to create specific estimator object

    description : str
        Application description.

    parser : argparse
    	Command line argument parser object.
    """
	def __init__(self):
		super(TextClassificationApp, self).__init__(
								description="Text classification classifiers with stacking")
		
		self.instantiator = EstimatorInstantiator()
		self._init_parser()

	def _init_parser(self):
		super(StackerApp, self)._init_parser()
		self.parser.add_argument("base_classifiers", type=str, help='Base-level classifier.')

		self.parser.add_argument("--base_params", type=str, help='Base-level classifier.', default="[]")

		self.parser.add_argument("meta_classifiers", type=str, help='Base-level classifier.')

		self.parser.add_argument("--meta_params", type=str, help='Base-level classifier.', default="[]")

	def _setup_instantiator(self, args):
		
		estimator, tuned_parameters = super(StackerApp, self)._setup_instantiator(
																			args)
		self.instantiator.set_general_params({'probability': True})
		return self._create_stacking(args), tuned_parameters


	def _create_stacking(self, args):
		random_instance = check_random_state(args.seed)


		params = json.loads(args.base_params)
		base_classifiers = json.loads(args.base_classifiers)

		if params == [] or params is None:
			params = [{} for i in range(len(base_classifiers))]

		base_level = [Pipeline(default_transformers[estimator] + [('estimator', self.instantiator.get_instance(estimator, params[i]))]) for i, estimator in enumerate(base_classifiers)]

		params = json.loads(args.meta_params)
		meta_classifiers = json.loads(args.meta_classifiers)
		
		if params == [] or params is None:
			params = [{} for i in range(len(base_classifiers))]

		meta_level = [self.instantiator.get_instance(estimator, params[i]) for i, estimator in enumerate(meta_classifiers)]
		
		stack = list()
		stack.append(base_level)
		stack.append(meta_level[0])

		return StackingClassifier(estimators_stack=stack,
						 random_state=random_instance.randint(MAX_INT))


	def _tfidf(self, args):
		return False

if __name__ == '__main__':
	app = StackerApp()

	app.run(app.parse_arguments())
