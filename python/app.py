from sklearn.datasets import load_iris, fetch_20newsgroups, load_svmlight_file, dump_svmlight_file
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
import json

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
		self.parser = argparse.ArgumentParser(description=self.description)

	def _init_parser(self):
		pass  

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
	def __init__(self, description="Classification application", default_tuning_params=default_tuning_params.copy()):
		super(ClassificationApp, self).__init__(description=description)
		
		self.default_tuning_params = default_tuning_params
		self.instantiator = EstimatorInstantiator()
		self._init_parser()

	def _init_parser(self):
		#super(ClassificationApp, self)._init_parser()
		models = self.instantiator.get_estimators()
		
		self.parser.add_argument("dataset", type=str,
                    help="SVM light format dataset. If \'toy\' is given then it is used 20ng as a toy example.", default='toy')

		self.parser.add_argument("-m", "--method", choices=models, default=list(models)[0])

		self.parser.add_argument("-j", "--n_jobs", type=int, help='Number of CPUs available to parallelize the execution (Default:1). If -1 is given then it gets all CPUs available', default=1)

		self.parser.add_argument("--trials", type=int,
							 help='Number of trials (Default:10).', default=10)

		self.parser.add_argument("--cv", type=int, help='Search for best parameters using grid search cross-validation (Default:1). When 1 is given there will not be search at all.', default=1)

		self.parser.add_argument("--start_fold", type=int, help='Starting trial (default: 1).',
			default=1)

		self.parser.add_argument("-s", "--seed", type=int, help='Random seed initializing the pseudo-random number generator (default: 42).', default=42)

		self.parser.add_argument("--test", action='store_true', help='Executes only one trial. It is used when you want to make a rapid test.')


		self.parser.add_argument("-o", "--output", type=str, help="Output file for results.", default="")

		self.parser.add_argument("-n", "--norm", help='Dataset sample normalization', choices=["max", "l1", "l2"], default=None)

		self.parser.add_argument("-d", "--dump", type=str, help='Save each fold stacking.',default="")
		self.parser.add_argument("--dump_meta_level", type=str, help='Save each fold stacking.',default="")


	def parse_arguments(self):
		return self.parser.parse_args()

	def _load_dataset(self, args):
		start = time.time()

		if args.dataset == 'toy':
			twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
			count_vect = CountVectorizer(min_df=3, stop_words='english')
			X = count_vect.fit_transform(twenty_train.data)
			y = twenty_train.target
		else:			
			X, y = load_svmlight_file(args.dataset)

		end = time.time()

		self.datasetLoadingTime = end - start;

		return X, y


	def _setup_instantiator(self, args):
		random_instance = check_random_state(args.seed)
		self.instantiator.set_general_params(vars(args))
		self.instantiator.set_general_params(
						{'random_state': random_instance.randint(0,MAX_INT)})
		
		#self.instantiator.set_params(estimators_params)

		return (self.instantiator.get_instance(args.method),
				 self.default_tuning_params[args.method])

	def _tfidf(self, args):
		return False

	def _evaluate_dump(self, k, y_test, pred, args):
		result = np.array([range(len(y_test)), y_test, pred])
			
		if not (args.output == "") :
			f=open(args.output,'ab')
			np.savetxt(f, result.transpose(), fmt='%d', header=str(k))
			f.close()

		self.folds_micro = self.folds_micro + [f1_score(y_true=y_test, y_pred=pred, average='micro')]
		self.folds_macro = self.folds_macro + [f1_score(y_true=y_test, y_pred=pred, average='macro')]

		print("F1-Score")
		print("\tMicro: ", self.folds_micro[-1])
		print("\tMacro: ", self.folds_macro[-1])

	def run(self, args):
		X, y = self._load_dataset(args)
		print(X.shape)
		kf = StratifiedKFold(y, n_folds=args.trials, shuffle=True,
												 random_state=args.seed)

		#kf = KFold(len(y), n_folds=args.trials,
		#					 shuffle=True, random_state=args.seed)


		estimator, tuned_parameters = self._setup_instantiator(args)

		folds_time = []
		self.folds_predict = []
		self.folds_macro = []
		self.folds_micro = []

		print(estimator.get_params(deep=False))

		k = 1
		for train_index, test_index in kf:
			
			if(k < args.start_fold):
				k = k + 1
				continue 

			# split dataset
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			
			tf_transformer = TfidfTransformer(norm=args.norm, use_idf=True,
											 smooth_idf=True, sublinear_tf=True)
			if self._tfidf(args):
				# Learn the idf vector from training set
				tf_transformer.fit(X_train)
				# Transform test and training frequency matrix
				# based on training idf vector		
				X_train = tf_transformer.transform(X_train)
				X_test = tf_transformer.transform(X_test)

			if(args.cv > 1):
				n_jobs = 1 if hasattr(estimator, "n_jobs") else args.n_jobs
				gs =  GridSearchCV(estimator, tuned_parameters,
							 n_jobs=n_jobs, refit=False,
							 cv=args.cv, verbose=1, scoring='f1_micro')
				gs.fit(X_train, y_train)
				print(gs.best_score_, gs.best_params_)
				estimator.set_params(**gs.best_params_)
				
			e = clone(estimator)
				
			# fit and predict
			start = time.time()
			e.fit(X_train, y_train)
			if True and hasattr(e, 'staged_predict'):
				ada_discrete_err = np.zeros((args.n_iterations,))
				for i, y_pred in enumerate(e.staged_predict(X_test)):
					ada_discrete_err[i] = np.mean(y_pred == y_test)
				print(ada_discrete_err)	
				for i, y_pred in enumerate(e.staged_predict(X_test)):
					ada_discrete_err[i] = f1_score(y_true=y_test, y_pred=y_pred, average='macro')
				print(ada_discrete_err)
				for i, y_pred in enumerate(e.oob_staged_decision_function()):
					(oob_ids, ) = np.where(y_pred.sum(1) > 0)
					y_pred = np.argmax(y_pred[oob_ids], axis=1)
					ada_discrete_err[i] = f1_score(y_true=y_train[oob_ids], y_pred=y_pred, average='micro')
				print(ada_discrete_err)
				for i, y_pred in enumerate(e.oob_staged_decision_function()):
					(oob_ids, ) = np.where(y_pred.sum(1) > 0)
					y_pred = np.argmax(y_pred[oob_ids], axis=1)
					ada_discrete_err[i] = f1_score(y_true=y_train[oob_ids], y_pred=y_pred, average='macro')
				print(ada_discrete_err)
				#e.prune(y_train)

			if args.dump_meta_level != "":
				if hasattr(e, 'oob_decision_function_'):
					# missing values
					e.oob_decision_function_[np.isnan(e.oob_decision_function_)] = 0
					print(e.oob_decision_function_[np.isnan(e.oob_decision_function_)])
					norm = e.oob_decision_function_.sum(1)
					print("Missing instances: %f" % (np.sum(norm == 0)/float(norm.shape[0])))
					norm[norm == 0] = 1.
					X_oob = e.oob_decision_function_/norm[:,np.newaxis]
					dump_svmlight_file(X_oob, y_train, args.dump_meta_level % ("train", args.method, k))
					#dump_svmlight_file(e.predict_proba(X_train), y_train, args.dump_meta_level % ("train", args.method, k))
					dump_svmlight_file(e.predict_proba(X_test), y_test, args.dump_meta_level % ("test", args.method, k))
				else:
					from xsklearn.ensemble import MetaLevelTransformerCV
					mt = MetaLevelTransformerCV([clone(estimator)], fit_whole_data=False)
					dump_svmlight_file(mt.fit_transform(X_train, y_train), y_train, args.dump_meta_level % ("train", args.method, k))
					dump_svmlight_file(e.predict_proba(X_test), y_test, args.dump_meta_level % ("test", args.method, k))

			pred = e.predict(X_test) 
			end = time.time()

			import pickle
			from sklearn.externals import joblib

			if not (args.dump == "") :
				#pickle.dump(e, open(args.dump,'wb'))
				joblib.dump(e, args.dump)

			# force to free memory
			del e

			# stores fold results
			folds_time = folds_time + [end - start]
			self._evaluate_dump(k, y_test, pred, args)
			k = k + 1

			if args.test:
				break

		print("F1-Score")
		print("\tMicro: ", np.average(self.folds_micro), np.std(self.folds_micro))
		print("\tMacro: ", np.average(self.folds_macro), np.std(self.folds_macro))

		print('loading time : ', self.datasetLoadingTime)
		print('times : ', np.average(folds_time), np.std(folds_time))

	
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

	def _init_parser(self):
		super(TextClassificationApp, self)._init_parser()

		self.parser.add_argument("-g","--n_gpus", type=int, help='Number of GPUs available to execute kNN-based classifiers (Default:1).', default=1)

		self.parser.add_argument("-i", "--n_iterations", type=int, help='Number of iteration for broof to perform (Default:200).', default=200)

		self.parser.add_argument("-t", "--trees", type=int, help='Number of trees (Default:200).', default=200)

		self.parser.add_argument("-k", "--n_neighbors", type=int, help='Number of nearest neirghbors (Default:30).', default=30)

		self.parser.add_argument("-f", "--max_features", help='The number of features to consider when looking for the best split (Default:sqrt).', default='sqrt')

		self.parser.add_argument("--criterion", help='Split criterion (Default:gini).', default='gini')

		self.parser.add_argument("--learning_rate", type=float, help='Algorithm learning rate. It controls algorithm\'s convergence.', default=1.0)

		self.parser.add_argument("-a","--alpha", type=float, help='Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).(Default: 1)', default=1)

		self.parser.add_argument("-c", "--C", type=float, help='Penalty parameter C of the error term. For SVM training. (Default: 1)', default=1)

		self.parser.add_argument("-b", "--base_estimator", choices=self.instantiator.get_estimators(), default='dt')


	def _setup_instantiator(self, args):
		random_instance = check_random_state(args.seed)
		
		self.instantiator.set_general_params(vars(args))
		self.instantiator.set_general_params(
						{'random_state': random_instance.randint(MAX_INT)})

		estimators_params = {
			'rf':	{'n_estimators': args.trees},
			'xt': 	{'n_estimators': args.trees},
			'bag': 	{'n_estimators': args.trees},
			'lazy':	{'n_estimators': args.trees},
			'lxt': 	{'n_estimators': args.trees},
			'broof':{'n_trees': args.trees},
			'bert': {'n_trees': args.trees},
			'xgb': {'n_estimators': args.trees}
		}

		self.instantiator.set_params(estimators_params)

		estimators_params = {
			'bag':	{'base_estimator': self.instantiator.get_instance(args.base_estimator)}
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
		except Exception as e:
			if args.max_features == "random":
				args.max_features = 1

		return args 

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

	def _init_parser(self):
		super(StackerApp, self)._init_parser()
		self.parser.add_argument("base_classifiers", type=str, help='Base-level classifier.')

		self.parser.add_argument("--base_params", type=str, help='Base-level classifier.', default="[]")

		self.parser.add_argument("meta_classifiers", type=str, help='Base-level classifier.')

		self.parser.add_argument("--meta_params", type=str, help='Base-level classifier.', default="[]")

	def _setup_instantiator(self, args):
		
		estimator, tuned_parameters = super(StackerApp, self)._setup_instantiator(args)
		self.instantiator.set_general_params({'probability': True, 'cv':5})
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

		meta_level = [Pipeline(default_transformers[estimator] + [('estimator', self.instantiator.get_instance(estimator, params[i]))]) for i, estimator in enumerate(meta_classifiers)]
		
		stack = list()
		stack.append(base_level)
		stack.append(meta_level)

		args.meta = meta_classifiers
		
		return StackingClassifier(estimators_stack=stack,
						 random_state=random_instance.randint(MAX_INT))


	def _evaluate_dump(self, k, y_test, pred, args):
		print(pred)
		
		if(pred.ndim == 2):
			for i, p in enumerate(pred.T):
				result = np.array([range(len(y_test)), y_test, p])
				
				if not (args.output == "") :
					f=open(args.output.replace("{combiner}", args.meta[i]),'ab')
					np.savetxt(f, result.transpose(), fmt='%d', header=str(k))
					f.close()

			return

		result = np.array([range(len(y_test)), y_test, pred])

		print("F1-Score")
		print("\tMicro: ", f1_score(y_true=y_test, y_pred=pred, average='micro'))
		print("\tMacro: ", f1_score(y_true=y_test, y_pred=pred, average='macro'))

		if not (args.output == "") :
			f=open(args.output, 'ab')
			np.savetxt(f, result.transpose(), fmt='%d', header=str(k))
			f.close()

	def _tfidf(self, args):
		return False

if __name__ == '__main__':
	app = StackerApp()

	app.run(app.parse_arguments())
