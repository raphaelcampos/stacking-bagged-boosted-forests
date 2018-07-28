import warnings
from app import TextClassificationApp

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


def all_sets(elems, i):
	sets = []

	for e in elems:
		tmp = []
		for s in sets:
			tmp.append(s + [e])
		tmp.append([e])
		sets = sets + tmp

	return sets

def _predict_proba_lr(X):
	"""Probability estimation for OvR logistic regression.
	Positive class probabilities are computed as
	1. / (1. + np.exp(-self.decision_function(X)));
	multiclass is handled by normalizing that over all classes.
	"""	
	prob = X*1000
	prob *= -1
	np.exp(prob, prob)
	prob += 1
	np.reciprocal(prob, prob)
	if prob.ndim == 1:
		return np.vstack([1 - prob, prob]).T
	else:
		# OvR normalization, like LibLinear's predict_probability
		prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
		return prob

warnings.filterwarnings("ignore")

class TextClassification2App(TextClassificationApp):
	def _init_parser(self):
		super(TextClassification2App, self)._init_parser()

		self.parser.add_argument("train", type=str,
                    help="SVM light format training set.")

		self.parser.add_argument("test", type=str,
                    help="SVM light format training set.")

		self.parser._actions.pop(1)


	def _load_dataset(self, args):
		start = time.time()
			
		X_train, y_train = load_svmlight_file(args.train)
		X_test, y_test = load_svmlight_file(args.test)	


		end = time.time()

		self.datasetLoadingTime = end - start;

		i = np.asarray([0,1,2,3,4,5,6,7,8])[:, np.newaxis]
		#i = np.arange(4,9)[:, np.newaxis]
		n_classes = len(np.unique(y_train))
		feats = (n_classes*i + np.arange(n_classes)).ravel()
		
		# return X_train.toarray()[:,feats], X_test.toarray()[:,feats], y_train, y_test
		return X_train.toarray(), X_test.toarray(), y_train, y_test

	def run(self, args):
		X_train, X_test, y_train, y_test = self._load_dataset(args)


		k = 0
		estimator, tuned_parameters = self._setup_instantiator(args)

		folds_time = []
		self.folds_predict = []
		self.folds_macro = []
		self.folds_micro = []

		print(estimator.get_params(deep=False))
			
		tf_transformer = TfidfTransformer(norm=args.norm, use_idf=False,
										 smooth_idf=False, sublinear_tf=False)
		if self._tfidf(args):
			# Learn the idf vector from training set
			tf_transformer.fit(X_train)
			# Transform test and training frequency matrix
			# based on training idf vector		
			X_train = tf_transformer.transform(X_train)
			X_test = tf_transformer.transform(X_test)


		if(args.cv > 1):
			n_jobs = 1 if hasattr(estimator,"n_jobs") or hasattr(estimator,"nthread") else args.n_jobs
			gs = GridSearchCV(estimator, tuned_parameters,
						 n_jobs=n_jobs, refit=False,
						 cv=args.cv, verbose=5, scoring='f1_macro')
			gs.fit(X_train, y_train)
			print(gs.best_score_, gs.best_params_)
			estimator.set_params(**gs.best_params_)
			print(estimator.get_params())

		e = clone(estimator)

		# fit and predict
		start = time.time()
		e.fit(X_train, y_train)
		pred = e.predict(X_test)
		
		#from scipy import io
		#io.savemat('piSet', {'piSetMean':X_test.reshape((X_test.shape[0], 9, X_test.shape[1]/9)).mean(1), 'piSet':e.predict_proba(X_test), "trueLabel" : y_test})
		#exit()
		end = time.time()

		if hasattr(e, 'staged_predict'):
			ada_discrete_err = np.zeros((args.n_iterations,))
			for i, y_pred in enumerate(e.staged_predict(X_test)):
				ada_discrete_err[i] = np.mean(y_pred == y_test)
			print(ada_discrete_err)
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

		print("F1-Score")
		print("\tMicro: ", np.average(self.folds_micro), np.std(self.folds_micro))
		print("\tMacro: ", np.average(self.folds_macro), np.std(self.folds_macro))

		print('loading time : ', self.datasetLoadingTime)
		print('times : ', np.average(folds_time), np.std(folds_time))


app = TextClassification2App()

app.run(app.parse_arguments())