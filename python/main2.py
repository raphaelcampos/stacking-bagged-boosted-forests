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

		feats = [range(7)]
		return X_train.toarray()[:,feats], X_test.toarray()[:,feats], y_train, y_test

	def run(self, args):
		X_train, X_test, y_train, y_test = self._load_dataset(args)

		print(X_train)

		k = 0
		estimator, tuned_parameters = self._setup_instantiator(args)

		folds_time = []
		self.folds_predict = []
		self.folds_macro = []
		self.folds_micro = []

		print(estimator.get_params(deep=False))
			
		tf_transformer = TfidfTransformer(norm=args.norm, use_idf=True,
										 smooth_idf=True, sublinear_tf=True)
		if self._tfidf(args):
			# Learn the idf vector from training set
			tf_transformer.fit(X_train)
			# Transform test and training frequency matrix
			# based on training idf vector		
			X_train = tf_transformer.transform(X_train)
			X_test = tf_transformer.transform(X_test)


		from xsklearn.ensemble import GaussianOutlierRemover, ThresoldOutlierRemover

		#out_remov = ThresoldOutlierRemover()
		#X_train, y_train = out_remov.fit_transform(X_train, y_train)

		#out_remov = GaussianOutlierRemover(0.01)
		#X_train, y_train = out_remov.fit_transform(X_train, y_train)

		from sklearn.decomposition import PCA
		#pca = PCA(copy=True, n_components=2, whiten=False)
		#X_train = pca.fit_transform(X_train)
		#X_test = pca.transform(X_test)

		if(args.cv > 1):
			n_jobs = 1 if hasattr(estimator,"n_jobs") else args.n_jobs
			gs = GridSearchCV(estimator, tuned_parameters,
						 n_jobs=n_jobs, refit=False,
						 cv=args.cv, verbose=1, scoring='f1_micro')
			gs.fit(X_train, y_train)
			print(gs.best_score_, gs.best_params_)
			estimator.set_params(**gs.best_params_)
			print(estimator.get_params())

		e = clone(estimator)

		# fit and predict
		start = time.time()
		e.fit(X_train, y_train)
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

		print("F1-Score")
		print("\tMicro: ", np.average(self.folds_micro), np.std(self.folds_micro))
		print("\tMacro: ", np.average(self.folds_macro), np.std(self.folds_macro))

		print('loading time : ', self.datasetLoadingTime)
		print('times : ', np.average(folds_time), np.std(folds_time))


app = TextClassification2App()

app.run(app.parse_arguments())