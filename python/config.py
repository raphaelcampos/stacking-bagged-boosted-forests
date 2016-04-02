"""
This file contains basic configuration for BaseApp class and its descendents.

Variables
---------
base_estimators: dict, {'estimator_name': estimator_class}
	Dictionaty containing base classifiers classes 
default_params: dict, {'estimator_name': dict}
	Dictionaty containing base classifiers default parameters
default_tuning_params: dict, {'estimator_name': dict}
	Dictionaty containing base classifiers default tunning parameter values
"""
from sklearn import naive_bayes, neighbors, svm, ensemble

from xsklearn.neighbors import LazyNNRF, LazyNNExtraTrees 
from xsklearn.ensemble import Broof, Bert
from xsklearn.linear_model import MLR

from sklearn.feature_extraction.text import TfidfTransformer

import numpy as np

base_estimators = {
	'svm': svm.SVC,
	'lsvm': svm.LinearSVC,
	'nb': naive_bayes.MultinomialNB,
	'knn': neighbors.KNeighborsClassifier,
	'rf': ensemble.RandomForestClassifier,
	'xt': ensemble.ExtraTreesClassifier,
	'lazy': LazyNNRF,
	'lxt': LazyNNExtraTrees,
	'broof': Broof,
	'bert': Bert,
	'mlr': MLR
}

# default parameters for text classification
default_params = {
	'svm': 	{'kernel': 'linear', 'C': 1, 'verbose': False, 'probability': False,
			 'degree': 3, 'shrinking': True, 'max_iter': -1, 
			 'decision_function_shape': None, 'random_state': None, 
			 'tol': 0.001, 'cache_size': 1000, 'coef0': 0.0, 'gamma': 'auto', 
			 'class_weight': None},
	'lsvm': {'loss': 'squared_hinge', 'C': 1, 'verbose': 0, 'intercept_scaling': 0.5,
			 'fit_intercept': True, 'max_iter': 1000, 'penalty': 'l2',
			 'multi_class': 'ovr', 'random_state': 1608637542, 'dual': True, 
			 'tol': 0.001, 'class_weight': None},
	'nb':  	{'alpha': 1, 'fit_prior': True, 'class_prior': None},
	'knn': 	{'n_neighbors': 30, 'n_jobs': 1, 'algorithm': 'brute',
			 'metric': 'cosine', 'metric_params': None, 'p': 2, 
			 'weights': 'distance', 'leaf_size': 30},
	'rf':  	{'warm_start': False, 'oob_score': False, 'n_jobs': 1, 'verbose': 0,
			 'max_leaf_nodes': None, 'bootstrap': True, 'min_samples_leaf': 1,
			 'n_estimators': 200, 'min_samples_split': 2,
			 'min_weight_fraction_leaf': 0.0, 'criterion': 'gini', 
			 'random_state': None, 'max_features': 'auto', 'max_depth': None, 
			 'class_weight': None},
	'xt':  	{'warm_start': False, 'oob_score': False, 'n_jobs': 1, 'verbose': 0,
			 'max_leaf_nodes': None, 'bootstrap': True, 'min_samples_leaf': 1, 
			 'n_estimators': 200, 'min_samples_split': 2, 
			 'min_weight_fraction_leaf': 0.0, 'criterion': 'gini', 
			 'random_state': None, 'max_features': 'auto', 'max_depth': None, 
			 'class_weight': None},
	'lazy':	{'warm_start': False, 'n_neighbors': 200, 'n_gpus': 0, 'n_jobs': 1,
			 'verbose': 0, 'max_leaf_nodes': None, 'bootstrap': True,
			 'oob_score': False, 'min_samples_leaf': 1, 'n_estimators': 200, 
			 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 
			 'criterion': 'gini', 'random_state': None, 'max_features': 'auto',
			 'max_depth': None, 'class_weight': None},
	'lxt': 	{'warm_start': False, 'n_neighbors': 200, 'n_gpus': 0, 'n_jobs': 1,
			 'verbose': 0, 'max_leaf_nodes': None, 'bootstrap': True,
			 'oob_score': False, 'min_samples_leaf': 1, 'n_estimators': 200, 
			 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 
			 'criterion': 'gini', 'random_state': None, 'max_features': 'auto',
			 'max_depth': None, 'class_weight': None},
	'broof':{'warm_start': False, 'n_jobs': 1, 'verbose': 0, 'n_iterations': 200,
			 'max_leaf_nodes': None, 'learning_rate': 1, 'n_trees': 8, 
			 'min_samples_leaf': 1, 'min_samples_split': 2, 
			 'min_weight_fraction_leaf': 0.0, 'criterion': 'gini', 
			 'random_state': None, 'max_features': 'auto', 'max_depth': None, 
			 'class_weight': None},
	'bert': {'warm_start': False, 'n_jobs': 1, 'verbose': 0, 'n_iterations': 200,
			 'max_leaf_nodes': None, 'learning_rate': 1, 'n_trees': 8, 
			 'min_samples_leaf': 1, 'min_samples_split': 2, 
			 'min_weight_fraction_leaf': 0.0, 'criterion': 'gini', 
			 'random_state': None, 'max_features': 'auto', 'max_depth': None, 
			 'class_weight': None},
	'mlr':	{}
}

default_tuning_params = {
	'svm': 	[{'C': 2.0 ** np.arange(-5, 15, 2)}],
	'lsvm': [{'C': 2.0 ** np.arange(-5, 15, 2)}],
	'nb':  	[{'alpha': [0.0001, 0.001,0.1,0.5,1,1.5,10,100]}],
	'knn': 	[{'n_neighbors': [10, 30, 100, 200, 300, 500], 'weights': ['uniform', 'distance']}],
	'rf': [{'criterion': ['entropy', 'gini'], 'n_estimators': [200], 
			'max_features': ['sqrt', 'log2', 0.08, 0.15, 0.30]}],
	'xt': [{'criterion': ['entropy', 'gini'], 
			'n_estimators': [200], 'max_features': ['sqrt', 'log2', 0.08,
			0.15, 0.30]}],
	'lazy': [{'n_neighbors': [10, 30, 100, 200, 300, 500], 'criterion': ['entropy', 'gini'], 
			'n_estimators': [200], 'max_features': ['sqrt', 'log2', 0.08,
			0.15, 0.30]}],
	'lxt': [{'n_neighbors': [10, 30, 100, 200, 300, 500], 'criterion': ['entropy', 'gini'], 
			'n_estimators': [200], 'max_features': ['sqrt', 'log2', 0.08,
			0.15, 0.30]}],
	'broof': [{'n_trees': [5, 8, 10, 15, 25], 'n_iterations': [50, 100, 200],
				'max_features': ['sqrt', 'log2', 0.08, 0.15, 0.30]}],
	'bertf': [{'n_trees': [5, 8, 10, 15, 25], 'n_iterations': [50, 100, 200],
				'max_features': ['sqrt', 'log2', 0.08, 0.15, 0.30]}],
	'mlr': []
}

default_transformers = {
	'svm': 	[('tfidf', TfidfTransformer(norm='max', use_idf=True, smooth_idf=True, sublinear_tf=True))],
	'lsvm': [('tfidf', TfidfTransformer(norm='max', use_idf=True, smooth_idf=True, sublinear_tf=True))],
	'nb':  	[],
	'knn': 	[('tfidf', TfidfTransformer(norm="max", use_idf=True, smooth_idf=True, sublinear_tf=True))],
	'rf': [],
	'xt': [],
	'lazy': [],
	'lxt': [],
	'broof': [],
	'bertf': [],
	'mlr': []
}
