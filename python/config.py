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
from sklearn import naive_bayes, neighbors, svm, ensemble, tree
from sklearn import linear_model, discriminant_analysis

from xsklearn.neighbors import LazyNNRF, LazyNNExtraTrees, LazyNNBroof
from xsklearn.ensemble import Broof, Bert, VIG, DecisionTemplates, SCANN
from xsklearn.linear_model import MLR, LinearSVM, LinearModelTree
from xsklearn import DSC

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectFromModel 

import xgboost as xgb

import numpy as np

base_estimators = {
	'lr': linear_model.LinearRegression,
	'svm': svm.SVC,
	'svr': svm.SVR,
	'lsvm': LinearSVM,
	'nb': naive_bayes.MultinomialNB,
	'knn': neighbors.KNeighborsClassifier,
	'rf': ensemble.RandomForestClassifier,
	'rfr': ensemble.RandomForestRegressor,
	'xt': ensemble.ExtraTreesClassifier,
	'lazy': LazyNNRF,
	'lxt': LazyNNExtraTrees,
	'broof': Broof,
	'lazy_broof': LazyNNBroof,
	'bert': Bert,
	'mlr': MLR,
	'dt': tree.DecisionTreeClassifier,#LinearModelTree
	'vig': VIG,
	'DT': DecisionTemplates,
	'lda': discriminant_analysis.LinearDiscriminantAnalysis,
	'qda': discriminant_analysis.QuadraticDiscriminantAnalysis,
	'csvm': svm.SVC,
	'reg': linear_model.LogisticRegression,
	'ridge': linear_model.RidgeClassifierCV,
	'gbt': ensemble.GradientBoostingClassifier,
	'adarf': ensemble.AdaBoostClassifier,
	'dsc': DSC,
	'xgb': xgb.XGBClassifier,
	'scann': SCANN,
	'cent': neighbors.NearestCentroid,
	'bag': ensemble.BaggingClassifier
}

# default parameters for text classification
default_params = {
	'svm': 	{'kernel': 'linear', 'C': 1, 'verbose': False, 'probability': False,
			 'degree': 3, 'shrinking': True, 'max_iter': -1, 
			 'decision_function_shape': None, 'random_state': None, 
			 'tol': 0.001, 'cache_size': 1000, 'coef0': 0.0, 'gamma': 'auto', 
			 'class_weight': None},
	'svr': 	{},
	'lsvm': {'loss': 'squared_hinge', 'C': 1, 'verbose': 0, 'intercept_scaling': 1,
			 'fit_intercept': True, 'max_iter': 1000, 'penalty': 'l2',
			 'multi_class': 'ovr', 'random_state': None, 'dual': False, 
			 'tol': 0.001, 'class_weight': None},
	'nb':  	{'alpha': 1, 'fit_prior': True, 'class_prior': None},
	'knn': 	{'n_neighbors': 30, 'n_jobs': 1, 'algorithm': 'brute',
			 'metric': 'cosine', 'metric_params': None, 'p': 2, 
			 'weights': 'distance', 'leaf_size': 30},
	'rf':  	{'warm_start': False, 'oob_score': True, 'n_jobs': 1, 'verbose': 0,
			 'max_leaf_nodes': None, 'bootstrap': True, 'min_samples_leaf': 1,
			 'n_estimators': 200, 'min_samples_split': 2,
			 'min_weight_fraction_leaf': 0.0, 'criterion': 'gini', 
			 'random_state': None, 'max_features': 'auto', 'max_depth': None, 
			 'class_weight': None},
	'rfr':  {},
	'lr': {},
	'xt':  	{'warm_start': False, 'oob_score': False, 'n_jobs': 1, 'verbose': 0,
			 'max_leaf_nodes': None, 'bootstrap': False, 'min_samples_leaf': 1, 
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
			 'verbose': 0, 'max_leaf_nodes': None, 'bootstrap': False,
			 'oob_score': False, 'min_samples_leaf': 1, 'n_estimators': 200, 
			 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 
			 'criterion': 'gini', 'random_state': None, 'max_features': 'auto',
			 'max_depth': None, 'class_weight': None},
	'lazy_broof':	{'warm_start': False, 'n_neighbors': 8, 'n_gpus': 0, 'n_jobs': 1,
			 'verbose': 0, 'max_leaf_nodes': None, 'bootstrap': True,
			 'oob_score': False, 'min_samples_leaf': 1, 'n_estimators': 24, 
			 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 
			 'criterion': 'gini', 'random_state': None, 'max_features': 'auto',
			 'max_depth': None, 'class_weight': None},
	'broof':{'warm_start': False, 'n_jobs': 1, 'verbose': 0, 'n_iterations': 200,
			 'max_leaf_nodes': None, 'learning_rate': 1, 'n_trees': 10, 
			 'min_samples_leaf': 1, 'min_samples_split': 2, 
			 'min_weight_fraction_leaf': 0.0, 'criterion': 'gini', 
			 'random_state': None, 'max_features': 'auto', 'max_depth': None, 
			 'class_weight': None, 'oob_error':True, 'selective_updates':True},
	'bert': {'warm_start': False, 'n_jobs': 1, 'verbose': 0, 'n_iterations': 200,
			 'max_leaf_nodes': None, 'learning_rate': 1, 'n_trees': 8,
			 'min_samples_leaf': 1, 'min_samples_split': 2,
			 'min_weight_fraction_leaf': 0.0, 'criterion': 'gini',
			 'random_state': None, 'max_features': 'auto', 'max_depth': None,
			 'class_weight': None, 'oob_error':True, 'selective_updates':True},
	'mlr':	{},
	'dt': 	{'max_features': 1.0, 'max_leaf_nodes': None,
			 'min_weight_fraction_leaf': 0.0, 'splitter': 'best',
			 'min_samples_leaf': 1, 'max_depth': 16, 'presort': False,
			 'criterion': 'gini', 'random_state': None, 'class_weight': None,
			 'min_samples_split': 2},
	'vig': {},
	'DT': {},
	'lda': {'n_components':None, 'priors':None, 'shrinkage':"auto",
			 'solver':'eigen', 'store_covariance':False, 'tol':0.0001},
	'qda': {},
	'csvm': {'kernel': 'linear', 'C': 1, 'verbose': False, 'probability': True,
			 'degree': 3, 'shrinking': True, 'max_iter': -1, 
			 'decision_function_shape': None, 'random_state': None, 
			 'tol': 0.001, 'cache_size': 1000, 'coef0': 0.0, 'gamma': 'auto', 
			 'class_weight': None},
	'reg': {},
	'ridge': {},
	'gbt': {'loss':'deviance', 'learning_rate': 0.1, 'n_estimators':200, 
			'subsample':1.0, 'min_samples_split':2, 'min_samples_leaf':1, 
			'min_weight_fraction_leaf':0.0, 'max_depth': 3, 'init':None, 
			'random_state':None, 'max_features': 0.5, 'verbose':0, 
			'max_leaf_nodes':None, 'warm_start':False, 'presort':'auto'},
	'adarf': {'n_estimators': 200, 'base_estimator': ensemble.RandomForestClassifier(n_jobs=8,max_features=0.15,n_estimators=20),
			 'random_state': None, 'learning_rate': 1, 'algorithm': 'SAMME.R'},
	'dsc': {'alpha': 2.0},
	'xgb': {'reg_alpha': 0, 'colsample_bytree': 1.0, 'silent': True, 'colsample_bylevel': 1.,
 				'scale_pos_weight': 1, 'learning_rate': 0.1, 'missing': None, 'max_delta_step': 0,
 				'nthread': 8, 'base_score': 0.5, 'n_estimators': 200, 'subsample': 1., 'reg_lambda': 1,
 				'seed': 42, 'min_child_weight': 2, 'objective': 'binary:logistic', 'max_depth': 5, 'gamma': 0},
 	'scann': {},
 	'cent': {'metric': 'cosine'},
 	'bag': {'warm_start': False, 'max_samples': 1.0, 'base_estimator': None, 
 					'n_jobs': 1, 'verbose': 0, 'bootstrap': True, 'oob_score': True,
 				 	'n_estimators': 10, 'random_state': None, 'max_features': 1.0,
 				 	'bootstrap_features': False}
}

default_tuning_params = {
	'svm': 	[{'C': 2.0 ** np.arange(-5, 15, 2)}],
	'svr': 	[{'C': 2.0 ** np.arange(-5, 15, 2)}],
	'lsvm': [{'C': 2.0 ** np.arange(-5, 9, 2)}],
	'nb':  	[{'alpha': [0.0001, 0.001, 0.01, 0.1,0.5,1,1.5,10,100]}],
	'knn': 	[{'n_neighbors': [10, 30, 100, 200, 300], 'weights': ['uniform', 'distance']}],
	'rf': [{'criterion': ['entropy', 'gini'], #'max_depth': [2, 5, 8, None],
			'n_estimators': [200], 'max_features': ['sqrt', 'log2', 0.08]}],	
	'rfr': [{'criterion': ['mae'], 'max_depth': [5],
			'n_estimators': [50], 'max_features': [0.15]}],
	'lr': [{}],
	'xt': [{'criterion': ['entropy', 'gini'], 
			'n_estimators': [200], 'max_features': ['sqrt', 'log2', 0.08,
			0.15, 0.30]}],
	'lazy': [{'n_neighbors': [10, 30, 100, 200, 300, 500], 'criterion': ['entropy', 'gini'], 
			'n_estimators': [100], 'max_features': ['sqrt']}],
	'lxt': [{'n_neighbors': [10, 30, 100, 200, 300, 500], 'criterion': ['entropy', 'gini'], 
			'n_estimators': [100], 'max_features': ['sqrt']}],
	'broof': [{}],
	'lazy_broof': [{}],
	'bert': [{'n_trees': [5, 8, 10, 15, 25], 'n_iterations': [50, 100, 200],
				'max_features': ['sqrt', 'log2', 0.08, 0.15, 0.30]}],
	'mlr': [],
	'dt': [{'criterion': ['entropy', 'gini'], 'max_depth': [None] + list(2.0 ** np.arange(0, 10, 1))}],
	'vig': [],
	'DT': [],
	'lda': [],
	'qda': [],
	'csvm': [],
	'reg': [{}],
	'ridge': [{}],
	'gbt': [{'learning_rate': [0.1, 0.2, 0.5, 0.8, 1.0], 'max_depth': [None] + list(2.0 ** np.arange(0, 10, 1))}],
	'adarf': [],
	'dsc': [{'alpha': [0, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0]}],
	'scann': [{}],
	#'xgb': [{'colsample_bytree': [0.1, 0.3, 0.5, 0.7, 1.0], 'subsample': [0.5, 0.7, 1.0]}]
	'xgb': [{'colsample_bytree': [0.1, 0.3, 0.5, 0.7, 1.0]}],#'reg_lambda': [0, 1e-5, 1e-2, 0.1, 1, 100], 'reg_alpha': [0, 1e-5, 1e-2, 0.1, 1, 100]}]
	'cent': [{}],
	'bag': [{}]
}

default_transformers = {
	'svm': 	[('tfidf', TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True))],
	'lsvm': [('tfidf', TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True))],
	'nb':  	[],
	'knn': 	[('tfidf', TfidfTransformer(norm="max", use_idf=True, smooth_idf=True, sublinear_tf=True))],
	'rf': [],
	'xt': [],
	'lazy': [],
	'lxt': [],
	'broof': [],
	'bert': [],
	#'mlr': [('selection', SelectFromModel(ensemble.ExtraTreesClassifier(n_jobs=8, n_estimators=300), threshold="mean"))],
	'mlr':[],
	'dt': [],
	'vig': [],
	'DT': [],
	'lda': [],
	'csvm': [('tfidf', TfidfTransformer(norm='l2', use_idf=False, smooth_idf=False, sublinear_tf=False))],
	'reg': [],
	'ridge': [],
	'gbt': [],
	'adarf': []
}

from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
class CentrionDRTransformer(object):

	def fit(self, X, y):
		self.centroids_ = neighbors.NearestCentroid(metric="cosine")\
												.fit(X, y).centroids_
		return self

	def transform(self, X):
		return normalize(cosine_similarity(X, self.centroids_))
