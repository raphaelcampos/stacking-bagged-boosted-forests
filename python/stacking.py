from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.cross_validation import train_test_split, cross_val_predict, cross_val_score, KFold, StratifiedKFold

from sklearn.grid_search import GridSearchCV

from sklearn.base import clone
from sklearn.metrics import f1_score

import numpy as np

class StackingClassifier(BaseEstimator, ClassifierMixin):
	"""About the class...
    Parameters
    ----------
    estimators_stack : list,
        Each element contains the estimator of the level of the stack.
    n_folds : int, optional (default=5)
        Number of folds to generate the next level traning set.
    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.
    Attributes
    ----------
    
    References
    ----------
    .. [1] David H. Wolpert, "Stacked Generalization", Neural Networks, 5, 241--259, 1992.
    """
	def __init__(self, estimators_stack, n_folds=5, verbose=0, probability=False, random_state=None):

		self.check_estimators(probability)

		self.estimators_stack = estimators_stack
		self.n_folds = n_folds
		self.random_state = random_state
		self.probability = probability
		self.verbose=verbose

	def check_estimators(self, probability=True):
		pass

	def fit(self, X, y, sample_weight=None):
		self.levels = len(self.estimators_stack)
		self.classes_ = np.unique(y)
		self.n_classes_ = len(self.classes_)

		X_tmp = X.copy()
		for l in range(self.levels - 1):
			kf = StratifiedKFold(y, n_folds=self.n_folds, shuffle=True, random_state=self.random_state)

			if self.probability:
				Xi = np.zeros((X.shape[0], len(self.estimators_stack[l])*self.n_classes_))
			else:
				Xi = np.zeros((X.shape[0], len(self.estimators_stack[l])))

			for train_index, test_index in kf:
				# split dataset
				X_train, X_test = X_tmp[train_index], X_tmp[test_index]
				y_train, y_test = y[train_index], y[test_index]

				for j, estimator in enumerate(self.estimators_stack[l]):
					e = clone(estimator)
					e.fit(X_train, y_train, sample_weight)
					
					if self.probability:
						Xi[test_index, j*self.n_classes_:((j+1)*self.n_classes_)] = e.predict_proba(X_test)
					else:
						Xi[test_index, j] = e.predict(X_test)

					# force to free memory
					del e

			for estimator in self.estimators_stack[l]:
				estimator.fit(X_tmp, y, sample_weight)

			X_tmp = Xi

		self.estimators_stack[l + 1].fit(X_tmp, y)

		return self

	def predict(self, X):
		self.levels = len(self.estimators_stack)
	
		X_tmp = X.copy()
		for l in range(self.levels - 1):
			if self.probability:
				Xi = np.zeros((X.shape[0], len(self.estimators_stack[l])*self.n_classes_))
			else:
				Xi = np.zeros((X.shape[0], len(self.estimators_stack[l])))

			for j, estimator in enumerate(self.estimators_stack[l]):
				if self.probability:
					Xi[:, j*self.n_classes_:( (j + 1)*self.n_classes_ )] = estimator.predict_proba(X_tmp)
				else:
					Xi[:, j] = estimator.predict(X_tmp)

			X_tmp = Xi

		return self.estimators_stack[l + 1].predict(X_tmp)



