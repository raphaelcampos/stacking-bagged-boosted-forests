from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.cross_validation import train_test_split, cross_val_predict, cross_val_score, KFold, StratifiedKFold

from sklearn.grid_search import GridSearchCV

from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

import numpy as np

class MetaLevelTransformerCV(object):
	"""docstring for MetaLevelTransformerCV"""
	def __init__(self, base_estimators,
				 n_folds = 5,
				 stratified = True,
				 probability = True,
				 random_state = None):
		super(MetaLevelTransformerCV, self).__init__()
		
		self.base_estimators = base_estimators
		self.n_folds = n_folds
		self.stratified = stratified
		self.probability = probability
		self.random_state = random_state


	def fit(self, X, y):
		"""
			Fit base estimators
		"""
		for estimator in self.base_estimators:
			estimator.fit(X, y)

	def transform(self, X):
		"""
			Transform X accordingly to the fitted base classifires.
			It is used at prediction step of stacking
		"""
		if self.probability:
			Xi = np.zeros((X.shape[0], len(self.base_estimators)*self.n_classes_))
		else:
			Xi = np.zeros((X.shape[0], len(self.base_estimators)))

		for j, estimator in enumerate(self.base_estimators):
			Xt = X.copy()
			if isinstance(estimator, Pipeline):
				for name, transform in estimator.steps[:-1]:
					Xt = transform.transform(Xt)

			if self.probability:
				Xi[:, j*self.n_classes_:( (j + 1)*self.n_classes_ )] = estimator.predict_proba(Xt)
			else:
				Xi[:, j] = estimator.predict(Xt)


		return Xi

	def fit_transform(self, X, y):
		"""
			Fit base estimators and transform X
			into meta level dataset using K-fold 
			cross-validation
		"""
		self.classes_ = np.unique(y)
		self.n_classes_ = len(self.classes_)

		kf = StratifiedKFold(y, n_folds=self.n_folds,
							 shuffle=True, random_state=self.random_state)

		if self.probability:
			Xi = np.zeros((X.shape[0], len(self.base_estimators)*self.n_classes_))
		else:
			Xi = np.zeros((X.shape[0], len(self.base_estimators)))

		for train_index, test_index in kf:
			# split dataset
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]

			for j, estimator in enumerate(self.base_estimators):
				e = clone(estimator)
				e.fit(X_train, y_train)
	
				Xt_test = X_test.copy()
				if isinstance(e, Pipeline):
					for name, transform in e.steps[:-1]:
						Xt_test = transform.transform(Xt_test)

				if self.probability:
					idxs = j*self.n_classes_ + np.searchsorted(
										self.classes_, np.unique(y_train))
					Xi[np.repeat(test_index, len(idxs)), np.tile(idxs,
					 len(test_index))] = e.predict_proba(Xt_test).reshape(
					 								len(idxs)*len(test_index))
				else:
					Xi[test_index, j] = e.predict(Xt_test)

				# force memory release
				del e

		#from sklearn.datasets import dump_svmlight_file
		#dump_svmlight_file(Xi, y, "meta_level_10fold.svm")
		#exit()
		self.fit(X, y)

		return Xi

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
    probability: bool, optional (default=True)
    	If it is true the stacking procedure will use predict_proba to
    	generate the next level training set. Otherwise it will use
    	predict result.

    Attributes
    ----------
    
    References
    ----------
    .. [1] David H. Wolpert, "Stacked Generalization", Neural Networks, 5, 241--259, 1992.
    """
	def __init__(self, estimators_stack, n_folds=5, verbose=0, probability=True, random_state=None):

		self.check_estimators(probability)

		self.estimators_stack = estimators_stack
		self.n_folds = n_folds
		self.random_state = random_state
		self.probability = probability
		self.verbose=verbose

	def check_estimators(self, probability=True):
		pass

	def fit(self, X, y):
		self.levels = len(self.estimators_stack)
		
		self.meta_transformers = []

		X_tmp = X.copy()
		for l in range(self.levels - 1):
			meta_transformer = MetaLevelTransformerCV(self.estimators_stack[l],
								 n_folds = self.n_folds,
								 probability=self.probability,
								 random_state=self.random_state)

			X_tmp = meta_transformer.fit_transform(X_tmp, y)
			
			self.meta_transformers.append(meta_transformer)

		
		self.estimators_stack[l + 1].fit(X_tmp, y)

		if hasattr(self.estimators_stack[l + 1], "coef_"):
			print(self.estimators_stack[l + 1].coef_)
		elif hasattr(self.estimators_stack[l + 1], "feature_importances_"):
			print(self.estimators_stack[l + 1].feature_importances_)

		return self

	def predict(self, X):
		self.levels = len(self.estimators_stack)
	
		X_tmp = X.copy()
		
		for l in range(self.levels - 1):
			X_tmp = self.meta_transformers[l].transform(X_tmp)

		return self.estimators_stack[l + 1].predict(X_tmp)


import scipy as sp
from scipy.linalg import inv
from scipy.special import gamma, multigammaln
from scipy.stats import multivariate_normal

def logB(W0, v0):
	D, _ = W0.shape
	return -((v0/2.)*np.log(sp.linalg.det(W0)) + 
				(v0*D/2.)*np.log(2.) + multigammaln(v0/2., D))  

def Li(W0, beta0, W, v, S, J):
	return (-logB(W, v) - 0.5*v*np.trace((S + inv(W0)
											 + ((beta0*N)/(beta0 + N))*J)*W)
				- 0.5*np.log(sp.linalg.det(W)))

from bayespy import nodes
from bayespy.inference import VB

class VIG(BaseEstimator):

	def __init__(self):
		super(VIG, self).__init__()

	def fit(self, X, y):
		n_samples, n_features = X.shape
		
		classes_ = np.unique(y)
		n_classes_ = len(classes_)
		
		n_estimators = n_features/n_classes_

		self.models_ = []
		for i, Y in enumerate(classes_):
			L = X[y == Y,:]
			
			N, D = L.shape
			
			#L_ = L.reshape((N, n_estimators, n_classes_)).mean(1)
			#pred = classes_.take(np.argmax(L_, axis=1), axis=0)
			
			#L = L[(pred == Y)]

			#N, D = L.shape
			

			Lambda = nodes.Wishart(D, np.identity(D)*(1./D))
			mu = nodes.Gaussian(np.zeros(D), Lambda)

			x = nodes.Gaussian(mu, Lambda, plates=(N,))
			x.observe(L)

			Q = VB(x, mu, Lambda)
			Q.update(repeat=200, tol=0)
			cov = np.linalg.inv(Lambda.u[0])
			m = mu.u[0]
			
			self.models_.append([m, cov, float(L.shape[0])/n_samples])

		self.n_classes_ = n_classes_
		self.classes_ = classes_

		return self

	def predict(self, X):
		n_samples = X.shape[0]
		n_classes_ = self.n_classes_ 
		pred = np.zeros((n_samples, n_classes_))
		
		for i, model in enumerate(self.models_):
			m, cov, prior = model
			pred[:,i] = multivariate_normal.pdf(X, mean=m, cov=cov)*prior

		return self.classes_.take(np.argmax(pred, axis=1), axis=0)
		

class VIG2(BaseEstimator):

	def __init__(self, eps=1e-10, beta = 1):
		super(VIG, self).__init__()
		self.eps=eps
		self.beta=0

	def _variational_inference(self, X, beta0, v0, m0, W0, E):
		# likelihood
		

		N, D = float(X.shape[0]), float(X.shape[1])
		
		
		#S = np.sum(np.square(X - x_), axis=0)

		W1 = W0
		i = 1

		Xo = X
		x_ =  X.mean(0)
		H = E
		while True:
			# E-step
			#pW = scipy.stats.wishart.pdf(df=v0, scale=W0)
			#pMu = multivariate_normal.pdf(obs, mean=m0, cov=inv(v0*W0))
			#pX = multivariate_normal.pdf(Xo, mean=np.squeeze(np.asarray(m0)), cov=inv(v0*W0))


			#Xe = np.multiply(Xo, pX.reshape((pX.shape[0],1)))
			# M-step
			

			# equation 11
			m = (beta0*m0 + N*x_)/(beta0 + N)
			
			# equation 12
			H = (beta0 + N)*(E)
			
			# equation 13
			v = v0 + N + 1

			# equation 14
			
			#S = np.sum(np.sum(np.square(X - x_), axis=1)).reshape((1,1))
			S = 0
			for x in X:
				diff = x - x_
				S = S + diff.dot(diff.T)

			J = (x_ - m)*(x_ - m).T  
			#W = inv(inv(W0) + (beta0 + N)*inv(H))
			W = inv(inv(W0) + (beta0 + N)*inv(H) + S + ((beta0*N)/(beta0 + N))*J)
			
			if i > 1:
				print(Li(W1, beta0, W, v, S, J) - Li(W1, beta0, W0, v, S, J))
				if Li(W1, beta0, W, v, S, J) - Li(W1, beta0, W0, v, S, J) < self.eps:
					break
			
			i = i + 1
			m0, H0, v0, W0 = m, H, v, W

		return m, H, v, W

	def fit(self, X, y):

		n_samples, n_features = X.shape
		self.classes_ = np.unique(y)

		self.gaussians_ = []

		m0 = np.zeros(D)
		beta0 = 10
		v0 = float(n_features)
		W0 = inv(sp.identity(n_features)*10)
		E = v0*W0
		
		for i in self.classes_:
			print("Y = %s" % i)
			L = X[y == i,:]
			m, _, v, W = vig._variational_inference(L, beta0, v0, m0, W0, E)

			self.gaussians_.append([np.squeeze(np.asarray(m)), W*v, L.shape[0]/float(n_samples)])

		return self

	def predict(self, X):

		n_classes_ = len(self.gaussians_)
		n_samples = X.shape[0]
		
		pred = np.zeros((n_samples, n_classes_))
		
		for i, g in enumerate(self.gaussians_):
			m, lamda, pGm = g
			print(m, inv(lamda))
			return multivariate_normal.pdf(X, mean=m, cov=inv(lamda))

		#return self.classes_.take(np.argmax(pred, axis=1), axis=0)

if __name__ == "__main__":

	from sklearn.datasets import load_svmlight_file
	#X, y = load_svmlight_file("meta_level.svm")

	true_mean = [2,4]
	true_cov = [[10,5],[5,10]]

	X = multivariate_normal.rvs(mean=true_mean, cov=true_cov, size=250, random_state=42)
	y = np.ones(X.shape[0])

	N, D = X.shape

	vig = VIG()
	vig.fit(X, y)
	pred = vig.predict(X)
	print(pred)
	print(multivariate_normal.pdf(X, mean=true_mean, cov=true_cov))