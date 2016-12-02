import numpy as np
import scipy
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.base import BaseEstimator


class SCANN(BaseEstimator):
	"""Stacking Correspondence Analysis Nearest Neighbour
    Attributes
    ----------
    models_ : list,
		Each element contains tuple mean, covariance matrix and prior probability.
		The attribute is created after fitting the model.
    References
    ----------
    .. [1] Merz, Christopher J.
    		1999. Using Correspondence Analysis to Combine Classifiers. 
    		Machine Learning. 36, pages 33-58.
    """
	def __init__(self):
		super(SCANN, self).__init__()

	def fit(self, X, y):
		self.classes = classes = np.unique(y)
		self.n_classes = n_classes = len(classes)

		n_samples, n_features = X.shape
		n_estimators = X.shape[1]/n_classes

		# Stage 1
		N = X.reshape((n_samples, n_estimators, n_classes))
		
		c = (np.hstack((np.argmax(N, axis=2), np.asarray(y, dtype=int)[:,np.newaxis])) + n_classes*np.arange(n_estimators + 1)).ravel()
		r = np.repeat(np.arange(n_samples), n_estimators + 1)

		N = scipy.sparse.coo_matrix((np.ones(len(c)),(r,c)), (n_samples, n_classes*(n_estimators + 1)))
		N_csc = N.tocsc()
		N_csr = N.tocsr()

		n = N.sum()
		r = np.asarray(N_csr.sum(1)/n).ravel()
		c = np.asarray(N_csc.sum(0)/n).ravel()
		
		# zero value leads to inf in afterwards operations
		# Thus, we set it to a really small number
		c[c == 0] = 1e-200
		
		P = N/n
		
		D_r = scipy.sparse.diags(r, offsets=0)
		D_c = scipy.sparse.diags(c, offsets=0)
		
		# Stage 2
		D_r.data = np.power(D_r.data, -0.5)

		D_c.data = np.power(D_c.data, -0.5)
		
		rc = r[:,np.newaxis] * c[:, np.newaxis].T
		

		A = scipy.sparse.dia_matrix.dot(D_c, D_r.dot(P - rc).T).T
		
		U, Sigma, V = scipy.linalg.svd(A, False)
		
		Sigma = scipy.sparse.diags(Sigma, offsets=0)

		# Stage 3
		self.F = Sigma.dot(D_r.dot(U).T).T
		self.G = Sigma.dot(D_c.dot(V.T).T).T

		self.U = U
		self.Sigma = Sigma
		self.V = V

		self.y = y

		return self

	def decision_function(self, X):
		
		classes = self.classes
		n_classes = self.n_classes

		n_samples, n_features = X.shape
		n_estimators = X.shape[1]/n_classes

		M = X.reshape((n_samples, n_estimators, n_classes))
		
		c = (np.argmax(M, axis=2) + n_classes*np.arange(n_estimators)).ravel()
		r = np.repeat(np.arange(n_samples), n_estimators)

		M = scipy.sparse.coo_matrix((np.ones(len(c))/float(n_estimators + 1),(r,c)), (n_samples, n_classes*(n_estimators))).tocsr()
		
		pred = np.zeros((n_samples, n_classes))
		for clazz in classes:
			labels = np.zeros((M.shape[0], n_classes))
			labels[:, clazz] = 1/float(n_estimators + 1)
			r_prof = scipy.sparse.hstack((M, labels))
			
			f = r_prof.dot(scipy.sparse.linalg.inv(self.Sigma).dot(self.G.T).T)
			
			dist = euclidean_distances(f[:,:], self.G[(clazz-n_classes),:])
			pred[:, clazz] = dist.ravel()
		
		return 1/(1 + pred)
		
	def predict(self, X):
		
		pred = self.decision_function(X)

		return self.classes.take(np.argmax(pred, axis=1), axis=0)