import numpy as np
import scipy
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.base import BaseEstimator
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import pairwise_distances

class DSC(BaseEstimator):
	"""Domain-Specific Classifier
    Attributes
    ----------
    https://arxiv.org/pdf/1307.2669v1.pdf
    """
	def __init__(self, alpha=2.0):
		super(DSC, self).__init__()
		self.alpha = alpha

	def fit(self, X, y):
		
		classes = np.unique(y)
		n_classes = classes.shape[0]
		print(classes)
		profs = np.zeros((n_classes, X.shape[1]))
		for j in classes:
			D_lab = X[y == j]
			n_samples = D_lab.shape[0]
			
			D_lab = normalize(D_lab, norm='l1')
			profs[j] = D_lab.sum(0)/n_samples
			
		s = profs.sum(0)

		profs[profs <=  self.alpha * (s - profs)] = 0 

		self.profs_ = scipy.sparse.csr_matrix(normalize(profs, norm='l1'))
		self.n_classes_ = n_classes
		self.classes_ = classes

	def predict(self, X):
		X = normalize(X, norm='l1')
		#pred = np.dot(X, self.profs_)
		#pred = pairwise_distances(X, self.profs_, metric="l1", n_jobs=-1)
		pred = X.dot(self.profs_.T).toarray()

		return self.classes_.take(np.argmax(pred, axis=1))