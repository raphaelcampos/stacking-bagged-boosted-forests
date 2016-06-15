import scipy
import numpy as np

from . import VIG

from sklearn.preprocessing import Normalizer

class OutlierRemover(object):
	"""docstring for OutlierRemover"""
	def __init__(self,):
		super(OutlierRemover, self).__init__()

	def fit(self, X, y):
		pass

	def transform(self, X, y):
		print(self.outliers_.shape[0]/float(X.shape[0]))
		
		return X[self.outliers_], y[self.outliers_]

	def fit_transform(self, X, y):
		return self.fit(X, y).transform(X, y)
		

class GaussianOutlierRemover(OutlierRemover):
	"""docstring for GaussianOutlierRemover"""
	def __init__(self, conf_level=0.99):
		super(GaussianOutlierRemover, self).__init__()
		self.conf_level =  conf_level

	def fit(self, X, y):
		self.models_ = VIG().fit(X, y).models_		

		classes = np.unique(y)

		self.outliers_ = np.ndarray(0, dtype=int)
		for i, model in enumerate(self.models_):
			m, cov, _ = model

			# https://en.wikipedia.org/wiki/Principal_axis_theorem
			eigenvals, eigenvecs = np.linalg.eigh(cov)

			chisquare_val = (scipy.stats.chi2.ppf(0.999, cov.shape[0]))
			print(chisquare_val,cov.shape[0])
	
			A = np.dot(np.dot(eigenvecs, np.diag(np.sqrt(chisquare_val*eigenvals))), eigenvecs.T)


			point = X[y == classes[i]] - m
			# http://math.stackexchange.com/questions/1403126/what-is-the-general-equation-equation-for-rotated-ellipsoid
			# ellipsoid equation : (m-x)'A(m-x) = 1
			dists = np.sum(np.dot(point, A) * point, axis=1)
			print(m)
			print(X[y == classes[i]].sum(0)/float(X[y == classes[i]].shape[0]))
			exit()
			print(X[y == classes[i]][dists > chisquare_val], dists.max())
			#print(np.sum(dists > chisquare_val))
			#exit()
			self.outliers_ = np.hstack((self.outliers_, np.where(y == classes[i])[0][np.where(dists <= (1))[0]]))

		return self

class ThresholdOutlierRemover(OutlierRemover):
	"""docstring for ThresholdOutlierRemover"""
	def __init__(self):
		super(ThresholdOutlierRemover, self).__init__()

	def remove_outliers(self, X, y):

		classes = np.unique(y)

		n_classes = len(classes)

		n_estimators = int(X.shape[1]/n_classes)

		Xt = X.reshape((X.shape[0], n_estimators, n_classes))

		yt = np.repeat(y, n_estimators).reshape((len(y), n_estimators))

		rate = (yt == classes.take(np.argmax(Xt, axis=2))).sum(1)

		return np.where(rate > 0.0)[0]

	def fit(self, X, y):
		self.outliers_ = self.remove_outliers(X, y)

		return self
		