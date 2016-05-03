from sklearn import svm

class LinearSVM(svm.LinearSVC):
	
	def predict_proba(self, X):
		return self._predict_proba_lr(X)