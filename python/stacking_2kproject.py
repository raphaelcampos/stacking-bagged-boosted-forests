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

def get_elem(elems, n):
	i = 0
	leftmost = -1
	tmp = []
	while i < len(elems):
		if (1 << i) & int(n):
			leftmost = leftmost + 1
			tmp.append(i)
		i = i + 1

	return tmp , leftmost
		

def all_sets(elems):
	n = len(elems)

	numbers = np.zeros(2**n - 1)
	numbers[:n] = 2**np.arange(n)

	last = n
	sets = []
	for num in numbers:
		s, leftmost_biton = get_elem(elems, num)
		sets.append(s)
		for i in range(leftmost_biton + 1, n):
			if num < numbers[i]:
				numbers[last] = num + numbers[i]
				last = last + 1
		
	return sets

def init_coef(Q, i, n):
	if i > n:
		return

	n, _ = Q.shape 
	mid = (n/2)
	Q[:mid,i] = -1

	init_coef(Q[:mid], i + 1, n)
	init_coef(Q[mid:], i + 1, n)



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

		i = np.arange(9)[:, np.newaxis]
		n_classes = len(np.unique(y_train))
		feats = (n_classes*i + np.arange(n_classes)).ravel()
		
		return X_train.toarray()[:,feats], X_test.toarray()[:,feats], y_train, y_test

	def run(self, args):
		X_train, X_test, y_train, y_test = self._load_dataset(args)

		k = 0
		estimator, tuned_parameters = self._setup_instantiator(args)

		folds_time = []
		self.folds_predict = []
		self.folds_macro = []
		self.folds_micro = []

		print(estimator.get_params(deep=False))

		output, args.output = args.output, ""

		from xsklearn.ensemble import GaussianOutlierRemover, ThresholdOutlierRemover

		#out_remov = ThresoldOutlierRemover()
		#X_train, y_train = out_remov.fit_transform(X_train, y_train)

		tf_transformer = TfidfTransformer(norm=args.norm, use_idf=False,
											 smooth_idf=False, sublinear_tf=False)
		if self._tfidf(args):
			# Learn the idf vector from training set
			tf_transformer.fit(X_train)
			# Transform test and training frequency matrix
			# based on training idf vector		
			X_train = tf_transformer.transform(X_train)
			X_test = tf_transformer.transform(X_test)

		id_estimators = np.asarray([2, 3, 4, 5, 6, 8])

		n_estimators = id_estimators.shape[0]
		sets =  all_sets(np.arange(n_estimators))
		for i in sets:
			print(i)
			i = id_estimators[np.asarray(i)][:, np.newaxis]
			n_classes = len(np.unique(y_train))
			feats = (n_classes*i + np.arange(n_classes)).ravel()

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
			e.fit(X_train[:, feats], y_train)
			pred = e.predict(X_test[:, feats])
			end = time.time()

			if len(i) == 1:
				pred = np.unique(y_train).take(np.argmax(X_test[:, feats], axis=1), axis=0)

			#fi = e.feature_importances_.reshape((9,7)).T
			#print(fi/(fi.sum(1))[:, np.newaxis])

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

		Qs = np.ones((2**n_estimators, 2**n_estimators))
		#init_coef(Qs, 1, n_estimators + 1)
		
		Qs[:, 1:(n_estimators+1)] = -1
		for i in range(2**n_estimators - 1):
			Qs[i + 1, np.asarray(sets[i]) + 1] = 1

		for i in range(n_estimators, 2**n_estimators - 1):
			Qs[:, i + 1] = Qs[:, np.asarray(sets[i]) + 1].prod(1) 


		
		w = np.dot(Qs[:, :].T, np.vstack((0.0, np.asarray(self.folds_macro)[:, np.newaxis])))/Qs[:, :].shape[0]
		#w = np.dot(Qs[1:, :].T, np.asarray(self.folds_micro)[:, np.newaxis])/Qs[1:, :].shape[0]
		
		SS = (2**n_estimators * w[1:]**2)

		print(SS/SS.sum()).T

		SS = SS/SS.sum()
		labels = np.asarray(["broof","lazy","bert","lxt","svm","nb","knn","rf","xt"])


		print("Threshold: %f" % ((1./Qs[1:, :].shape[0]) + np.std(SS)))
		ids, _ = np.where(SS >= 1./Qs[1:, :].shape[0])
		for i in ids:
			print(labels[id_estimators[sets[i]]], round(100*SS[i][0], 2))


		if not output == "":
			try:
				fil = np.load(output)
				mic = np.hstack((fil['micro'], np.asarray(self.folds_micro)[:,np.newaxis]))
				np.savez(output, Q=Qs, labels=labels[id_estimators], sets=sets, micro=mic)
			except(Exception):
				np.savez(output, Q=Qs, labels=labels[id_estimators], sets=sets, micro=np.asarray(self.folds_micro)[:,np.newaxis])

		print("F1-Score")
		print("\tMicro: ", np.average(self.folds_micro), np.std(self.folds_micro))
		print("\tMacro: ", np.average(self.folds_macro), np.std(self.folds_macro))

		print('loading time : ', self.datasetLoadingTime)
		print('times : ', np.average(folds_time), np.std(folds_time))



fil = np.load("4uni_all_estimators.npz")


micro = np.vstack((np.zeros(fil['micro'].shape[1]),fil['micro']*100))
print(micro)

sets = fil['sets']
labels = fil['labels']
Qs = fil['Q']

k, r = np.log2(micro.shape[0]), micro.shape[1]

micro[1:] = (micro[1:]-micro[1:].min(0))
avg = micro.mean(1)[:, np.newaxis]

eij = (micro - avg)


w = np.dot(Qs.T, avg)/Qs[:, :].shape[0]

SSE = (eij**2).sum()

SSY = (micro**2).sum()

SS_ = (2**k)*r * w**2
SST = SSY - SS_[0]

print(100*SSE/SST)
print(100*SS_[1:]/SST)
print(SSE, SSY, SS_[0], k, r)

s_e = np.sqrt(SSE/((2**k)*(r-1)))
s_qi = s_e/np.sqrt(((2**k)*(r)))

import scipy as sp

t = sp.stats.t.ppf(1-(0.05/2),(2**k)*(r-1))*np.array([-1,1])

print(s_e, s_qi, t)

print(w[-10:-1] + t*s_qi)
print(SS_[1:]/SST).T

imp = (SS_[1:]/SST)
print(labels)
print("Threshold: %f" % np.median(imp))

ids, _ = np.where(imp > np.median(imp))
for i in ids:
	print(labels[sets[i]], i, imp[i], avg[i+1])



print(ids.shape)
exit()


app = TextClassification2App()

app.run(app.parse_arguments())