from sklearn.neighbors import KNeighborsClassifier as kNN

from sklearn.ensemble import RandomForestClassifier as ForestClassifier
from sklearn.ensemble import ExtraTreesClassifier as ExtraTreesClassifier

from sklearn.base import BaseEstimator, ClassifierMixin



import numpy as np

import multiprocessing as mp

from math import ceil

class LazyNNRF(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 n_neighbors=30,
                 n_estimators=200,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        
        
        self.kNN = kNN(n_jobs=n_jobs, n_neighbors=n_neighbors, weights='distance', algorithm='brute', metric='cosine')

        # everyone's params 
        self.n_jobs = n_jobs

        # kNN params
        self.n_neighbors = n_neighbors
        
        # ForestBase params
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.bootstrap=bootstrap
        self.oob_score=oob_score
        self.random_state=random_state
        self.verbose=verbose
        self.warm_start=warm_start
        self.class_weight=class_weight


    def fit(self, X, y, sample_weight=None):
        """Build a forest of trees from the training set (X, y).
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.
        Returns
        -------
        self : object
            Returns self.
        """
        self.X_train = X
        self.y_train = y

        self.kNN.fit(self.X_train, self.y_train)
        
        return self

    def runForests(self, X, idx, q, p):
        pred = []
        for i,ids in enumerate(idx):
            rf = ForestClassifier(n_estimators=self.n_estimators,
                                 criterion=self.criterion,
                                 max_depth=self.max_depth,
                                 min_samples_split=self.min_samples_split,
                                 min_samples_leaf=self.min_samples_leaf,
                                 min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                 max_features=self.max_features,
                                 max_leaf_nodes=self.max_leaf_nodes)

            rf.fit(self.X_train[ids], self.y_train[ids])
            pred = pred + [rf.predict(X[i])[0]]

        q.put((p, pred))
        return

    def predict(self, X):
        """Predict class for X.
        The predicted class of an input sample is computed as the majority
        prediction of the trees in the forest.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes.
        """
        # get knn for all test sample
        idx = self.kNN.kneighbors(X, return_distance=False)
        
        jobs = []
        q = mp.Queue() 
        length = len(idx)
        chunk_size = int(ceil(length/float(self.n_jobs)))

        # Run processes
        for p in xrange(1, self.n_jobs + 1):
            s = (p-1)*chunk_size
            e = p*chunk_size if p*chunk_size <= length else length
            process = mp.Process(target=self.runForests, args=(X[s:e],idx[s:e],q, p,))
            jobs.append(process)
            process.start()

        # Exit the completed processes
        for p in jobs:
            p.join()
        
        # Get process results from the output queue
        results = [q.get() for p in jobs]

        # make sure that it retrieves results in the correct order
        results.sort()
        pred = []
        for r in results:
            pred = pred + r[1]

        return np.array(pred)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

class LazyNNExtraTrees(LazyNNRF):
    def runForests(self, X, idx, q, p):
        pred = []
        for i,ids in enumerate(idx):
            rf = ExtraTreesClassifier(n_estimators=self.n_estimators,
                                 criterion=self.criterion,
                                 max_depth=self.max_depth,
                                 min_samples_split=self.min_samples_split,
                                 min_samples_leaf=self.min_samples_leaf,
                                 min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                 max_features=self.max_features,
                                 max_leaf_nodes=self.max_leaf_nodes)

            rf.fit(self.X_train[ids], self.y_train[ids])
            pred = pred + [rf.predict(X[i])[0]]

        q.put((p, pred))
        return
