from sklearn.neighbors import NearestNeighbors as kNN

from sklearn.ensemble import RandomForestClassifier as ForestClassifier
from sklearn.ensemble import ExtraTreesClassifier as ExtraTreesClassifier
from ..ensemble import Broof, Bert
from ..neighbors import cuKNeighborsSparseClassifier

import time

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array

import numpy as np

import scipy

import multiprocessing as mp

from math import ceil

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_selection.base import SelectorMixin
from scipy.sparse import vstack, hstack
from sklearn.utils import check_array, check_random_state
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils.validation import check_is_fitted

class ReduceFeatureSpace(BaseEstimator, SelectorMixin):
    """Feature selector that removes all low-variance features.
    This feature selection algorithm looks only at the features (X), not the
    desired outputs (y), and can thus be used for unsupervised learning.
    Read more in the :ref:`User Guide <variance_threshold>`.
    Parameters
    ----------
    threshold : float, optional
        Features with a training-set variance lower than this threshold will
        be removed. The default is to keep all features with non-zero variance,
        i.e. remove the features that have the same value in all samples.
    Attributes
    ----------
    variances_ : array, shape (n_features,)
        Variances of individual features.
    Examples
    --------
    The following dataset has integer features, two of which are the same
    in every sample. These are removed with the default setting for threshold::
        >>> X = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]
        >>> selector = VarianceThreshold()
        >>> selector.fit_transform(X)
        array([[2, 0],
               [1, 4],
               [1, 1]])
    """

    def __init__(self, threshold=0.):
        self.threshold = threshold

    def fit(self, X, y=None):
        """Learn empirical variances from X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Sample vectors from which to compute variances.
        y : any
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.
        Returns
        -------
        self
        """
        X = check_array(X, ('csr', 'csc'), dtype=np.float64)

        if hasattr(X, "toarray"):   # sparse matrix
            #_, self.variances_ = mean_variance_axis(X, axis=0)
            self.variances_ = self._get_nonzero_columns(X)
        else:
            self.variances_ = np.var(X, axis=0)

        return self

    def _get_support_mask(self):
        check_is_fitted(self, 'variances_')

        return self.variances_ > self.threshold

    def _get_nonzero_columns(self, X):
        n_samples = X.shape[0]
        n_features = X.shape[1]

        X_data = np.asarray(X.data, dtype=np.float64)     # might copy!
        X_indices = X.indices

        non_zero = X_indices.shape[0]
        
        selected_features = np.zeros(n_features, dtype=np.float64)

        for i in xrange(non_zero):
            col_ind = X_indices[i]
            selected_features[col_ind] += int(not X_data[i] == 0)

        return selected_features

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
                 class_weight=None,
                 n_gpus=1):
        

        # everyone's params 
        self.n_jobs = n_jobs

        # kNN params
        self.n_neighbors = n_neighbors
        self.n_gpus = n_gpus
        
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

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        self.X_train = X
        self.y_train = y

        if self.n_gpus > 0:
            self.kNN = cuKNeighborsSparseClassifier(n_neighbors=self.n_neighbors, n_gpus=self.n_gpus)
        else:
            self.kNN = kNN(n_jobs=self.n_jobs, n_neighbors=self.n_neighbors, algorithm='brute', metric='cosine')

        self.kNN.fit(self.X_train, self.y_train)
        
        return self

    def runForests(self, X, idx, q, p):
        pred = np.zeros((len(idx), self.n_classes_))

        selector = ReduceFeatureSpace() 
        for i,ids in enumerate(idx):
            ids = ids[np.logical_and(ids < self.X_train.shape[0], ids >= 0)]
            X_t = selector.fit_transform(vstack((self.X_train[ids],X[i])))

            X_t, X_i = X_t[:len(ids)], X_t[len(ids):]
            #X_t, X_i = (self.X_train[ids],X[i])
            density = X_t.nnz/float(X_t.shape[0])
            rf = ForestClassifier(n_estimators=self.n_estimators,
                                 criterion=self.criterion,
                                 max_depth=self.max_depth,
                                 min_samples_split=self.min_samples_split,
                                 min_samples_leaf=self.min_samples_leaf,
                                 min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                 max_features=self.max_features,
                                 max_leaf_nodes=self.max_leaf_nodes,
                                 random_state=self.random_state)

            rf.fit(X_t, self.y_train[ids])
            pred[i, np.searchsorted(self.classes_, rf.classes_)] = rf.predict_proba(X_i)[0]

        q.put((p, pred))
        return

    def predict_proba(self, X):
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

        results = []
        liveprocs = list(jobs)
        while liveprocs:
            try:
                while 1:
                    results = results + [(q.get(False))]
                    #print results
            except(Exception, e):
                pass

            time.sleep(0.005)    # Give tasks a chance to put more data in
            if not q.empty():
                continue
            liveprocs = [p for p in liveprocs if p.is_alive()]

        # make sure that it retrieves results in the correct order
        results.sort()
        pred = np.zeros((X.shape[0], self.n_classes_))
        for r in results:
            p = r[0]
            s = (p-1)*chunk_size
            e = p*chunk_size if p*chunk_size <= length else length
            pred[s:e,:] = r[1] 
            
        return pred

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
        pred = self.predict_proba(X)
        return self.classes_.take(np.argmax(pred, axis=1), axis=0)


    def score(self, X, y):
        return np.mean(self.predict(X) == y)


class LazyNNExtraTrees(LazyNNRF):
    def runForests(self, X, idx, q, p):
        pred = np.zeros((len(idx), self.n_classes_))

        selector = ReduceFeatureSpace() 
        for i,ids in enumerate(idx):
            ids = ids[np.logical_and(ids < self.X_train.shape[0], ids >= 0)]
            X_t = selector.fit_transform(vstack((self.X_train[ids],X[i])))

            X_t, X_i = X_t[:len(ids)], X_t[len(ids):]
            #X_t, X_i = (self.X_train[ids],X[i])
            density = X_t.nnz/float(X_t.shape[0])
            rf = ExtraTreesClassifier(n_estimators=self.n_estimators,
                                 criterion=self.criterion,
                                 max_depth=self.max_depth,
                                 min_samples_split=self.min_samples_split,
                                 min_samples_leaf=self.min_samples_leaf,
                                 min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                 max_features=self.max_features,
                                 max_leaf_nodes=self.max_leaf_nodes,
                                 random_state=self.random_state)

            rf.fit(X_t, self.y_train[ids])
            pred[i, np.searchsorted(self.classes_, rf.classes_)] = rf.predict_proba(X_i)[0]

        q.put((p, pred))
        return

class LazyNNBroof(LazyNNRF):
    def __init__(self,
                 n_iterations=200,
                 learning_rate=1,
                 n_neighbors=30,
                 n_estimators=5,
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
                 class_weight=None,
                 n_gpus=1):

        super(LazyNNBroof, self).__init__(n_neighbors=n_neighbors,
                                        n_estimators=n_estimators,
                                        criterion=criterion,
                                        max_depth=max_depth,
                                        min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf,
                                        min_weight_fraction_leaf=min_weight_fraction_leaf,
                                        max_features=max_features,
                                        max_leaf_nodes=max_leaf_nodes,
                                        bootstrap=bootstrap,
                                        oob_score=oob_score,
                                        n_jobs=n_jobs,
                                        random_state=random_state,
                                        verbose=verbose,
                                        warm_start=warm_start,
                                        class_weight=class_weight,
                                        n_gpus=n_gpus)

        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        # everyone's params 
        self.n_jobs = n_jobs

        # kNN params
        self.n_neighbors = n_neighbors
        self.n_gpus = n_gpus
        
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

    def runForests(self, X, idx, q, p):
        random_guesses = 0 
        pred = np.zeros((len(idx), self.n_classes_))

        selector = ReduceFeatureSpace() 
        for i,ids in enumerate(idx):
            ids = ids[np.logical_and(ids < self.X_train.shape[0], ids >= 0)]
            X_t = selector.fit_transform(vstack((self.X_train[ids],X[i])))

            X_t, X_i = X_t[:len(ids)], X_t[len(ids):]
            #X_t, X_i = (self.X_train[ids],X[i])
            density = X_t.nnz/float(X_t.shape[0])
            rf = Broof(n_trees=self.n_estimators,
                        n_iterations=self.n_iterations,
                        learning_rate=self.learning_rate,
                        criterion=self.criterion,
                        max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split,
                        min_samples_leaf=self.min_samples_leaf,
                        min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                        max_features=self.max_features,
                        max_leaf_nodes=self.max_leaf_nodes,
                        random_state=self.random_state)
            try:
                rf.fit(X_t, self.y_train[ids])
                pred[i, np.searchsorted(self.classes_, rf.classes_)] = rf.predict_proba(X_i)[0]
            except(Exception, e):
                # ignore adaboost worse than random guess
                random_instance = check_random_state(self.random_state)
                pred[i, random_instance.randint(self.n_classes_, size=1)] = 1;
                random_guesses = random_guesses + 1
                pass

        print("Random guesses:", random_guesses)
        q.put((p, pred))
        return
