from sklearn.neighbors import NearestNeighbors as kNN

from sklearn.ensemble import RandomForestClassifier as ForestClassifier
from sklearn.ensemble import ExtraTreesClassifier as ExtraTreesClassifier

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array

import numpy as np

import scipy


import multiprocessing as mp

from math import ceil

import ctypes
from ctypes import *

VALID_METRICS = ['cosine']

class Entry(Structure):
    _fields_ = [
       ("doc_id", c_int),
       ("term_id", c_int),
       ("tf", c_int),
       ("tf_idf", c_float)]

    def __init__(self, doc_id, term_id, tf, tf_idf):
        self.doc_id = doc_id
        self.term_id = term_id
        self.tf = tf
        self.tf_idf = tf_idf 

class cuSimilarity(Structure):
    _fields_ = [
       ("doc_id", c_int),
       ("distance", c_float)]

class InvertedIndex(Structure):
    _fields_ = [
       ("d_index", POINTER(c_int)),
       ("d_count", POINTER(c_int)),
       ("d_inverted_index", POINTER(Entry)),
       ("d_norms", POINTER(c_float)),
       ("d_normsl1", POINTER(c_float)),
       ("num_docs", c_int),
       ("num_entries", c_int),
       ("num_terms", c_int)]

class cuKNeighborsSparseClassifier(object):

    def __init__(self, n_neighbors=30, metric='cosine'):
        
        super(cuKNeighborsSparseClassifier, self).__init__()

        self._init_params(n_neighbors=n_neighbors, metric=metric)

        self._init_methods()

    def _init_params(self, n_neighbors=None, metric='cosine'):
        
        self.n_neighbors = n_neighbors
        self.metric = metric


    def _init_methods(self):
        dll = ctypes.CDLL('./gtknn.so', mode=ctypes.RTLD_GLOBAL)

        func = dll.make_inverted_index
        func.argtypes = [c_int, c_int, POINTER(Entry), c_int]
        func.restype = InvertedIndex

        self._make_inverted_index = func

        func = dll.KNN
        func.argtypes = [InvertedIndex, POINTER(Entry), c_int, c_int]
        func.restype = POINTER(cuSimilarity)

        self._KNN = func


    def _get_entries(self, X):
        cx = scipy.sparse.coo_matrix(X)
        return [Entry(i, j, int(v), 0) for i,j,v in zip(cx.row, cx.col, cx.data)]

    def fit(self, X, y):

        entries = self._get_entries(X) 
        
        num_docs = X.shape[0]
        num_terms = X.shape[1]

        self.inverted_idx = self._make_inverted_index(num_docs, num_terms, (type(entries[0])*len(entries))(*entries), len(entries))



    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        """Finds the K-neighbors of a point.
        Returns indices of and distances to the neighbors of each point.
        Parameters
        ----------
        X : array-like, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            The query point or points.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.
        n_neighbors : int
            Number of neighbors to get (default is the value
            passed to the constructor).
        return_distance : boolean, optional. Defaults to True.
            If False, distances will not be returned
        Returns
        -------
        dist : array
            Array representing the lengths to points, only present if
            return_distance=True
        ind : array
            Indices of the nearest points in the population matrix.
        Examples
        --------
        In the following example, we construct a NeighborsClassifier
        class from an array representing our data set and ask who's
        the closest point to [1,1,1]
        >>> samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
        >>> from sklearn.neighbors import NearestNeighbors
        >>> neigh = NearestNeighbors(n_neighbors=1)
        >>> neigh.fit(samples) # doctest: +ELLIPSIS
        NearestNeighbors(algorithm='auto', leaf_size=30, ...)
        >>> print(neigh.kneighbors([[1., 1., 1.]])) # doctest: +ELLIPSIS
        (array([[ 0.5]]), array([[2]]...))
        As you can see, it returns [[0.5]], and [[2]], which means that the
        element is at distance 0.5 and is the third element of samples
        (indexes start at 0). You can also query for multiple points:
        >>> X = [[0., 1., 0.], [1., 0., 1.]]
        >>> neigh.kneighbors(X, return_distance=False) # doctest: +ELLIPSIS
        array([[1],
               [2]]...)
        """
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        X = check_array(X, accept_sparse='csr')

        #train_size = self._fit_X.shape[0]
        #if n_neighbors > train_size:
        #    raise ValueError(
        #        "Expected n_neighbors <= n_samples, "
        #        " but n_samples = %d, n_neighbors = %d" %
        #        (train_size, n_neighbors)
        #    )

        n_samples, _ = X.shape
        sample_range = np.arange(n_samples)[:, None]

        neigh_ind = None
        for x in X:
            query = self._get_entries(x)
            similarities = self._KNN(self.inverted_idx, (type(query[0])*len(query))(*query), n_neighbors, len(query))


            idxs = np.array(similarities[:n_neighbors], dtype=cuSimilarity._fields_)['doc_id']

            if neigh_ind == None:
                neigh_ind = np.array(idxs)
            else:
                neigh_ind = np.vstack((neigh_ind, idxs))

        return neigh_ind
        


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
