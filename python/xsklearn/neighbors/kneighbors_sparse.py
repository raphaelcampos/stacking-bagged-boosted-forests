from sklearn.utils import check_array
from sklearn.neighbors import NearestNeighbors as kNN

import numpy as np

import ctypes
from ctypes import *

import os.path

VALID_METRICS = ['cosine']

class Entry(Structure):
    _fields_ = [
       ("doc_id", c_int),
       ("term_id", c_int),
       ("tf", c_int),
       ("tf_idf", c_float)]

    def __init__(self, doc_id, term_id, tf, tf_idf):
        self.set(doc_id, term_id, tf, tf_idf)

    def set(self, doc_id, term_id, tf, tf_idf):
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
       ("num_terms", c_int),
       ("device_id", c_int)]

class cuKNeighborsSparseClassifier(object):

    def __init__(self, n_neighbors=30, metric='cosine', n_gpus=1):
        
        super(cuKNeighborsSparseClassifier, self).__init__()

        self._init_params(n_neighbors=n_neighbors, metric=metric)

        self._init_methods()

        self.n_gpus = n_gpus

    def __del__(self):
        if hasattr(self, 'inverted_idx'):
            self._free_inverted_indexes(self.inverted_idx, self.n_gpus)

    def _init_params(self, n_neighbors=None, metric='cosine'):
        
        self.n_neighbors = n_neighbors
        self.metric = metric


    def _init_methods(self):
        dll_name = "gtknn.so"
        dll = ctypes.CDLL(os.path.dirname(os.path.abspath(__file__))
                             + os.path.sep + dll_name)

        func = dll.make_inverted_index
        func.argtypes = [c_int, c_int, POINTER(Entry), c_int]
        func.restype = InvertedIndex

        func = dll.make_inverted_indices
        func.argtypes = [c_int, c_int, POINTER(Entry), c_int, c_int]
        func.restype = POINTER(InvertedIndex)

        self._make_inverted_index = func

        func = dll.csr_make_inverted_indices
        func.argtypes = [c_int, c_int, POINTER(c_float), POINTER(c_int), POINTER(c_int), c_int, c_int, c_int]
        func.restype = POINTER(InvertedIndex)

        self._csr_make_inverted_indices = func

        func = dll.KNN
        func.argtypes = [InvertedIndex, POINTER(Entry), c_int, c_int]
        func.restype = POINTER(cuSimilarity)

        self._KNN = func

        func = dll.kneighbors
        func.argtypes = [POINTER(InvertedIndex), c_int, POINTER(c_float), POINTER(c_int),POINTER(c_int), c_int, c_int, c_int]
        func.restype = POINTER(POINTER(c_int))

        self._kneighbors = func

        func = dll.device_infos
        func.restype = c_int

        self._device_infos = func

        func = dll.freeInvertedIndexes
        func.argtypes = [POINTER(InvertedIndex), c_int]

        self._free_inverted_indexes = func
        
    def fit(self, X, y):

        X = check_array(X, accept_sparse='csr')
     
        num_docs = X.shape[0]
        num_terms = X.shape[1]

        self.y = y
        
        self.inverted_idx = self._csr_make_inverted_indices(num_docs, num_terms,
                                             (c_float*X.nnz)(*X.data),
                                             (c_int*len(X.indices))(*X.indices),
                                             (c_int*len(X.indptr))(*X.indptr),
                                             X.nnz, len(X.indptr), self.n_gpus)

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

        idxs = self._kneighbors(self.inverted_idx, n_neighbors,
                                     (c_float*X.nnz)(*X.data),
                                     (c_int*len(X.indices))(*X.indices),
                                     (c_int*len(X.indptr))(*X.indptr),
                                     X.nnz, len(X.indptr), self.n_gpus)
        
        return np.ctypeslib.as_array(idxs, shape=(len(X.indptr)-1,n_neighbors))
        

    def predict(self, X):
        """Predict the class labels for the provided data
        Parameters
        ----------
        X : array-like, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            Test samples.
        Returns
        -------
        y : array of shape [n_samples] or [n_samples, n_outputs]
            Class labels for each data sample.
        """
        X = check_array(X, accept_sparse='csr')

        neigh_ind = self.kneighbors(X)
        
        from scipy import stats
        classes_ = np.unique(self.y)
        _y = self.y

        _y = self.y.reshape((-1, 1))
        classes_ = [classes_]

        n_outputs = len(classes_)
        n_samples = X.shape[0]
        

        y_pred = np.empty((n_samples, n_outputs), dtype=classes_[0].dtype)
        for k, classes_k in enumerate(classes_):
            mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
            
            mode = np.asarray(mode.ravel(), dtype=np.intp)
            y_pred[:, k] = classes_k.take(mode)

        return y_pred.T[0]