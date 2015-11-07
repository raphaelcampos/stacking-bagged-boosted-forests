import numpy as np
import ctypes
from ctypes import *

import scipy

from sklearn.datasets import fetch_20newsgroups, load_svmlight_file
from sklearn.cross_validation import train_test_split

# Example reference
#http://bikulov.org/blog/2013/10/01/using-cuda-c-plus-plus-functions-in-python-via-star-dot-so-and-ctypes/

# extract cuda_sum function pointer in the shared object cuda_sum.so
def get_cuda_sum():
    dll = ctypes.CDLL('./cuda_sum.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.cuda_sum
    func.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), c_size_t]
    return func

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

def get_make_inverted_index():
    dll = ctypes.CDLL('./gtknn.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.make_inverted_index
    func.argtypes = [c_int, c_int, POINTER(Entry), c_int]
    func.restype = InvertedIndex
    return func

def get_KNN():
    dll = ctypes.CDLL('./gtknn.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.KNN
    func.argtypes = [InvertedIndex, POINTER(Entry), c_int, c_int]
    func.restype = POINTER(cuSimilarity)
    return func

# create __cuda_sum function with get_cuda_sum()
__cuda_sum = get_cuda_sum()

__make_inverted_index = get_make_inverted_index()

__KNN = get_KNN()

# convenient python wrapper for __cuda_sum
# it does all job with types convertation
# from python ones to C++ ones 
def cuda_sum(a, b, c, size):
    a_p = a.ctypes.data_as(POINTER(c_float))
    b_p = b.ctypes.data_as(POINTER(c_float))
    c_p = c.ctypes.data_as(POINTER(c_float))

    __cuda_sum(a_p, b_p, c_p, size)

# testing, sum of two arrays of ones and output head part of resulting array
if __name__ == '__main__':
    X, y = load_svmlight_file("../release/datasets/reuters90.svm")
    print "loadded"
    entries = []
    num_docs = X.shape[0]
    num_terms = X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10, random_state=42)

    cx = scipy.sparse.coo_matrix(X_train)
    
    entries = [Entry(i, j, int(v), 0) for i,j,v in zip(cx.row, cx.col, cx.data)]

    qx = scipy.sparse.coo_matrix(X_test[0])

    query = [Entry(i, j, int(v), 0) for i,j,v in zip(qx.row, qx.col, qx.data)]

    print "entries..."
    #entries = np.array(entries).astype(Entry)
    arr = (type(entries[0])*len(entries))(*entries)
    #for a in arr:
    #    print a.doc_id, 
    print "inverted_idx..."
    inverted_idx = __make_inverted_index(num_docs, num_terms, arr, len(entries))

    k = 30
    similarities = __KNN(inverted_idx, (type(query[0])*len(query))(*query), k, len(query))
    
    #print (cuSimilarity*k)(*similarities)
    #
    arr = np.array(similarities[:k], dtype=cuSimilarity._fields_)
    print arr['distance']
    exit()
    for i in range(k):
       print y_train[similarities[i].doc_id], y_test[0]
    #size=int(1024*1024*1)

    #a = np.ones(size).astype('float32')
    #b = np.ones(size).astype('float32')
    #c = np.zeros(size).astype('float32')

    #cuda_sum(a, b, c, size)

    #print c[:10]
