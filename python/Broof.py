
import warnings
from warnings import warn

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier as ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import six
from sklearn.ensemble.base import _partition_estimators
from sklearn.tree._tree import DTYPE, DOUBLE
import numpy as np

from sklearn.externals.joblib import Parallel, delayed

from sklearn.utils import check_random_state, check_array, compute_sample_weight
from sklearn.utils.fixes import bincount

def _generate_sample_indices(random_state, n_samples):
    """Private function used to _parallel_build_trees function."""
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples)

    return sample_indices

def _generate_unsampled_indices(random_state, n_samples):
    """Private function used to forest._set_oob_score fuction."""
    sample_indices = _generate_sample_indices(random_state, n_samples)
    sample_counts = bincount(sample_indices, minlength=n_samples)
    unsampled_mask = sample_counts == 0
    indices_range = np.arange(n_samples)
    unsampled_indices = indices_range[unsampled_mask]

    return unsampled_indices

def _parallel_helper(obj, methodname, *args, **kwargs):
    """Private helper to workaround Python 2 pickle limitations"""
    return getattr(obj, methodname)(*args, **kwargs)


class BoostedRandomForestClassifier(RandomForestClassifier):
    """A random forest classifier.
    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various sub-samples of the dataset and use averaging to
    improve the predictive accuracy and control over-fitting.
    The sub-sample size is always the same as the original
    input sample size but the samples are drawn with replacement if
    `bootstrap=True` (default).
    Read more in the :ref:`User Guide <forest>`.
    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.
    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
        Note: this parameter is tree-specific.
    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
        Note: this parameter is tree-specific.
    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        Ignored if ``max_leaf_nodes`` is not None.
        Note: this parameter is tree-specific.
    min_samples_split : integer, optional (default=2)
        The minimum number of samples required to split an internal node.
        Note: this parameter is tree-specific.
    min_samples_leaf : integer, optional (default=1)
        The minimum number of samples in newly created leaves.  A split is
        discarded if after the split, one of the leaves would contain less then
        ``min_samples_leaf`` samples.
        Note: this parameter is tree-specific.
    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the input samples required to be at a
        leaf node.
        Note: this parameter is tree-specific.
    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
        If not None then ``max_depth`` will be ignored.
        Note: this parameter is tree-specific.
    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.
    oob_score : bool
        Whether to use out-of-bag samples to estimate
        the generalization error.
    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.
    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.
    class_weight : dict, list of dicts, "balanced", "balanced_subsample" or None, optional
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``
        The "balanced_subsample" mode is the same as "balanced" except that weights are
        computed based on the bootstrap sample for every tree grown.
        For multi-output, the weights of each column of y will be multiplied.
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.
    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.
    classes_ : array of shape = [n_classes] or a list of such arrays
        The classes labels (single output problem), or a list of arrays of
        class labels (multi-output problem).
    n_classes_ : int or list
        The number of classes (single output problem), or a list containing the
        number of classes for each output (multi-output problem).
    n_features_ : int
        The number of features when ``fit`` is performed.
    n_outputs_ : int
        The number of outputs when ``fit`` is performed.
    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).
    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
    oob_decision_function_ : array of shape = [n_samples, n_classes]
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN.
    References
    ----------
    .. [1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.
    See also
    --------
    DecisionTreeClassifier, ExtraTreesClassifier
    """


    def fit(self, X, y, sample_weight=None):

        self.oob_score = True

        super(BoostedRandomForestClassifier, self).fit(X, y, sample_weight)

        if self.oob_score:

            y = np.atleast_1d(y)
            if y.ndim == 2 and y.shape[1] == 1:
                warn("A column-vector y was passed when a 1d array was"
                     " expected. Please change the shape of y to "
                     "(n_samples,), for example using ravel().",
                     DataConversionWarning, stacklevel=2)

            if y.ndim == 1:
                # reshape is necessary to preserve the data contiguity against vs
                # [:, np.newaxis] that does not.
                y = np.reshape(y, (-1, 1))

            #y, expanded_class_weight = self._validate_y_class_weight(y)
            self.n_classes_ = [self.n_classes_]
            self._set_oob_score(X, y, sample_weight)


    def _set_oob_score(self, X, y, sample_weight=None):

        if sample_weight == None:
            return

        """Compute out-of-bag score"""
        X = check_array(X, dtype=DTYPE, accept_sparse='csr')

        n_classes_ = self.n_classes_

        n_samples = y.shape[0]

        oob_decision_function = []
        oob_score = 0.0
        predictions = []
        oob_err = []
        oob_ = []

        sample_weight_tmp = sample_weight

        for k in range(self.n_outputs_):
            predictions.append(np.zeros((n_samples, n_classes_[k])))
            oob_err.append(np.ones(len(self.estimators_)))

        for i, estimator in enumerate(self.estimators_):
            unsampled_indices = _generate_unsampled_indices(
                estimator.random_state, n_samples)
            p_estimator = estimator.predict_proba(X[unsampled_indices, :],
                                                  check_input=False)

            if self.n_outputs_ == 1:
                p_estimator = [p_estimator]

            for k in range(self.n_outputs_):
                predictions[k][unsampled_indices, :] += p_estimator[k]
                #np.mean(
                #    np.average(incorrect, weights=sample_weight[unsampled_indices], axis=0))
                incorrect = y[unsampled_indices, k] != np.argmax(p_estimator[k], axis=1)
                estimator_error = np.average(incorrect, weights=sample_weight[unsampled_indices], axis=0)
                
                estimator_weight = (
                    np.log((1. - estimator_error) / estimator_error))

                oob_err[k][i] = estimator_error
                sample_weight_tmp[unsampled_indices] *= np.exp(estimator_weight * (2*incorrect - 1))
            
            oob_.append(unsampled_indices)

        for k in range(self.n_outputs_):
            if (predictions[k].sum(axis=1) == 0).any():
                warn("Some inputs do not have OOB scores. "
                     "This probably means too few trees were used "
                     "to compute any reliable oob estimates.")

            decision = (predictions[k] /
                        predictions[k].sum(axis=1)[:, np.newaxis])
            oob_decision_function.append(decision)
            oob_score += np.mean(y[:, k] ==
                                 np.argmax(predictions[k], axis=1), axis=0)

        if self.n_outputs_ == 1:
            self.oob_decision_function_ = oob_decision_function[0]
            self.oob_err_ = oob_err[0]
        else:
            self.oob_decision_function_ = oob_decision_function
            self.oob_err_ = oob_err

        sample_weight = sample_weight_tmp 

        self.oob_samples_ = oob_
        self.oob_score_ = oob_score / self.n_outputs_

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the mean predicted class probabilities of the trees in the forest. The
        class probability of a single tree is the fraction of samples of the same
        class in a leaf.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # Parallel loop
        all_proba = Parallel(n_jobs=n_jobs, verbose=self.verbose,
                             backend="threading")(
            delayed(_parallel_helper)(e, 'predict_proba', X,
                                      check_input=False)
            for e in self.estimators_)

        adjust  = np.exp(1 - self.oob_err_)
        
        # Reduce
        proba = all_proba[0]*adjust[0]
        
        if self.n_outputs_ == 1:
            for j in range(1, len(all_proba)):
                proba += all_proba[j]*adjust[j]

            proba /= len(self.estimators_)

        else:
            for j in range(1, len(all_proba)):
                for k in range(self.n_outputs_):
                    proba[k] += all_proba[j][k]

            for k in range(self.n_outputs_):
                proba[k] /= self.n_estimators

        return proba

class Broof(AdaBoostClassifier):         
    def __init__(self,
                 n_estimators=50,
                 learning_rate=1,
                 random_state=None,
                 weighting_algorithm='broof',
                 n_trees=30,
                 n_jobs=1):
        
        self.weighting_algorithm = weighting_algorithm
        self.n_jobs = n_jobs
        self.n_trees = n_trees
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        super(Broof, self).__init__(
            base_estimator=BoostedRandomForestClassifier(criterion='gini', max_features='sqrt', n_estimators=n_trees, n_jobs=n_jobs, bootstrap=True, oob_score=True),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm="SAMME",
            random_state=random_state)

    def _boost(self, iboost, X, y, sample_weight):
        return self._boost_broof(iboost, X, y, sample_weight)


    def _boost_broof(self, iboost, X, y, sample_weight):
        estimator = self._make_estimator()
        print "sample_weight : ",sample_weight
        try:
            estimator.set_params(random_state=self.random_state)
        except ValueError:
            pass

        estimator.fit(X, y, sample_weight=sample_weight)

        if iboost == 0:
            self.classes_ = np.array(getattr(estimator, 'classes_', None))
            self.n_classes_ = len(self.classes_)

        unsampled_indices = np.where((estimator.oob_decision_function_ > 0.0).any(1))[0]

        y_predict_proba = estimator.oob_decision_function_[unsampled_indices, :]
        
        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1),axis=0)

        # Instances incorrectly classified
        incorrect = y_predict != y[unsampled_indices]

        # Error fraction
        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight[unsampled_indices], axis=0))

        print iboost, np.average(estimator.oob_err_), estimator_error, 1 - estimator.oob_score_, len(unsampled_indices)

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        n_classes = self.n_classes_

        # Stop if the error is at least as bad as random guessing
        if estimator_error >= 1. - (1. / n_classes):
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError('BaseClassifier in AdaBoostClassifier '
                                 'ensemble is worse than random, ensemble '
                                 'can not be fit.')
            return None, None, None

        # Boost weight using multi-class AdaBoost SAMME alg
        estimator_weight = self.learning_rate * (
            np.log((1. - estimator_error) / estimator_error)) # + np.log(n_classes - 1))

        print self.learning_rate * (
            np.log((1. - estimator.oob_err_) / estimator.oob_err_)), estimator.oob_err_

        # Only boost the weights if I will fit again
        #if not iboost == self.n_estimators - 1:
            # Only boost positive weights
        #    sample_weight[unsampled_indices] *= np.exp(estimator_weight * (2*incorrect - 1))

        del estimator.oob_decision_function_
        del estimator.oob_samples_

        return sample_weight, estimator_weight, estimator_error 


    def fit(self, X, y, sample_weight=None):
        """Build a boosted classifier/regressor from the training set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is
            forced to DTYPE from tree._tree if the base classifier of this
            ensemble weighted boosting classifier is a tree or forest.
        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.
        Returns
        -------
        self : object
            Returns self.
        """
        # Check parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if sample_weight is None:
            # Initialize weights to 1 / n_samples
            sample_weight = np.empty(X.shape[0], dtype=np.float)
            sample_weight[:] = 1. / X.shape[0]
        else:
            # Normalize existing weights
            sample_weight = sample_weight / sample_weight.sum() 

            # Check that the sample weights sum is positive
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        # Check parameters
        self._validate_estimator()

        # Clear any previous fit results
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float)

        for iboost in range(self.n_estimators):
            # Boosting step
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X, y,
                sample_weight)

            # Early termination
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            #if iboost < self.n_estimators - 1:
            #    # Normalize
            #    sample_weight /= sample_weight_sum

        return self


    def decision_function(self, X): 
    
        #check_is_fitted(self, "n_classes_")
        X = self._validate_X_predict(X)

        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        pred = None
    
        pred = sum((estimator.predict(X) == classes).T
                    for estimator, w in zip(self.estimators_,
                                            self.estimator_weights_))

        #pred /= self.estimator_weights_.sum()
        if n_classes == 2:
            pred[:, 0] *= -1
            return pred.sum(axis=1)

        print pred
        return pred
        """
        proba = sum(estimator.predict_proba(X)
                        for estimator, w in zip(self.estimators_,
                                                self.estimator_weights_))

        #proba /= self.estimator_weights_.sum()
        proba = np.exp((1. / (n_classes - 1)) * proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba /= normalizer

        print proba
        return proba"""

    def predict(self, X):
        """Predict classes for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        pred = self.decision_function(X)
        
        if self.n_classes_ == 2:
            return self.classes_.take(pred > 0, axis=0)

        return self.classes_.take(np.argmax(pred, axis=1), axis=0)

    def set_params(self, **params):
        """Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The former have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.
        Returns
        -------
        self
        """
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        for key, value in six.iteritems(params):
            split = key.split('__', 1)
            
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if name not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (name, self))
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if key not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (key, self.__class__.__name__))
                if key == 'n_trees':
                    setattr(self.base_estimator, 'n_estimators', value)
                else:
                    setattr(self, key, value)

        return self
