from sklearn.ensemble import RandomForestClassifier as ForestClassifier
from sklearn.ensemble import ExtraTreesClassifier as ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier

from numpy.core.umath_tests import inner1d

import numpy as np

class Broof(AdaBoostClassifier):         
    def __init__(self,
                 n_estimators=50,
                 estimator_params=tuple(),
                 learning_rate=1,
                 random_state=None):
        
        super(Broof, self).__init__(
            base_estimator=ForestClassifier(n_estimators=20, n_jobs=8, oob_score=True),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=None)
    
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
        
        return super(Broof, self).fit(X, y, sample_weight)

    def _boost(self, iboost, X, y, sample_weight):
        """Implement a single boost using the SAMME.R real algorithm."""
        estimator = self._make_estimator()

        try:
            estimator.set_params(random_state=self.random_state)
        except ValueError:
            pass

        estimator.fit(X, y, sample_weight=sample_weight)

        unsampled_indices = np.where(np.any(estimator.oob_decision_function_ != 0.0, axis=1))[0]
        y_predict_proba = estimator.oob_decision_function_[unsampled_indices, :]

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1),
                                       axis=0)

        # Instances incorrectly classified
        incorrect = y_predict != y[unsampled_indices,]

        # Error fraction
        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        # Construct y coding as described in Zhu et al [2]:
        #
        #    y_k = 1 if c == k else -1 / (K - 1)
        #
        # where K == n_classes_ and c, k in [0, K) are indices along the second
        # axis of the y coding with c being the index corresponding to the true
        # class label.
        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1. / (n_classes - 1), 1.])
        y_coding = y_codes.take(classes == y[unsampled_indices, np.newaxis])

        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        y_predict_proba[y_predict_proba <= 0] = 1e-5

        # Boost weight using multi-class AdaBoost SAMME.R alg
        estimator_weight = (-1. * self.learning_rate
                                * (((n_classes - 1.) / n_classes) *
                                   inner1d(y_coding, np.log(y_predict_proba))))

        # Only boost the weights if it will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight[unsampled_indices,] *= np.exp(estimator_weight *
                                    ((sample_weight > 0) |
                                     (estimator_weight < 0)))

        return sample_weight, 1., estimator_error
