from sklearn.ensemble import RandomForestClassifier as ForestClassifier
from sklearn.ensemble import ExtraTreesClassifier as ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier

import numpy as np

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

        super(Broof, self).__init__(
            base_estimator=ForestClassifier(criterion='gini',max_features='auto',n_estimators=n_trees, n_jobs=n_jobs, bootstrap=True, oob_score=True),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm="SAMME",
            random_state=random_state)

    def _boost(self, iboost, X, y, sample_weight):
        return self._boost_broof(iboost, X, y, sample_weight)


    def _boost_broof(self, iboost, X, y, sample_weight):
        """Implement a single boost using the SAMME discrete algorithm."""
        estimator = self._make_estimator()

        try:
            estimator.set_params(random_state=self.random_state)
        except ValueError:
            pass

        estimator.fit(X, y, sample_weight=sample_weight)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        unsampled_indices = np.where((estimator.oob_decision_function_ > 0.0).any(1))[0]

        y_predict_proba = estimator.oob_decision_function_[unsampled_indices, :]
        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1),axis=0)

        # Instances incorrectly classified
        incorrect = y_predict != y[unsampled_indices]

        # Error fraction
        estimator_error = np.mean(
            np.average(incorrect, axis=0))

        #print iboost, estimator_error, 1 - estimator.oob_score_, len(unsampled_indices)

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
            np.log((1. - estimator_error) / estimator_error) + np.log(n_classes - 1))

        # Only boost the weights if I will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight[unsampled_indices] *= np.exp(estimator_weight * (2*incorrect - 1))

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

            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum

        return self


    def decision_function(self, X):	
	
        #check_is_fitted(self, "n_classes_")
        X = self._validate_X_predict(X)

        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        pred = None
	
        pred = sum((estimator.predict(X) == classes).T * w * estimator.oob_score_
                    for estimator, w in zip(self.estimators_,
                                            self.estimator_weights_))

        pred /= self.estimator_weights_.sum()
        if n_classes == 2:
            pred[:, 0] *= -1
            return pred.sum(axis=1)
        return pred


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
