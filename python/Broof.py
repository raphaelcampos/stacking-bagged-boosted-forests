from sklearn.ensemble import RandomForestClassifier as ForestClassifier
from sklearn.ensemble import ExtraTreesClassifier as ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier

    

import numpy as np

class Broof(AdaBoostClassifier):         
    def __init__(self,
                 n_estimators=50,
                 estimator_params=tuple(),
                 learning_rate=1,
                 random_state=None):
        
        super(Broof, self).__init__(
            base_estimator=ForestClassifier(n_estimators=30, n_jobs=8, oob_score=True),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm="SAMME",
            random_state=random_state)

    def _boost(self, iboost, X, y, sample_weight):
        """Implement a single boost using the SAMME discrete algorithm."""
        estimator = self._make_estimator()

        try:
            estimator.set_params(random_state=self.random_state)
        except ValueError:
            pass

        estimator.fit(X, y, sample_weight=sample_weight)
        
        #y_predict = estimator.predict(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        y_predict_proba = estimator.oob_decision_function_
        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1),
                                       axis=0)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))

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
            np.log((1. - estimator_error) / estimator_error) +
            np.log(n_classes - 1.))

        # Only boost the weights if I will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(estimator_weight * incorrect *
                                    ((sample_weight > 0) |
                                     (estimator_weight < 0)))

        return sample_weight, estimator_weight, estimator_error
