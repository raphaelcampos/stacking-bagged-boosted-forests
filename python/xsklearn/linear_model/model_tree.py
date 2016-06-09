from sklearn.linear_model.base import LinearClassifierMixin, LinearModel
from sklearn.preprocessing import LabelBinarizer

from scipy.optimize import nnls
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class LinearModelTree(DecisionTreeClassifier):
    """Classifier using Multi-response linear regression.
    Parameters
    ----------
    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_classes, n_features)
        Weight vector(s).
    intercept_ : float | array, shape = (n_targets,)
        Independent term in decision function. Set to 0.0 if
        ``fit_intercept = False``.
    Notes
    -----
    For multi-class classification, n_class classifiers are trained in
    a one-versus-all approach. It uses the non-negative least squares solver
    available in scipy.optimze.nnls based on [MLR2].
    References
    ----------
    [MLR1] Kai Ming Ting and Ian H. Witten. 1999. Issues in stacked generalization. J. Artif. Int. Res. 10, 1 (May 1999), 271-289.
    [MLR2] Lawson C., Hanson R.J., (1987) Solving Least Squares Problems, SIAM
    """
    
    def fit(self, X, y, sample_weight=None):
        """Fit Multi-response linear regression model.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples,n_features]
            Training data
        y : array-like, shape = [n_samples]
            Target values
        sample_weight : float or numpy array of shape (n_samples,)
            Sample weight.
            .. versionadded:: 0.17
               *sample_weight* support to Classifier.
        Returns
        -------
        self : returns an instance of self.
        """

        super(LinearModelTree, self).fit(X, y, sample_weight)

        print(self.tree_.children_left, self.tree_.children_right)

        X_leaves = self.apply(X)

        # all leaves indexes
        leaves = np.where(self.tree_.threshold == -2)[0]

        
        indexes = np.argsort(X_leaves)
        

        print(X_leaves)
        exit(0)

        groups = np.searchsorted(X_leaves, leaves, "left", indexes)

        self.models = [LinearRegression(copy_X=False).fit(X[groups[i]:groups[i+1],:],
                     y[groups[i]:groups[i+1]]) for i in range(len(groups)-1)]


        return self


