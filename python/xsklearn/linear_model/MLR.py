from sklearn.linear_model.base import LinearClassifierMixin, LinearModel
from sklearn.preprocessing import LabelBinarizer

from scipy.optimize import nnls
import numpy as np

class MLR(LinearClassifierMixin, LinearModel):
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
    def __init__(self):
        super(MLR, self).__init__()
        
        self.fit_intercept = False
        self.intercept_ = 0

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

        n_samples, n_features = X.shape

        self._label_binarizer = LabelBinarizer(pos_label=1, neg_label=0)
        Y = self._label_binarizer.fit_transform(y)

        #X, y, X_mean, y_mean, X_std = LinearModel._center_data(
        #    X, y, self.fit_intercept, self.normalize, self.copy_X,
        #    sample_weight=sample_weight)

        n_classes_ = len(self._label_binarizer.classes_)
        self.coef_ = np.zeros((n_classes_, n_features))
        X_copy = X.copy()
        for i, y in enumerate(Y.T):
        	r, _ = nnls(X_copy, y)
        	self.coef_[i,:] = r
        
        #self._set_intercept(X_mean, y_mean, X_std)

        return self

    @property
    def classes_(self):
        return self._label_binarizer.classes_