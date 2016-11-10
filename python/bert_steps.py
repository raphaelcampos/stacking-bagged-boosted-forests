print(__doc__)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from xsklearn.ensemble import Bert, Broof, BoostedExtraTreesClassifier, BoostedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


class BertBroof(BoostedForestClassifier):
    def __init__(self,
                 n_iterations=200,
                 learning_rate=1,
                 n_trees=5,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 aux_mem=False):    

        super(BertBroof, self).__init__(
            base_estimator=[BoostedExtraTreesClassifier(n_estimators=n_trees,
                                 criterion=criterion,
                                 max_depth=max_depth,
                                 min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf,
                                 min_weight_fraction_leaf=min_weight_fraction_leaf,
                                 max_features=max_features,
                                 max_leaf_nodes=max_leaf_nodes,
                                 n_jobs=n_jobs,
                                 bootstrap=True),
                            BoostedRandomForestClassifier(n_estimators=n_trees,
                                 criterion=criterion,
                                 max_depth=max_depth,
                                 min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf,
                                 min_weight_fraction_leaf=min_weight_fraction_leaf,
                                 max_features=max_features,
                                 max_leaf_nodes=max_leaf_nodes,
                                 n_jobs=n_jobs,
                                 bootstrap=True)],
            n_estimators = n_iterations,
            learning_rate = learning_rate,
            random_state = random_state,
            aux_mem=aux_mem
            )

        self.n_jobs = n_jobs
        self.n_iterations = n_iterations
        self.n_trees = n_trees
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.aux_mem=aux_mem

        self.random_instance = check_random_state(self.random_state)

    def _make_estimator(self, append=True):
        """Make and configure a copy of the `base_estimator_` attribute.
        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        p = 0.0
        
        chosen_id = self.random_instance.choice([0,1], p=[p, 1 - p])

        estimator = clone(self.base_estimator[chosen_id])
        estimator.set_params(**dict((p, getattr(self, p))
                                    for p in self.estimator_params))

        if append:
            self.estimators_.append(estimator)

        return estimator


h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis",
         "Quadratic Discriminant Analysis"]

n_iter = 200
clf = AdaBoostClassifier(random_state=42, n_estimators=n_iter, base_estimator=DecisionTreeClassifier(max_depth=100, max_features=1, splitter='best'))
#clf = Bert(n_iterations=n_iter, n_trees=1, max_depth=None, max_features=1, n_jobs=1, random_state=42)

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=2, n_samples=400, class_sep=0.5)
print(X.shape)
rng = np.random.RandomState(2)

X += 2 * rng.uniform(size=X.shape)

linearly_separable = (X, y)

datasets = [#make_moons(noise=0.8, random_state=0, n_samples=400),
            #make_circles(noise=0.3, factor=0.5, random_state=1, n_samples=1000),
            linearly_separable
            ]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds in datasets:
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(1 , 1, i)
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i = 1
    plt.show()

    clf.fit(X_train, y_train)
    score = clf.staged_score(X_test, y_test)
    step = 1
    for S, Z, est in zip(score, clf.staged_decision_function(np.c_[xx.ravel(), yy.ravel()]), clf.estimators_):
        # iterate over classifiers
        ax = plt.subplot(1, 1, i)
        
        
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        #size = np.repeat(2.0, X_train.shape[0]) + (6-2.0)*(est.sample_weight - est.sample_weight.min())/(est.sample_weight.max() - est.sample_weight.min() + 0.0001)
        size=np.repeat(2.0, X_train.shape[0])
        size = 2*np.pi*size**2
        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, s=size, alpha=0.75)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   alpha=0.4)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title("BERT: Iter. %d" % (step))
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % S).lstrip('0'),
                size=15, horizontalalignment='right')
        i = 1

        plt.savefig("step%s_.png" % str(step).zfill(2))
        plt.close()
        #plt.savefig("step00.png")
        #plt.close()
        #exit()
        step += 1


figure.subplots_adjust(left=.02, right=.98)
plt.show()