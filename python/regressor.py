from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import f1_score

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups

import numpy as np
from sys import argv

class BROOF:
    def __init__(self):
        pass

    def broof_m_oob_forest(self, rounds, X, y, num_trees):
        """
        This algorithm considers that each tree within the forest of each round of the boosting algorithm has a  out of
        bag. If the number of trees is large, all docs in the training data set will be picked at least one time. Then,
        we can consider that all training set is an out of bag sample at least once in a tree.
        :param rounds: The number of rounds of the boosting algorithm
        :param X: The training data set in a np.array like format of shape [n_documents][n_features]
        :param y: The Ground Truth label for X in an np.array like format
        :param num_trees: The number of trees desired in each forest of the boosting algorithm
        :return: a triple of alpha, score and forest for each round of the boosting algorithm
        """
        weight_train_docs = [1.0/len(y)] * len(y)  # [1.0/X.shape[0]] * X.shape[0] #[1.0/len(trainDocs)] * len(trainDocs # )

        weight_train_docs = np.asarray(weight_train_docs)
        l_alpha = list()
        l_forest = list()
        for rnd in range(rounds):
            print 'Round ', rnd+1, '...'
            rf = RandomForestClassifier(n_estimators=num_trees, criterion='gini', max_features="auto", max_depth=None,
                                        min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=None, oob_score=True,
                                        bootstrap=True, n_jobs=-1, random_state=None, verbose=0)
            rf.fit(X, y, weight_train_docs)  # fit the random forest model
            prediction = list()

            miss = 0.0
            notoob = list()
            """
            for index in range(len(y)):  # if prediction[index] != y[index]: # miss = miss + 1
                class_ = -1000.0
                if np.isnan(rf.oob_decision_function_[index][0]):
                    notoob.append(index)
                    prediction.append(np.nan)
                else:
                    for idpred in range(len(rf.oob_decision_function_[index])):
                        if rf.oob_decision_function_[index][idpred] > class_:
                            class_ = idpred
                    if class_ != y[index]:
                        miss += 1  # weight_train_docs[index]
                    prediction.append(class_)
            """
            if rnd == 0:
                self.classes_ = getattr(rf, 'classes_', None)
                self.n_classes_ = len(self.classes_)

            unsampled_indices = np.where((rf.oob_decision_function_ > 0.0).any(1))[0]
            y_predict_proba = rf.oob_decision_function_[unsampled_indices, :]
            y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1),axis=0)

            # Instances incorrectly classified
            incorrect = y_predict != y[unsampled_indices]

            # Error fraction
            OOBerr = np.mean(
                np.average(incorrect, weights=weight_train_docs[unsampled_indices], axis=0))

            #OOBerr = miss/(len(y)-len(notoob))
            alpha = np.log((1.-OOBerr)/OOBerr) + np.log(self.n_classes_ - 1)
            l_alpha.append(alpha)
            l_forest.append(rf)

            if not rnd == rounds - 1:
                # Only boost positive weights
                weight_train_docs[unsampled_indices] *= np.exp(alpha *  (incorrect))

            """
            for index in range(len(weight_train_docs)):
                if index in notoob:
                    continue
                if prediction[index] != y[index]:
                    weight_train_docs[index] = weight_train_docs[index]*np.exp(alpha)
                else:
                    weight_train_docs[index] = weight_train_docs[index]*np.exp(-alpha)
            """
        print 'Model fitting accomplished'
        return (l_alpha, l_forest)

    def classify_broof_m_oob_forest(self, Xt, tup_lalpha_lforest):
        print 'Classifying ...'
        l_alpha = np.asarray(tup_lalpha_lforest[0])
        l_forest = tup_lalpha_lforest[1]
        prediction = list()
        classes_ = dict()

        X = Xt
        #X = self._validate_X_predict(X)

        n_classes = len(l_forest[0].classes_)
        classes = l_forest[0].classes_[:, np.newaxis]
        pred = None

        pred = sum((estimator.predict(X) == classes).T * w
                    for estimator, w in zip(l_forest,
                                            l_alpha))

        pred #/= l_alpha.sum()
        if n_classes == 2:
            pred[:, 0] *= -1
            return pred.sum(axis=1)
        return classes.take(np.argmax(pred, axis=1), axis=0)
        """
        prediction = np.asarray(prediction)
        fpred = list()
        for i in range(prediction.shape[1]):  # i is the index for each document
            for class_ in l_forest[0].classes_:
                classes_[class_] = 0.0

            for j in range(prediction.shape[0]):  # j is an index for each predictor
                c = prediction[j][i]
                classes_[c] += l_alpha[j] * l_forest[j].oob_score_

            max = -999999999.0
            pred_class = -100000.0
            for key in classes_:
                if classes_[key] > max:
                    max = classes_[key]
                    pred_class = key
            fpred.append(pred_class)
        print 'Done'
        #print fpred
        return np.asarray(fpred)
        """

def main(argv):
    train, test, rounds, n_trees = argv[1], argv[2], int(argv[3]), int(argv[4])
    print "Fitting the model: "
    print "   Train:    ", train
    print "   Test:     ", test
    print "   Rounds:   ", rounds
    print "   N. Trees: ", n_trees

    #xtrain = load_svmlight_file(train)
    #pwxtest = load_svmlight_file(test)
    n_train = 11094
    #X, y = load_svmlight_file(train)

    #tf_transformer = TfidfTransformer(use_idf=True)

    #X = tf_transformer.fit_transform(X)
    
    twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)

    count_vect = CountVectorizer(min_df=6, stop_words='english')
    X = count_vect.fit_transform(twenty_train.data)
    y = twenty_train.target
    
    tfidf_transformer = TfidfTransformer()
    X = tfidf_transformer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=42)
    print X_test, X_test.shape
    xtrain = X_train,y_train
    pwxtest = X_test,y_test

    X = xtrain[0]
    #X = X.toarray()
    y = xtrain[1]

    Xt = pwxtest[0]
    """Xt = Xt.toarray()
    if X.shape[1] > Xt.shape[1]:
        a = np.array([[0 for x in range(X.shape[1] - Xt.shape[1])] for x in range(len(Xt))])
        Xt = np.hstack((Xt, a))
    elif Xt.shape[1] > X.shape[1]:
        a = np.array([[0 for x in range(Xt.shape[1] - X.shape[1])] for x in range(len(X))])
        X = np.hstack((X, a))
    """
    yt = pwxtest[1]

    b = BROOF()
    tup_lalpha_lforest = b.broof_m_oob_forest(rounds, X, y, n_trees)
    res = b.classify_broof_m_oob_forest(Xt=Xt, tup_lalpha_lforest=tup_lalpha_lforest)

    #rf = RandomForestClassifier(n_estimators=200, criterion='gini', max_features="auto", max_depth=None,
    #                                    min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=None, oob_score=True,
    #                                    bootstrap=True, n_jobs=-1, random_state=None, verbose=0)
    #rf.fit(X,y)
    #res = rf.predict(Xt)
    print "F1 Score "
    print "\tMicro: ", f1_score(y_true=yt, y_pred=res, average='micro')
    print "\tMacro: ", f1_score(y_true=yt, y_pred=res, average='macro')



if __name__ == '__main__':
    if len(argv) != 5:
        print "\n\nError, the parameters are: "
        print "\t train: training file in svm light format"
        print "\t test: test file in svm light format"
        print "\t rounds: rounds of the boosting algorithm"
        print "\t n_tress: number of trees for each random forest\n\n"
    else:
        main(argv=argv)