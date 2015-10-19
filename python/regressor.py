from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import f1_score
import numpy as np
from sys import argv

from sklearn.cross_validation import train_test_split

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
        weight_train_docs = [1.0/len(X)] * len(X)  # [1.0/X.shape[0]] * X.shape[0] #[1.0/len(trainDocs)] * len(trainDocs # )

        weight_train_docs = np.asarray(weight_train_docs)
        l_alpha = list()
        l_forest = list()
        for rnd in range(rounds):
            print 'Round ', rnd+1, '...'
            rf = RandomForestClassifier(n_estimators=num_trees, criterion='entropy', max_features=0.30, max_depth=None,
                                        min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=None, oob_score=True,
                                        bootstrap=True, n_jobs=-1, random_state=None, verbose=0)
            rf.fit(X, y, weight_train_docs)  # fit the random forest model
            prediction = list()

            miss = 0.0
            notoob = list()
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

            print len(notoob)

            OOBerr = miss/(len(y)-len(notoob))
            print OOBerr, 1 - rf.oob_score_ 
            alpha = np.log((1-OOBerr)/OOBerr)
            l_alpha.append(alpha)
            l_forest.append(rf)
            for index in range(len(weight_train_docs)):
                if index in notoob:
                    continue
                if prediction[index] != y[index]:
                    weight_train_docs[index] = weight_train_docs[index]*np.exp(alpha)
                else:
                    weight_train_docs[index] = weight_train_docs[index]*np.exp(-alpha)
        print 'Model fitting accomplished'
        return (l_alpha, l_forest)

    def classify_broof_m_oob_forest(self, Xt, tup_lalpha_lforest):
        print 'Classifying ...'
        l_alpha = np.asarray(tup_lalpha_lforest[0])
        l_forest = tup_lalpha_lforest[1]
        prediction = list()
        classes_ = dict()

        for f in l_forest:
            prediction.append(f.predict(Xt))

        prediction = np.asarray(prediction)
        fpred = list()
        print l_alpha[0], l_forest[0].oob_score_
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

def main(argv):
    train, test, rounds, n_trees = argv[1], argv[2], int(argv[3]), int(argv[4])
    print "Fitting the model: "
    print "   Train:    ", train
    print "   Test:     ", test
    print "   Rounds:   ", rounds
    print "   N. Trees: ", n_trees

    xtrain = load_svmlight_file(train)
    pwxtest = load_svmlight_file(test)

    #X, Xt, y, yt = train_test_split(X, y, test_size=0.10, random_state=42)

    X = xtrain[0]
    X = X.toarray()
    y = xtrain[1]

    Xt = pwxtest[0]
    Xt = Xt.toarray()
    if X.shape[1] > Xt.shape[1]:
        a = np.array([[0 for x in range(X.shape[1] - Xt.shape[1])] for x in range(len(Xt))])
        Xt = np.hstack((Xt, a))
    elif Xt.shape[1] > X.shape[1]:
        a = np.array([[0 for x in range(Xt.shape[1] - X.shape[1])] for x in range(len(X))])
        X = np.hstack((X, a))
    yt = pwxtest[1]

    b = BROOF()
    tup_lalpha_lforest = b.broof_m_oob_forest(rounds, X, y, n_trees)
    res = b.classify_broof_m_oob_forest(Xt=Xt, tup_lalpha_lforest=tup_lalpha_lforest)

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
