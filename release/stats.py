from sklearn.metrics import f1_score,classification_report

import numpy as np

import argparse

def stats(results):
  folds_macro = []
  folds_micro = []

  arr = np.loadtxt(results)

  folds = np.split(arr, np.where(arr[:,0] == 0)[0])

  for fold in folds:
    pred, y_test = fold[:,1], fold[:,2]
    folds_micro = folds_micro + [f1_score(y_true=y_test, y_pred=pred, average='micro')]
    folds_macro = folds_macro + [f1_score(y_true=y_test, y_pred=pred, average='macro')]

  folds_micro.pop(0)
  folds_macro.pop(0) 

  return np.mean(folds_micro)*100, \
          np.average(folds_macro)*100



folds_macro = []
folds_micro = []

parser = argparse.ArgumentParser(description="This script creates Venn's diagram.")

parser.add_argument("result", type=str,
                    help="result files. Each row needs to be in the following format:\n\t\"doc id\" \"actual class\" \"predicted class\"")
args = parser.parse_args()

arr = np.loadtxt(args.result)

folds = np.split(arr, np.where(arr[:,0] == 0)[0])

for fold in folds:
	pred, y_test = fold[:,1], fold[:,2]
	folds_micro = folds_micro + [f1_score(y_true=y_test, y_pred=pred, average='micro')]
	folds_macro = folds_macro + [f1_score(y_true=y_test, y_pred=pred, average='macro')]


print folds_micro, folds_macro
folds_micro.pop(0)
folds_macro.pop(0)
print "F1-Score"
print "\tMicro: ", np.mean(folds_micro)*100, np.std(folds_micro, ddof=1)*100
print "\tMacro: ", np.average(folds_macro)*100, np.std(folds_macro,ddof=1)*100