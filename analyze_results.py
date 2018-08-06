# -*- coding: utf-8 -*-
import pandas as pd
from IPython import embed
from tqdm import tqdm
import math
import numpy as np
import argparse
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default = None)
args = parser.parse_args()

lines = [line.decode('utf-8').rstrip('\n') for line in open(args.dataset)]
no_pp = []
pp_meta_features = []
pp_raw = []
pp_weighted = []
pp_weighted_only = []
for i, line in enumerate(lines):
	# line = line.replace("\n","")
	if(line=="no_PP"):
		micro = float(lines[i+4])		
		no_pp.append(micro)
	elif(line=="PPraw"):
		micro = float(lines[i+4])
		pp_raw.append(micro)
	elif(line=="PPweighted"):
		micro = float(lines[i+4])
		pp_weighted.append(micro)
	elif(line=="PPweighted_only"):
		micro = float(lines[i+4])
		pp_weighted_only.append(micro)
	elif(line=="PPmetafeatures_only"):
		micro = float(lines[i+4])
		pp_meta_features.append(micro)

print("MICRO on dataset "+args.dataset+": ")
print("No PP: " +str(np.mean(no_pp)))

t_value, p_value = stats.ttest_rel(pp_meta_features,no_pp)
print("+ PP only " + str(np.mean(pp_meta_features)) + " > baseline = " + str(t_value>0) + " pvalue: "+ str(p_value))

t_value, p_value = stats.ttest_rel(pp_raw,no_pp)
print("+ ML-PP raw: " +str(np.mean(pp_raw)) + " > baseline = " + str(t_value>0) + " pvalue: "+ str(p_value))

t_value, p_value = stats.ttest_rel(pp_weighted,no_pp)
print("+ ML-PP weighted: " +str(np.mean(pp_weighted)) + " > baseline = " + str(t_value>0) + " pvalue: "+ str(p_value))

t_value, p_value = stats.ttest_rel(pp_weighted_only,no_pp)
print("Only ML-PP weighted: " +str(np.mean(pp_weighted_only)) + " > baseline = " + str(t_value>0) + " pvalue: "+ str(p_value))
print("========================")


t_value, p_value = stats.ttest_rel(pp_raw,pp_meta_features)
print("+ ML-PP raw: " +str(np.mean(pp_raw)) + " > baseline2 (+PP) = " + str(t_value>0) + " pvalue: "+ str(p_value))

t_value, p_value = stats.ttest_rel(pp_weighted,pp_meta_features)
print("+ ML-PP weighted: " +str(np.mean(pp_weighted)) + " > baseline2 (+PP) = " + str(t_value>0) + " pvalue: "+ str(p_value))

t_value, p_value = stats.ttest_rel(pp_weighted_only,pp_meta_features)
print("Only ML-PP weighted: " +str(np.mean(pp_weighted_only)) + " > baseline2 (+PP) = " + str(t_value>0) + " pvalue: "+ str(p_value))