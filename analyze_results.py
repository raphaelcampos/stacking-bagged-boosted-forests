# -*- coding: utf-8 -*-
import pandas as pd
from IPython import embed
from tqdm import tqdm
import math
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default = None)
args = parser.parse_args()

lines = [line.decode('utf-8').rstrip('\n') for line in open("results_"+args.dataset+".txt")]
no_pp = []
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

print("MICRO on dataset "+args.dataset+": ")
print("No PP: " +str(np.mean(no_pp)))
print("+ PP raw: " +str(np.mean(pp_raw)))
print("+ PP weighted: " +str(np.mean(pp_weighted)))
print("Only PP weighted: " +str(np.mean(pp_weighted_only)))
# embed()