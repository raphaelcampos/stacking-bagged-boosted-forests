import sys
import argparse
import math
import numpy as np

parser = argparse.ArgumentParser(description="This script creates best combination possible.")

parser.add_argument("result", type=str,
                    help="result files. Each row needs to be in the following format:\n\t\"doc id\" \"actual class\" \"predicted class\"", nargs=3)

parser.add_argument("-c", "--correct_only", action="store_false",
                    help="only evaluate the classes which were correctly classified")

parser.add_argument("--cols", type=int,
                    help="Number of columns to display the diagrams", default=4)

parser.add_argument("--labels", type=str,
                    help="Set labels", default='A,B,C')


args = parser.parse_args()

only_correct_class = args.correct_only

set_labels = args.labels.split(',')

sets = []
classess = set()
docs = []
for i, arg in enumerate(args.result) :
	results = np.loadtxt(arg, dtype=int, usecols=(0, 1, 2))
	folds = np.split(results, np.where(results[:,0] == 0)[0])
	folds.pop(0)
	sets.append([])
	docs.append([])
	for results in folds:
		classess = classess | (set(results[:,1]) & set(results[:,2])) 
		sets[i] = sets[i] + [{"docs": results, "label": "kNN"}]
		docs[i] = docs[i] + [set(results[results[:,1]==results[:,2]][:,0])]

#print "classess:", classess

for i in range(len(docs[0])):
	A, B, C = docs 
	A, B, C = A[i], B[i], C[i]

	ABC = ((A | C) | B)
	print "#", i
	for r in sets[0][i]['docs']:
		if r[0] in ABC:
			print r[0], r[1], r[1]
		else:
			print r[0], r[1], r[2]