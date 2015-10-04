import sys
import argparse
import math
import numpy as np

from matplotlib import pyplot as plt
from matplotlib_venn import *

from matplotlib import gridspec

parser = argparse.ArgumentParser(description="This script creates Venn's diagram.")

parser.add_argument("result", type=str,
                    help="result files. Each row needs to be in the following format:\n\t\"doc id\" \"actual class\" \"predicted class\"", nargs=3)

parser.add_argument("-c", "--correct_only", action="store_false",
                    help="only evaluate the classes which were correctly classified")

parser.add_argument("--cols", type=int,
                    help="Number of columns to display the diagrams", default=4)

args = parser.parse_args()

only_correct_class = args.correct_only

sets = []
classess = set()
for arg in args.result :
	results = np.loadtxt(arg, dtype=int, usecols=(0, 1, 2))
	classess = classess | (set(results[:,1]) & set(results[:,2])) 
	sets = sets + [{"docs": results, "label": "kNN"}]


#classess = classess & set(xrange(80,90))

print "classess:", classess

cols = args.cols
rows = int(math.ceil((len(classess))/float(cols)))

figure = plt.figure()

gs = gridspec.GridSpec(rows, cols)
for i, clazz in enumerate(classess):
	s = []
	for j in xrange(0,len(sets)):
		r = sets[j]['docs']
		s = s + [set(r[r[:,2] == clazz][:,0]) if only_correct_class else set(r[r[:,2] == clazz][:,0]) & set(r[r[:,1] == clazz][:,0])]
	
	row = i/cols
	col = i - row*cols

	if len(classess) - 1 == i and not len(classess)%cols == 0:
		ax = figure.add_subplot(gs[row,col:])
	else:
		ax = figure.add_subplot(gs[row,col])
	ax.set_title("class " + str(clazz))
	v = venn3([s[0], s[1], s[2]], ('lazy', 'rf', 'knn'), ax=ax)


plt.show()
