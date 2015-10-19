import sys
import argparse
import math
import numpy as np

from matplotlib import pyplot as plt
from matplotlib_venn import *

from matplotlib import gridspec

def subset_values(venn, sets):
	A,B,C = sets
	venn.get_label_by_id('001').set_text(len(C - ( (C & B) | (A & C) - (A & B & C))))
	venn.get_label_by_id('010').set_text(len(B - ( (C & B) | (A & B) - (A & B & C))))
	venn.get_label_by_id('011').set_text(len(B & C) - len(A & B & C))
	venn.get_label_by_id('100').set_text(len(A - ( (A & B) | (A & C) - (A & B & C))))
	venn.get_label_by_id('101').set_text(len(A & C) - len(A & B & C))
	venn.get_label_by_id('110').set_text(len(A & B) - len(A & B & C))
	venn.get_label_by_id('111').set_text(len(A & B & C))

		

parser = argparse.ArgumentParser(description="This script creates Venn's diagram.")

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
for arg in args.result :
	results = np.loadtxt(arg, dtype=int, usecols=(0, 1, 2))
	classess = classess | (set(results[:,1]) & set(results[:,2])) 
	sets = sets + [{"docs": results, "label": "kNN"}]
	docs = docs + [set(results[results[:,1]==results[:,2]][:,0])]

#classess = classess & set(xrange(len(classess)/2+1,len(classess)))

print "classess:", classess

A,B,C = docs

all_docs = set(sets[0]['docs'][:,0])
print len(A)/float(len(sets[0]['docs']))
print len(C)/float(len(sets[0]['docs']))
print len(B)/float(len(sets[0]['docs']))
print len( (A | C) )/float(len(sets[0]['docs']))
print len( all_docs - (A | C) )/float(len(all_docs))

#for r in sets[0]['docs']:
#	if r[0] in (A | C):
#		print r[0], r[1], r[1]
#	else:
#		print r[0], r[1], r[2]

cols = args.cols
rows = int(math.ceil((len(classess))/float(cols)))

figure = plt.figure()

gs = gridspec.GridSpec(rows, cols)
for i, clazz in enumerate(classess):
	s = []
	all_null = True
	for j in xrange(0,len(sets)):
		r = sets[j]['docs']

		v = set(r[r[:,2] == clazz][:,0]) if only_correct_class else set(r[r[:,2] == clazz][:,0]) & set(r[r[:,1] == clazz][:,0])

		all_null = all_null and len(v) == 0
		s = s + [v]
	
	if all_null:
		continue

	row = i/cols
	col = i - row*cols

	if len(classess) - 1 == i and not len(classess)%cols == 0:
		ax = figure.add_subplot(gs[row,col:])
	else:
		ax = figure.add_subplot(gs[row,col])
	ax.set_title("class " + str(clazz))

	#v = venn3([s[0], s[1], s[2]], ('lazy', 'rf', 'broof'), ax=ax)
	subset_values(venn3(subsets=(1,1,1,1,1,1,1,1,1), set_labels=set_labels, ax=ax), [s[0], s[1], s[2]])
	v = venn3_circles(subsets=(1,1,1,1,1,1,1,1,1), ax=ax, linestyle='dashed')
	v[0].set_ec('red')
	v[0].set_alpha(0.5)
	v[1].set_ec('green')
	v[2].set_ec('blue')
	v[2].set_alpha(0.5)

plt.show()
