import argparse
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import scipy.sparse as sp

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Concatenates datasets by features.')
	parser.add_argument('datasets', metavar='data', type=str, nargs='+',
	                    help='datasets which will be concatenate')
	parser.add_argument('-o', '--output', type=str,
	                    help='file to sava the concatenated datasets')

	args = parser.parse_args()

	X_out = None
	for data in args.datasets:
		X, y = load_svmlight_file(data)
		X_out = X if X_out is None else sp.hstack((X_out, X))
	
	if not args.output is None:
		dump_svmlight_file(X_out, y, args.output)