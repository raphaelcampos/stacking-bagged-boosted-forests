
import numpy as np

def get_sizes(y_estimators, y_true):
	if not isinstance(y_estimators, tuple):
		raise(Exception("y_estimators must be a tuple")) 

	for y_estimator in y_estimators:
		if y_estimator.shape != y_true.shape:
			raise(Exception("estimator prediction size must be equal to y_true: %s %s" % (y_estimator.shape, y_true.shape))) 

	mask1, mask2 = y_estimators
	mask1, mask2 = mask1 == y_true, mask2 == y_true

	N00 = np.logical_and(~mask1, ~mask2).sum()
	N01 = np.logical_and(mask1, ~mask2).sum()
	N10 = np.logical_and(~mask1, mask2).sum()
	N11 = np.logical_and(mask1, mask2).sum()

	return N00, N01, N10, N11

def disagreement_degree(y_estimators, y_true):
	N00, N01, N10, N11 = get_sizes(y_estimators, y_true)

	return (N01 + N10) / float(N00 + N01 + N10 + N11)

def normalized_disagreement_degree(y_estimators, y_true):
	N00, N01, N10, N11 = get_sizes(y_estimators, y_true)

	N = float(N00 + N01 + N10 + N11)

	# accuracies
	Ri = (N10 + N11) / N  
	Rj = (N01 + N11) / N

	dis = (N01 + N10) / N
	dis_min = Ri + Rj - 2 * min(Ri, Rj)
	dis_max = 2 - Ri - Rj
	print(Rj, Ri, dis, dis_min, dis_max, (dis - dis_min) / (dis_max - dis_min), N00, N01, N10, N11)
	return round(dis - dis_min, 10) / round(dis_max - dis_min, 10)

def combination_pontential_degree(y_estimators, y_true):
	N00, N01, N10, N11 = get_sizes(y_estimators, y_true)

	N = float(N00 + N01 + N10 + N11)

	# accuracies
	Ri = (N10 + N11) / N  
	Rj = (N01 + N11) / N

	dis = (N01 + N10) / N
	dis_min = Ri + Rj - 2 * min(Ri, Rj)
	dis_max = 2 - Ri - Rj
	print(Rj, Ri, dis, dis_min, dis_max, (dis - dis_min) / (dis_max - dis_min), N00, N01, N10, N11)
	return (1 - (N00 / N)) / (max(Rj, Ri))

	#return round(dis - dis_min, 10) / round(dis_max - dis_min, 10)


def double_fault(y_estimators, y_true):
	N00, N01, N10, N11 = get_sizes(y_estimators, y_true)

	N = float(N00 + N01 + N10 + N11)
	return (N00) / float(N00 + N01 + N10 + N11)


if __name__ == '__main__':

	y_true = np.asarray([1, 1, 1, 1, 1])
	y_estimators = (np.asarray([0,1,1,1,1]), np.asarray([0, 1, 0, 1, 0]))

	result = get_sizes(y_estimators, y_true)
	print(result)

	print(disagreement_degree(y_estimators, y_true))
	print(normalized_disagreement_degree(y_estimators, y_true))