import numpy as np
from IPython import embed
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.externals.joblib import Memory
from os import listdir
from os.path import isfile, join
import scipy.sparse as sp

dataset_folder = "./release/datasets/"

def get_data(dataset_name):
    data = load_svmlight_file(dataset_name)
    return data[0], data[1]

def join_and_save_datasets(dataset_name, train_or_test, remove_pp, PP_mode, fold):
	onlyfiles = sorted([f for f in listdir(dataset_folder) if isfile(join(dataset_folder, f))])	
	if(PP_mode == "metafeatures_only"):		
		correct_set_only_files = [f for f in onlyfiles if dataset_name in f and train_or_test in f and "perform_meta" not in f]		
	elif(remove_pp):
		correct_set_only_files = [f for f in onlyfiles if dataset_name in f and train_or_test in f and "perform_meta" not in f and "pp_meta_features" not in f]
	else:
		correct_set_only_files = [f for f in onlyfiles if dataset_name in f and train_or_test in f and "pp_meta_features" not in f]	
	X_out = None
	fold_files_only = [f for f in correct_set_only_files if fold in f]
	print(fold_files_only)
	if(PP_mode == "raw" or PP_mode == "metafeatures_only"):		
		for file in fold_files_only:
			print(file)
			X, y = get_data(dataset_folder+file)
			X_out = X if X_out is None else sp.hstack((X_out, X))
			print(X_out.shape)
	elif(PP_mode == "weighted"):
		not_pp = [f for f in fold_files_only if "perform_meta" not in f]
		not_pp.sort()
		pp = [f for f in fold_files_only if "perform_meta" in f]
		pp.sort()
		for pp_file, probs_file in zip(pp, not_pp):
			# print(pp_file, probs_file)
			X_pp, y = get_data(dataset_folder+pp_file)
			X_probs, _ = get_data(dataset_folder+probs_file)

			# X_pp_weighted = np.multiply(X_probs.todense(),  X_pp.todense()[:,1])
			X_pp_weighted = np.multiply(X_probs.todense(),  X_pp.todense()[:,0])
			X_out = X_probs if X_out is None else sp.hstack((X_out, X_probs))
			X_out = sp.hstack((X_out, X_pp_weighted))
	elif(PP_mode == "weighted_only"):
		# embed()
		not_pp = [f for f in fold_files_only if "perform_meta" not in f]
		not_pp.sort()
		pp = [f for f in fold_files_only if "perform_meta" in f]
		pp.sort()
		for pp_file, probs_file in zip(pp, not_pp):
			# print(pp_file, probs_file)
			X_pp, y = get_data(dataset_folder+pp_file)
			X_probs, _ = get_data(dataset_folder+probs_file)

			X_pp_weighted = np.multiply(X_probs.todense(),  X_pp.todense()[:,0])			
			X_out = X_pp_weighted if X_out is None else np.hstack((X_out, X_pp_weighted))
	print(X_out.shape)
	return X_out, y

def main():
	for fold in ["fold1", "fold2", "fold3", "fold4", "fold5"]:	
		# for PP_mode in ["weighted", "raw", "weighted_only", "metafeatures_only"]:
		for PP_mode in ["metafeatures_only"]:
			for dataset_name in ['amazon', 'bbc', 'debate', 'digg', 'myspace', 'nyt', 'tweets', 'yelp', 'youtube']:
				if(PP_mode == "raw"):
					X_out, y = join_and_save_datasets(dataset_name, "train", True, PP_mode, fold)
					dump_svmlight_file(X_out, y, dataset_folder+"performance_prediction/"+"train_"+ dataset_name + "_no_PP"+ "_" + fold)
					X_out, y = join_and_save_datasets(dataset_name, "test", True, PP_mode, fold)
					dump_svmlight_file(X_out, y, dataset_folder+"performance_prediction/"+"test_"+ dataset_name + "_no_PP"+ "_" + fold)
				if(PP_mode == "metafeatures_only"):
					X_out, y = join_and_save_datasets(dataset_name, "train", True, PP_mode, fold)
					dump_svmlight_file(X_out, y, dataset_folder+"performance_prediction/"+"train_"+ dataset_name + "_PPmetafeatures_only"+ "_" + fold)
					X_out, y = join_and_save_datasets(dataset_name, "test", True, PP_mode, fold)
					dump_svmlight_file(X_out, y, dataset_folder+"performance_prediction/"+"test_"+ dataset_name + "_PPmetafeatures_only"+ "_" + fold)
				else:
					X_out, y = join_and_save_datasets(dataset_name, "train", False, PP_mode, fold)
					dump_svmlight_file(X_out, y, dataset_folder+"performance_prediction/"+"train_"+ dataset_name + "_PP" + PP_mode + "_" + fold, fold)
					X_out, y = join_and_save_datasets(dataset_name, "test", False, PP_mode, fold)
					dump_svmlight_file(X_out, y, dataset_folder+"performance_prediction/"+"test_"+ dataset_name + "_PP" + PP_mode+ "_" + fold, fold)
					
if __name__ == '__main__':
	main()