#!/bin/bash

dataset_dir=$1
output_dir=$2
n_jobs=$3
trials=2

datasets=('20ng' 'reuters90' 'acm')

################################################################################
#				 					${dataset} 									   #
################################################################################ 
for dataset in ${datasets}
do
	#Suport Vector Machine (LIBSVM)
	#method=svm
	#python ../python/main.py -m ${method} --cv 5 -g 0 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

	# kNearestNeighbors
	#method=knn
	#python ../python/main.py -m ${method} --cv 5 -g 0 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

	# Naive Bayes
	#method=nb
	#python ../python/main.py -m ${method} --cv 5 -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

	# Random Forest
	#method=rf
	#python ../python/main.py -m ${method} --cv 5 -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

	# Extremely Randomized Trees
	#method=xt
	#python ../python/main.py -m ${method} --cv 5 -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

	# Lazy Random Forest
	method=culazy
	python ../python/main.py -m ${method} --cv 5 -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

	# Lazy Extremely Randomized Trees
	method=culxt
	python ../python/main.py -m ${method} --cv 5 -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

	# BROOF
	#method=broof
	#python ../python/main.py -m ${method} --cv 5 -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

	# BERT
	#method=bert
	#python ../python/main.py -m ${method} --cv 5 -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}
done
################################################################################