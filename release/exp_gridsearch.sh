#!/bin/bash

dataset_dir=$1
output_dir=$2
n_jobs=$3
trials=5

datasets=('medline')

################################################################################
#				 					${dataset} 									   #
################################################################################ 
for dataset in ${datasets}
do
	#Suport Vector Machine (LIBSVM)
	method=svm
	python ../python/main.py -m ${method} --cv 5 -g 0 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

	# kNearestNeighbors
	method=knn
	python ../python/main.py -m ${method} --cv 5 -g 0 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

	# Naive Bayes
	method=nb
	python ../python/main.py -m ${method} --cv 5 -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

	# Random Forest
	method=rf
	python ../python/main.py -m ${method} --cv 5 -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

	# Extremely Randomized Trees
	method=xt
	python ../python/main.py -m ${method} --cv 5 -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

	# Lazy Random Forest
	method=lazy
	#python ../python/main.py -m ${method} --cv 5 -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}


	# Lazy Extremely Randomized Trees
	method=lxt
	#python ../python/main.py -m ${method} --cv 5 -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

	# BROOF
	#method=broof
	#python ../python/main.py -m ${method} --cv 5 -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

	# BERT
	#method=bert
	#python ../python/main.py -m ${method} --cv 5 -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}
done
################################################################################

#dataset=4uni
#method=lazy
#python ../python/main.py -m ${method} -t 200 -k 200 -f sqrt -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

#dataset=20ng
#method=lazy
#python ../python/main.py -m ${method} -t 200 -k 200 -f sqrt -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

#dataset=acm
#method=lazy
#python ../python/main.py -m ${method} -t 200 -k 300 -f sqrt -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

#dataset=reuters90
#method=lazy
#python ../python/main.py -m ${method} -t 200 -k 200 -f sqrt -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

#dataset=4uni
#method=lxt
#python ../python/main.py -m ${method} -t 200 -k 200 -f sqrt -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

#dataset=20ng
#method=lxt
#python ../python/main.py -m ${method} -t 200 -k 200 -f sqrt -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

#dataset=acm
#method=lxt
#python ../python/main.py -m ${method} -t 200 -k 300 -f sqrt -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

#dataset=reuters90
#method=lxt
#python ../python/main.py -m ${method} -t 200 -k 200 -f sqrt -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

dataset=4uni
method=broof
#python ../python/main.py -m ${method} -t 8 -i 200 -f 0.08 -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

dataset=20ng
method=broof
#python ../python/main.py -m ${method} -t 8 -i 200 -f log2 -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

dataset=acm
method=broof
#python ../python/main.py -m ${method} -t 8 -i 200 -f sqrt -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

dataset=reuters90
method=broof
#python ../python/main.py -m ${method} -t 8 -i 200 -f 0.08 -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

dataset=4uni
method=bert
#python ../python/main.py -m ${method} -t 8 -i 200 -f 0.3 -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

dataset=20ng
method=bert
#python ../python/main.py -m ${method} -t 5 -i 200 -f log2 -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

dataset=acm
method=bert
#python ../python/main.py -m ${method} -t 8 -i 200 -f sqrt -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

dataset=reuters90
method=bert
#python ../python/main.py -m ${method} -t 8 -i 200 -f 0.3 -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}