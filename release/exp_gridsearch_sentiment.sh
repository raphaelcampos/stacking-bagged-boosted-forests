#!/bin/bash

dataset_dir=$1
output_dir=$2
n_jobs=$3
trials=10

datasets=('amazon bbc debate digg myspace tweets twitter nyt yelp youtube')

################################################################################
# dataset 									   #
################################################################################ 
for dataset in ${datasets}
do
	echo ${dataset}
	#Suport Vector Machine (LINEARSVM)
	method=lsvm
	python ../python/main.py -m ${method} --cv 5 -j ${n_jobs} -g 0 -n l2 --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

	# kNearestNeighbors
	method=knn
	python ../python/main.py -m ${method} --cv 5 -j ${n_jobs} -g 0 -n l2 --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

	# Naive Bayes
	method=nb
	python ../python/main.py -m ${method} --cv 5 -j ${n_jobs} -g 0 -n l2 --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

	# Decision Tree
	method=dt
	python ../python/main.py -m ${method} --cv 5 -j ${n_jobs} -g 0 -n l2 --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

	# Random Forest
	method=rf
	python ../python/main.py -m ${method} --cv 5 -j ${n_jobs} -g 0 -n l2 --trials ${trials} --o ${output_dir}/results_${method}-l2_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}-l2_${dataset}

	# Extremely Randomized Trees
	method=xt
	python ../python/main.py -m ${method} --cv 5 -j ${n_jobs} -g 0 -n l2 --trials ${trials} --o ${output_dir}/results_${method}-l2_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}-l2_${dataset}

	# Lazy Random Forest
	method=lazy
	python ../python/main.py -m ${method} --cv 5 -j ${n_jobs} -g 0 -n l2 --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}


	# Lazy Extremely Randomized Trees
	method=lxt
	python ../python/main.py -m ${method} --cv 5 -j ${n_jobs} -g 0 -n l2 --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

	# BROOF
	method=broof
	#python ../python/main.py -m ${method} --cv 5 -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

	# BERT
	#method=bert
	#python ../python/main.py -m ${method} --cv 5 -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}
done
################################################################################

# BERT
dataset=amazon
method=bert
#python ../python/main.py -m ${method} -g 0 -n l2 -t 20 -i 100 --learning_rate 0.1 -f log2 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

dataset=bbc
#python ../python/main.py -m ${method} -g 0 -n l2 -t 20 -i 100 --learning_rate 0.1 -f sqrt  -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

dataset=debate
#python ../python/main.py -m ${method} -t 20 -i 100 --learning_rate 0.1 -f sqrt -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

dataset=digg
#python ../python/main.py -m ${method} -t 20 -i 100 --learning_rate 0.1 -f sqrt -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

dataset=myspace
#python ../python/main.py -m ${method} -t 20 -i 100 --learning_rate 0.1 -f log2 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

dataset=tweets
#python ../python/main.py -m ${method} -t 20 -i 100 --learning_rate 0.1 -f log2 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

dataset=twitter
#python ../python/main.py -m ${method} -t 20 -i 100 --learning_rate 0.1 -f log2 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

dataset=nyt
#python ../python/main.py -m ${method} -t 20 -i 100 --learning_rate 0.1 -f log2 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

dataset=yelp
#python ../python/main.py -m ${method} -t 20 -i 100 --learning_rate 0.1 -f sqrt -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

dataset=youtube
#python ../python/main.py -m ${method} -t 20 -i 100 --learning_rate 0.1 -f 0.08 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}

# BERT
dataset=amazon
method=broof
#python ../python/main.py -m ${method} -g 0 -n l2 -t 20 -i 100 --learning_rate 0.1 -f log2 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}$

dataset=bbc
#python ../python/main.py -m ${method} -t 20 -g 0 -n l2 -i 100 --learning_rate 0.1 -f sqrt  -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}$

dataset=debate
#python ../python/main.py -m ${method} -t 20 -i 100 --learning_rate 0.1 -f 0.08 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}$

dataset=digg
#python ../python/main.py -m ${method} -t 20 -i 100 --learning_rate 0.1 -f log2 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}$

dataset=myspace
#python ../python/main.py -m ${method} -t 20 -i 100 --learning_rate 0.1 -f log2 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}$

dataset=tweets
#python ../python/main.py -m ${method} -t 20 -i 100 --learning_rate 0.1 -f log2 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}$

dataset=twitter
#python ../python/main.py -m ${method} -t 20 -i 100 --learning_rate 0.1 -f sqrt -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}$

dataset=nyt
#python ../python/main.py -m ${method} -t 20 -i 100 --learning_rate 0.1 -f log2 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}$

dataset=yelp
#python ../python/main.py -m ${method} -t 20 -i 100 --learning_rate 0.1 -f sqrt -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}$

dataset=youtube
#python ../python/main.py -m ${method} -t 20 -i 100 --learning_rate 0.1 -f 0.15 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${method}_${dataset}$

