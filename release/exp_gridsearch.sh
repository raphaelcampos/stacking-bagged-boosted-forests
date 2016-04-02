dataset_dir=$1
output_dir=$2
n_jobs=$3
trials=5

datasets=('20ng' 'reuters90' 'acm')

################################################################################
#				 					4UNI 									   #
################################################################################ 
for dataset in ${datasets}
do
	#Suport Vector Machine (LIBSVM)
	method=svm
	python ../python/main.py -m ${method} --cv 5 -g 0 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_4uni ${dataset_dir}/webkb.svm > ${output_dir}/grid_${method}_4uni

	# kNearestNeighbors
	method=knn
	python ../python/main.py -m ${method} --cv 5 -g 0 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_4uni ${dataset_dir}/webkb.svm > ${output_dir}/grid_${method}_4uni

	# Naive Bayes
	method=nb
	python ../python/main.py -m ${method} --cv 5 -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_4uni ${dataset_dir}/webkb.svm > ${output_dir}/grid_${method}_4uni

	# Random Forest
	method=rf
	python ../python/main.py -m ${method} --cv 5 -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_4uni ${dataset_dir}/webkb.svm > ${output_dir}/grid_${method}_4uni

	# Extremely Randomized Trees
	method=xt
	python ../python/main.py -m ${method} --cv 5 -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_4uni ${dataset_dir}/webkb.svm > ${output_dir}/grid_${method}_4uni

	# Lazy Random Forest
	#method=lazy
	#python ../python/main.py -m ${method} --cv 5 -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_4uni ${dataset_dir}/webkb.svm > ${output_dir}/grid_${method}_4uni

	# Lazy Extremely Randomized Trees
	#method=lxt
	#python ../python/main.py -m ${method} --cv 5 -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_4uni ${dataset_dir}/webkb.svm > ${output_dir}/grid_${method}_4uni

	# BROOF
	#method=broof
	#python ../python/main.py -m ${method} --cv 5 -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_4uni ${dataset_dir}/webkb.svm > ${output_dir}/grid_${method}_4uni

	# BERT
	#method=bert
	#python ../python/main.py -m ${method} --cv 5 -g 1 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${method}_4uni ${dataset_dir}/webkb.svm > ${output_dir}/grid_${method}_4uni
done
################################################################################