dataset_dir=$1
output_dir=$2
n_jobs=$3
trials=5

datasets=('4uni' '20ng' 'acm' 'reuters90')
norms=('max' 'l1' 'l2')

################################################################################
#				 					${dataset} 								#
################################################################################ 
for dataset in ${datasets[@]}
do
	#Suport Vector Machine (LIBSVM)
	method=svm
	for norm in ${norms[@]}
	do
		python ../python/main.py -m ${method} --cv 5 -n ${norm} -g 0 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${norm}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${norm}_${dataset}
	done
	
	norm=none
	python ../python/main.py -m ${method} --cv 5 -g 0 -j ${n_jobs} --trials ${trials} --o ${output_dir}/results_${norm}_${dataset} ${dataset_dir}/${dataset}.svm > ${output_dir}/grid_${norm}_${dataset}
done