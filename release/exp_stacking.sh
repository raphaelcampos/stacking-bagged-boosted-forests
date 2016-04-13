dataset_dir=$1
output_dir=$2
n_jobs=$3
trials=5

# COMB SOTA
combsota(){
method=combsota
dataset=4uni
python ../python/app.py -c 0.125 -a 0.001 -k 30 -t 200 -g 1 -j ${n_jobs} --trials ${trials} -o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm '["svm","nb","knn","rf","xt"]' '["rf"]' --base_params '[{},{},{},{"max_features": 0.08, "n_estimators": 200, "criterion":"gini"},{"max_features": 0.3, "n_estimators": 200, "criterion": "entropy"}]' --meta_params '[{"max_features":"sqrt"}]' > ${output_dir}/grid_${method}_${dataset}

dataset=20ng
python ../python/app.py -c 0.125 -a 0.001 -k 10 -t 200 -g 1 -j ${n_jobs} --trials ${trials} -o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm '["svm","nb","knn","rf","xt"]' '["rf"]' --base_params '[{},{},{},{"max_features": "log2", "n_estimators": 200, "criterion":"gini"},{"max_features": "log2", "n_estimators": 200, "criterion": "gini"}]' --meta_params '[{"max_features":"sqrt"}]' > ${output_dir}/grid_${method}_${dataset}

dataset=acm
python ../python/app.py -c 0.5 -a 0.5 -k 30 -t 200 -g 1 -j ${n_jobs} --trials ${trials} -o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm '["svm","nb","knn","rf","xt"]' '["rf"]' --base_params '[{},{},{},{"max_features": "sqrt", "n_estimators": 200, "criterion":"gini"},{"max_features": "sqrt", "n_estimators": 200, "criterion": "gini"}]' --meta_params '[{"max_features":"sqrt"}]' > ${output_dir}/grid_${method}_${dataset}

dataset=reuters90
python ../python/app.py -c 0.5 -a 0.1 -k 10 -t 200 -g 1 -j ${n_jobs} --trials ${trials} -o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm '["svm","nb","knn","rf","xt"]' '["rf"]' --base_params '[{},{},{},{"max_features": 0.08, "n_estimators": 200, "criterion":"gini"},{"max_features": 0.3, "n_estimators": 200, "criterion": "gini"}]' --meta_params '[{"max_features":"sqrt"}]' > ${output_dir}/grid_${method}_${dataset}
}
# COMB_ALL
#python ../python/app.py -c 0.03125 -k 200 -t 200 -i 200 -f 0.15 -g 0 -j 8 datasets/webkb.svm '["knn","svm","rf","bert","broof","lazy","lxt"]' '["rf"]' --base_params '[{"n_neighbors":30}, {},{},{"n_trees":8},{"n_trees":8},{"max_features":"sqrt"},{"max_features":"sqrt"}]' --meta_params '[{"max_features":"sqrt"}]'

# Staking broof + lazy and rf as combiner
comb1(){
dataset_dir=$1
output_dir=$2
n_jobs=$3
trials=5
method=comb1
dataset=4uni
#python ../python/app.py -k 200 -t 200 -i 200 -f 0.08 -g 1 -j ${n_jobs} --trials ${trials} -o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm '["broof","lazy"]' '["rf"]' --base_params '[{"n_trees":8},{"max_features":"sqrt"}]' --meta_params '[{"max_features":"sqrt"}]' > ${output_dir}/grid_${method}_${dataset} &

dataset=20ng
#python ../python/app.py -k 200 -t 200 -i 200 -f log2 -g 1 -j ${n_jobs} --trials ${trials} -o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm '["broof","lazy"]' '["rf"]' --base_params '[{"n_trees":20},{"max_features":"sqrt"}]' --meta_params '[{"max_features":"sqrt"}]' > ${output_dir}/grid_${method}_${dataset} &

dataset=acm
#python ../python/app.py -k 300 -t 200 -i 200 -f sqrt -g 1 -j ${n_jobs} --trials ${trials} -o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm '["broof","lazy"]' '["rf"]' --base_params '[{"n_trees":8},{"max_features":"sqrt"}]' --meta_params '[{"max_features":"sqrt"}]' > ${output_dir}/grid_${method}_${dataset} &

dataset=reuters90
#python ../python/app.py -k 200 -t 200 -i 200 -f 0.08 -g 1 -j ${n_jobs} --trials ${trials} -o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm '["broof","lazy"]' '["rf"]' --base_params '[{"n_trees":8},{"max_features":"sqrt"}]' --meta_params '[{"max_features":"sqrt"}]' > ${output_dir}/grid_${method}_${dataset} &
}

comb2(){
	dataset_dir=$1
	output_dir=$2
	n_jobs=$3
	trials=$4
	method=comb2
	dataset=4uni
	python ../python/app.py -k 200 -t 200 -i 200 -f 0.3 -g 1 -j ${n_jobs} --trials ${trials} -o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm '["bert","lxt"]' '["rf"]' --base_params '[{"n_trees":8},{"max_features":"sqrt"}]' --meta_params '[{"max_features":"sqrt"}]' > ${output_dir}/grid_${method}_${dataset} &

	dataset=20ng
	python ../python/app.py -k 200 -t 200 -i 200 -f log2 -g 1 -j ${n_jobs} --trials ${trials} -o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm '["bert","lxt"]' '["rf"]' --base_params '[{"n_trees":5},{"max_features":"sqrt"}]' --meta_params '[{"max_features":"sqrt"}]' > ${output_dir}/grid_${method}_${dataset} &

	dataset=acm
	python ../python/app.py -k 300 -t 200 -i 200 -f sqrt -g 1 -j ${n_jobs} --trials ${trials} -o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm '["bert","lxt"]' '["rf"]' --base_params '[{"n_trees":8},{"max_features":"sqrt"}]' --meta_params '[{"max_features":"sqrt"}]' > ${output_dir}/grid_${method}_${dataset} &

	dataset=reuters90
	python ../python/app.py -k 200 -t 200 -i 200 -f 0.3 -g 1 -j ${n_jobs} --trials ${trials} -o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm '["bert","lxt"]' '["rf"]' --base_params '[{"n_trees":8},{"max_features":"sqrt"}]' --meta_params '[{"max_features":"sqrt"}]' > ${output_dir}/grid_${method}_${dataset} &
}

comb3(){
	dataset_dir=$1
	output_dir=$2
	n_jobs=$3
	trials=$4
	method=comb3
	dataset=4uni
	python ../python/app.py -k 200 -t 200 -i 200 -f 0.3 -g 1 -j ${n_jobs} --trials ${trials} -o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm '["broof","lazy","bert","lxt"]' '["rf"]' --base_params '[{"n_trees":8, "max_features":0.08},{"max_features":"sqrt"},{"n_trees":8},{"max_features":"sqrt", "max_features":0.3}]' --meta_params '[{"max_features":"sqrt"}]' > ${output_dir}/grid_${method}_${dataset} &

	dataset=20ng
	python ../python/app.py -k 200 -t 200 -i 200 -f log2 -g 1 -j ${n_jobs} --trials ${trials} -o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm '["broof","lazy","bert","lxt"]' '["rf"]' --base_params '[{"n_trees":20},{"max_features":"sqrt"},{"n_trees":5},{"max_features":"sqrt"}]' --meta_params '[{"max_features":"sqrt"}]' > ${output_dir}/grid_${method}_${dataset} &

	dataset=acm
	python ../python/app.py -k 300 -t 200 -i 200 -f sqrt -g 1 -j ${n_jobs} --trials ${trials} -o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm '["broof","lazy","bert","lxt"]' '["rf"]' --base_params '[{"n_trees":8},{"max_features":"sqrt"},{"n_trees":8},{"max_features":"sqrt"}]' --meta_params '[{"max_features":"sqrt"}]' > ${output_dir}/grid_${method}_${dataset} &

	dataset=reuters90
	python ../python/app.py -k 200 -t 200 -i 200 -f 0.3 -g 1 -j ${n_jobs} --trials ${trials} -o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm '["broof","lazy","bert","lxt"]' '["rf"]' --base_params '[{"n_trees":8, "max_features":0.08},{"max_features":"sqrt"},{"n_trees":8, "max_features":0.3},{"max_features":"sqrt"}]' --meta_params '[{"max_features":"sqrt"}]' > ${output_dir}/grid_${method}_${dataset} &
}

#comb2 ${dataset_dir} ${output_dir} ${n_jobs} ${trials}
comb3 ${dataset_dir} ${output_dir} ${n_jobs} ${trials}