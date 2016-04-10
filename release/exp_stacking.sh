dataset_dir=$1
output_dir=$2
n_jobs=$3
trials=5

command="python ../python/app.py -c 0.125 -k 30 -t 200 -f 0.15 -g 0 -j 8 --trials 5 datasets/webkb.svm '[\"knn\",\"rf\",\"nb\"]' '[\"rf\"]' --base_params '[{\"n_neighbors\":30},{},{}]' --meta_params '[{\"max_features\":\"sqrt\"}]'"

#echo ${command}

# Staking broof + lazy and rf as combiner
method=comb1
dataset=4uni
python ../python/app.py -k 200 -t 200 -i 200 -f 0.08 -g 1 -j ${n_jobs} --trials ${trials} -o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm '["broof","lazy"]' '["rf"]' --base_params '[{"n_trees":8},{"max_features":"sqrt"}]' --meta_params '[{"max_features":"sqrt"}]' > ${output_dir}/grid_${method}_${dataset} &

dataset=20ng
python ../python/app.py -k 200 -t 200 -i 200 -f log2 -g 1 -j ${n_jobs} --trials ${trials} -o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm '["broof","lazy"]' '["rf"]' --base_params '[{"n_trees":8},{"max_features":"sqrt"}]' --meta_params '[{"max_features":"sqrt"}]' > ${output_dir}/grid_${method}_${dataset} &

dataset=acm
python ../python/app.py -k 300 -t 200 -i 200 -f sqrt -g 1 -j ${n_jobs} --trials ${trials} -o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm '["broof","lazy"]' '["rf"]' --base_params '[{"n_trees":8},{"max_features":"sqrt"}]' --meta_params '[{"max_features":"sqrt"}]' > ${output_dir}/grid_${method}_${dataset} &

dataset=reuters90
python ../python/app.py -k 200 -t 200 -i 200 -f 0.08 -g 1 -j ${n_jobs} --trials ${trials} -o ${output_dir}/results_${method}_${dataset} ${dataset_dir}/${dataset}.svm '["broof","lazy"]' '["rf"]' --base_params '[{"n_trees":8},{"max_features":"sqrt"}]' --meta_params '[{"max_features":"sqrt"}]' > ${output_dir}/grid_${method}_${dataset} &

# COMB SOTA
#python ../python/app.py -c 0.125 -k 30 -t 200 -f 0.15 -g 0 -j 8 datasets/webkb.svm '["knn","svm","rf","nb"]' '["rf"]' --meta_params '[{"max_features":"sqrt"}]'

# COMB_ALL
#python ../python/app.py -c 0.03125 -k 200 -t 200 -i 200 -f 0.15 -g 0 -j 8 datasets/webkb.svm '["knn","svm","rf","bert","broof","lazy","lxt"]' '["rf"]' --base_params '[{"n_neighbors":30}, {},{},{"n_trees":8},{"n_trees":8},{"max_features":"sqrt"},{"max_features":"sqrt"}]' --meta_params '[{"max_features":"sqrt"}]'