dump_folder=$1
n_jobs=$2
trials=$3

command="python ../python/app.py -c 0.125 -k 30 -t 200 -f 0.15 -g 0 -j 8 --trials 5 datasets/webkb.svm '[\"knn\",\"rf\",\"nb\"]' '[\"rf\"]' --base_params '[{\"n_neighbors\":30},{},{}]' --meta_params '[{\"max_features\":\"sqrt\"}]'"

echo ${command}

eval $command

# COMB SOTA
#python ../python/app.py -c 0.125 -k 30 -t 200 -f 0.15 -g 0 -j 8 datasets/webkb.svm '["knn","svm","rf","nb"]' '["rf"]' --meta_params '[{"max_features":"sqrt"}]'

# COMB_ALL
#python ../python/app.py -c 0.03125 -k 200 -t 200 -i 200 -f 0.15 -g 0 -j 8 datasets/webkb.svm '["knn","svm","rf","bert","broof","lazy","lxt"]' '["rf"]' --base_params '[{"n_neighbors":30}, {},{},{"n_trees":8},{"n_trees":8},{"max_features":"sqrt"},{"max_features":"sqrt"}]' --meta_params '[{"max_features":"sqrt"}]'