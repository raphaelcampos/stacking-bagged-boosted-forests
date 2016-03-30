python ../python/app.py -c 0.03125 -k 30 -t 200 -f 0.15 -g 0 -j 8 datasets/webkb.svm '["knn","rf","nb"]' '["rf"]' --base_params '[{"n_neighbors":30},{"max_features":"sqrt"},{}]' --meta_params '[{"max_features":"sqrt"}]'

#python ../python/app.py -c 0.03125 -k 30 -t 200 -f 0.15 -g 0 -j 8 datasets/webkb.svm '["knn","svm","rf","nb"]' '["rf"]' --base_params '[{"n_neighbors":30},{},{"max_features":"sqrt"},{}]' --meta_params '[{"max_features":"sqrt"}]'

# COMB_ALL
#python ../python/app.py -c 0.03125 -k 200 -t 200 -i 200 -f 0.15 -g 0 -j 8 datasets/webkb.svm '["knn","svm","rf","bert","broof","lazy","lxt"]' '["rf"]' --base_params '[{"n_neighbors":30}, {},{},{"n_trees":8},{"n_trees":8},{"max_features":"sqrt"},{"max_features":"sqrt"}]' --meta_params '[{"max_features":"sqrt"}]'