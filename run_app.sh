# datasets=('amazon' 'bbc' 'debate' 'digg' 'myspace' 'nyt' 'tweets' 'yelp' 'youtube')
# datasets=('debate' 'digg' 'myspace' 'nyt' 'tweets')
datasets=('yelp' 'youtube')
for dataset in ${datasets[@]}
do
	python python/main.py -m rf -t 200 -f 0.08 -p rfr --trials 5 -j 4 --dump_meta_level release/datasets/%s_meta_${dataset}_%s_fold%s release/datasets/sentiment_analysis/tf/${dataset}.svm --pp_features release/datasets/sentiment_analysis/${dataset}_pp_meta_features.svm
	python python/main.py -m lsvm --cv 5 -g 0 -p rfr --trials 5 -j 4 --dump_meta_level release/datasets/%s_meta_${dataset}_%s_fold%s release/datasets/sentiment_analysis/tf/${dataset}.svm --pp_features release/datasets/sentiment_analysis/${dataset}_pp_meta_features.svm
	python python/main.py -m broof --cv 5 -g 1 -p rfr --trials 5 -j 4 --dump_meta_level release/datasets/%s_meta_${dataset}_%s_fold%s release/datasets/sentiment_analysis/tf/${dataset}.svm --pp_features release/datasets/sentiment_analysis/${dataset}_pp_meta_features.svm
	# python python/main.py -m bert --cv 5 -g 1 -p rfr --trials 5 -j 4 --dump_meta_level release/datasets/%s_meta_${dataset}_%s_fold%s release/datasets/sentiment_analysis/tf/${dataset}.svm --pp_features release/datasets/sentiment_analysis/${dataset}_pp_meta_features.svm
done
