# folds=('fold1') #'fold2' 'fold3' 'fold4' 'fold5')
folds=('fold1' 'fold2' 'fold3' 'fold4' 'fold5')
inputs=('no_PP' 'PPraw' 'PPweighted' 'PPweighted_only' 'PPmetafeatures_only')
datasets=('amazon' 'bbc' 'debate' 'digg' 'myspace' 'nyt' 'tweets' 'yelp' 'youtube')
# datasets=('amazon')

for dataset in ${datasets[@]}
	do
	for fold in ${folds[@]}
	do
		for input in ${inputs[@]}
		do
			for seed in 1 2 3 4 42 41 40 39 38 37
			do
			echo ${fold} >> results/${dataset}
			echo ${input} >> results/${dataset}
			
				python python/main2.py -m rf -t 200 -f 0.15 release/datasets/performance_prediction/train_${dataset}_${input}_${fold}  release/datasets/performance_prediction/test_${dataset}_${input}_${fold} -s ${seed} >> results/${dataset}
			done
		done
	done
done