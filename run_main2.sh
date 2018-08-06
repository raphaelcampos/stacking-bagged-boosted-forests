# folds=('fold1') #'fold2' 'fold3' 'fold4' 'fold5')
folds=('fold1' 'fold2' 'fold3' 'fold4' 'fold5')
inputs=('no_PP' 'PPraw' 'PPweighted' 'PPweighted_only')
dataset='yelp'
# inputs=('no_PP' 'PPweighted')
for input in ${inputs[@]}
do
	for fold in ${folds[@]}
	do
	echo ${fold} 
	echo ${input}
	
		python python/main2.py -m rf -t 200 -f 0.15 release/datasets/performance_prediction/train_${dataset}_${input}_${fold}  release/datasets/performance_prediction/test_${dataset}_${input}_${fold}
	done
done
