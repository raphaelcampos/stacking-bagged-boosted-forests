dataset_dir=$1
output_dir=$2
n_jobs=$3
trials=5

dataset=acm
execute(){
	m=$1
	f=$2
	t=$3
	i=$4
	lr=$5
	dataset=$6
	python ../python/main.py -m ${m} ${dataset_dir}/${dataset}.svm -f ${f} -t ${t} -i ${i} --learning_rate ${lr} --trials ${trials} -j ${n_jobs} --dump_meta_level ${output_dir}/%s_%s_meta${dataset}_fold%d.svm
}

# RF
execute rf 0.08 200 1 1.0 4uni
execute rf sqrt 200 1 1.0 acm
execute rf log2 200 1 1.0 20ng
execute rf sqrt 200 1 1.0 reuters90

# BROOF
execute broof 0.08 8 200 1.0 4uni
execute broof sqrt 8 200 1.0 acm
execute broof log2 8 200 1.0 20ng
execute broof sqrt 8 200 1.0 reuters90

# BERT
execute bert 0.3 8 200 1.0 4uni
execute bert sqrt 8 200 1.0 acm
execute bert log2 8 200 1.0 20ng
execute bert sqrt 8 200 1.0 reuters90