cur_path=`pwd`;

dataset=$1;


for method in 'broof'; do
	# get results
	../test_${method} datasets/${dataset}/treino2_temp datasets/$dataset/teste2_temp ../venn-diagrams/${dataset}/t0/resul_${method}

	# replace : for white space
	cat ../venn-diagrams/${dataset}/t0/resul_${method} | perl -pe 's/:/\ /g' > ../venn-diagrams/${dataset}/t0/resul_${method}_s
	mv ../venn-diagrams/${dataset}/t0/resul_${method}_s ../venn-diagrams/${dataset}/t0/resul_${method}
done

python ../venn.py ../venn-diagrams/${dataset}/t0/resul_lazy ../venn-diagrams/${dataset}/t0/resul_rf ../venn-diagrams/${dataset}/t0/resul_knn

python ../venn.py ../venn-diagrams/${dataset}/t0/resul_lazy ../venn-diagrams/${dataset}/t0/resul_rf ../venn-diagrams/${dataset}/t0/resul_broof