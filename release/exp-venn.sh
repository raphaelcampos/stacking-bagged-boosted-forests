cur_path=`pwd`;

dataset=$1;
trial=$2;
trees=$3

for method in 'broof'; do
	# get results
	../test_${method} datasets/${dataset}/treino${trial}_temp datasets/$dataset/teste${trial}_temp ../venn-diagrams/${dataset}/t${trial}/resul_${method} ${trees}

	# replace : for white space
	cat ../venn-diagrams/${dataset}/t${trial}/resul_${method} | perl -pe 's/:/\ /g' > ../venn-diagrams/${dataset}/t${trial}/resul_${method}_s
	mv ../venn-diagrams/${dataset}/t${trial}/resul_${method}_s ../venn-diagrams/${dataset}/t${trial}/resul_${method}
done
 
python ../venn.py -c --labels lazy,rf,knn ../venn-diagrams/${dataset}/t${trial}/resul_lazy ../venn-diagrams/${dataset}/t${trial}/resul_rf ../venn-diagrams/${dataset}/t${trial}/resul_knn

python ../venn.py -c --labels lazy,rf,broof ../venn-diagrams/${dataset}/t${trial}/resul_lazy ../venn-diagrams/${dataset}/t${trial}/resul_rf ../venn-diagrams/${dataset}/t${trial}/resul_broof
