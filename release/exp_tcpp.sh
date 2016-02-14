cur_path=`pwd`;

foldDir=$1;
resultFold=$2;
n_trees=$3;
max_features=$4;

#rm -r "${resultFold}/"*
for f in `ls "${foldDir}/treino"*`
do 
	bfn=`basename $f`;
	suffix=${bfn:8};
	tmp=`echo $bfn | awk '{split($1, a, "_"); print a[1]}'`;
	i=${tmp:6};

	# pro linear, descomente a linha abaixo. Pro nao-linear mantenha comentada.
	lin="lin";
	
	dataset=`basename ${foldDir}`
	for method in knn broof lazy rf;
	do
	  ../test_${method} "${cur_path}/${foldDir}/treino${i}_$suffix" "${cur_path}/${foldDir}/teste${i}_$suffix" "${resultFold}/saida_tmp" ${n_trees} ${max_features};

	  output_file="${resultFold}/results_${method}_${dataset}"
	  echo "#${i}" >> ${output_file}
	  cat "${resultFold}/saida_tmp" >> ${output_file}
	done
	
  	rm "${resultFold}/saida_tmp"
done