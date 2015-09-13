cur_path=`pwd`;

foldDir=$1;
resultOut=$2;

rm "${resultOut}"

for f in `ls "${cur_path}/${foldDir}/treino"*`
do 
  bfn=`basename $f`;
  suffix=${bfn:8};
  tmp=`echo $bfn | awk '{split($1, a, "_"); print a[1]}'`;
  i=${tmp:6};

  # pro linear, descomente a linha abaixo. Pro nao-linear mantenha comentada.
  lin="lin";

  ./lazynn_rf "${cur_path}/${foldDir}/treino${i}_$suffix" "${cur_path}/${foldDir}/teste${i}_$suffix" -m knn_rf -k 30 --trial ${i} -r "${resultOut}" -a;

done

./stats "${resultOut}"