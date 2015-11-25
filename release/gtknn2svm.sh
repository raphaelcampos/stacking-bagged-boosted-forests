cur_path=`pwd`;

foldDir=$1;
#resultOut=$2;

#rm "${resultOut}"

mkdir "${cur_path}/${foldDir}/svm"
for f in `ls "${cur_path}/${foldDir}/treino"*`
do 
  bfn=`basename $f`;
  suffix=${bfn:8};
  tmp=`echo $bfn | awk '{split($1, a, "_"); print a[1]}'`;
  i=${tmp:6};

  # pro linear, descomente a linha abaixo. Pro nao-linear mantenha comentada.
  lin="lin";

  ./gtknn2svm "${cur_path}/${foldDir}/treino${i}_$suffix" > "${cur_path}/${foldDir}/svm/treino${i}_$suffix"
  ./gtknn2svm "${cur_path}/${foldDir}/teste${i}_$suffix" > "${cur_path}/${foldDir}/svm/teste${i}_$suffix"

done
