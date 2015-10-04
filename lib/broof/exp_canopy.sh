#!/bin/bash

cur_path=`pwd`;

# FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME
# FIXME LIBLINEAR OR LIBSVM? CHANGE BELOW ! FIXME
# FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME

svm_path="/home/tsalles/Apps/libsvm-3.16";
svm_train="$svm_path/svm-train";
svm_test="$svm_path/svm-predict";
svm_scale="$svm_path/svm-scale";

function svm_classify {
  local treino=$1;
  local teste=$2;
  local testeOrig=$3;
  local round=$4;

#  $svm_scale -l 0 -u 1 -s ${output}/scale $treino > ${output}/tmp; mv ${output}/tmp $treino;
#  $svm_scale           -r ${output}/scale $teste  > ${output}/tmp; mv ${output}/tmp $teste ;

  local C=0; local G=0;
  cd $svm_path/tools;
    ./grid.py -log2c -5,15,2 -log2g -5,15,2 -svmtrain $svm_train -e 0.001 $cur_path/$treino > ${cur_path}/saidaGrid;
    C=`cat ${cur_path}/saidaGrid | tail -n 1 | awk '{print $1;}'`;
    G=`cat ${cur_path}/saidaGrid | tail -n 1 | awk '{print $2;}'`;
    rm *.out;
  cd $cur_path;

  echo "iter=$round c=$C g=$G" >> ${output}/ParamsSVM_BL;
  $svm_train -c $C -g $G -e 0.001 $treino ${output}/modelo_svm;
  $svm_test $teste ${output}/modelo_svm ${output}/predict_svm;
  ./createAffMatrix_origId.pl $teste $testeOrig ${output}/predict_svm $round >> ${output}/ResSVM_BL;
  rm ${output}/predict_svm ${output}/modelo_svm;
}

dataset=$1;
numFolds=$2;
output=$3;

./crossValidation $dataset ${output}/dados $numFolds `wc -l $dataset | awk '{print $1;}'`;
for (( i = 0; i < $numFolds; i++ ))
do
  echo "Validacao Cruzada: Iteracao $i.";
  rm -rf ${output}/treino* ${output}/teste* 2> /dev/null
  for (( j = 0; j < $numFolds; j++ ))
  do
    if [ $i -ne $j ]
    then
      cat ${output}/dados$j >> ${output}/treino;
    else
      cat ${output}/dados$j > ${output}/teste;
    fi
  done
exit;

# ./tsalles2libsvm ${output}/treino  > ${output}/treino_svm;
# ./tsalles2libsvm ${output}/teste   > ${output}/teste_svm; 
#  $svm_scale -l 0 -u 1 -s ${output}/scale ${output}/treino_svm > ${output}/tmp; mv ${output}/tmp ${output}/treino_svm;
#  $svm_scale           -r ${output}/scale ${output}/teste_svm  > ${output}/tmp; mv ${output}/tmp ${output}/teste_svm;
#  awk -f libsvm2tsalles.awk ${output}/treino_svm > ${output}/treino;
#  awk -f libsvm2tsalles.awk ${output}/teste_svm  > ${output}/teste;

#  # projection:
#  ./cluster/main ${output}/treino 0.0001 0.00001 > ${output}/tmp; mv ${output}/tmp ${output}/treino;
#  exit;

  echo "#$i" >> ${output}/ResLRF;
  ./tcpp -d ${output}/treino -t ${output}/teste -m rf-knn -x 0.05 -k 30 -n 200 >> ${output}/ResLRF;

  echo "#$i" >> ${output}/ResRF;
  ./tcpp -d ${output}/treino -t ${output}/teste -m rf     -x 0.05 -k 30 -n 200 >> ${output}/ResRF;

  echo "#$i" >> ${output}/ResKNN;
  ./tcpp -d ${output}/treino -t ${output}/teste -m knn -k 30 -R >> ${output}/ResKNN;

#  echo "#$i" >> ${output}/ResNB;
#  ./tcpp -d ${output}/treino -t ${output}/teste -m lnaivebayes >> ${output}/ResNB;

# ./tsalles2libsvm ${output}/treino  > ${output}/treino_svm;
# ./tsalles2libsvm ${output}/teste   > ${output}/teste_svm; 
# svm_classify ${output}/treino_svm ${output}/teste_svm ${output}/teste $i;

done
rm ${output}/treino* ${output}/teste* ${output}/dados*;
