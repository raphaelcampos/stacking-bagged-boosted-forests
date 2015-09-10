#!/bin/bash

cur_path=`pwd`;
svm_path="/home/tsalles/Apps/liblinear-1.92/";
svm_train="$svm_path/train";
svm_test="$svm_path/predict";
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
    ./grid.py -log2c -5,15,2 -log2g 1,1,1 -svmtrain $svm_train -e 0.001 $cur_path/$treino > ${cur_path}/saidaGrid 2>/dev/null;
    C=`cat ${cur_path}/saidaGrid | tail -n 1 | awk '{print $1;}'`;
#    G=`cat ${cur_path}/saidaGrid | tail -n 1 | awk '{print $2;}'`;
    rm *.out;
  cd $cur_path;

  echo "iter=$round c=$C" >> ${output}/ParamsSVM_BL;
  $svm_train -c $C -e 0.001 $treino ${output}/modelo_svm 2>/dev/null;
  $svm_test $teste ${output}/modelo_svm ${output}/predict_svm 2>/dev/null;
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

#  ./undersample ${output}/treino 0.3 > ${output}/treino_smp;
#  mv ${output}/treino_smp ${output}/treino;

#  echo "#$i" >> ${output}/ResRF;
#  time ./tcpp -d ${output}/treino -t ${output}/teste -m rf -x 0.05 -n 100 >> ${output}/ResRF;

# ./tsalles2libsvm ${output}/treino  > ${output}/treino_svm;
# ./tsalles2libsvm ${output}/teste   > ${output}/teste_svm; 
#  $svm_scale -l 0 -u 1 -s ${output}/scale ${output}/treino_svm > ${output}/tmp; mv ${output}/tmp ${output}/treino_svm;
#  $svm_scale           -r ${output}/scale ${output}/teste_svm  > ${output}/tmp; mv ${output}/tmp ${output}/teste_svm;
#  awk -f libsvm2tsalles.awk ${output}/treino_svm > ${output}/treino;
#  awk -f libsvm2tsalles.awk ${output}/teste_svm  > ${output}/teste;

#  echo "#$i" >> ${output}/ResLRF;
#  time ./tcpp -d ${output}/treino -t ${output}/teste -m rf-knn -x 0.5 -k 30 -n 200 >> ${output}/ResLRF;

#  echo "#$i" >> ${output}/ResRF;
#  time ./tcpp -d ${output}/treino -t ${output}/teste -m rf     -x 0.5 -k 30 -n 200 >> ${output}/ResRF;

#  echo "#$i" >> ${output}/ResKNN;
#  time ./tcpp -d ${output}/treino -t ${output}/teste -m knn -k 30 >> ${output}/ResKNN;
#
#  echo "#$i" >> ${output}/ResNB;
#  time ./tcpp -d ${output}/treino -t ${output}/teste -m lnaivebayes -a 0.001 -l 0.0001 >> ${output}/ResNB;
#
# ./tsalles2libsvm ${output}/treino  > ${output}/treino_svm;
# ./tsalles2libsvm ${output}/teste   > ${output}/teste_svm; 
# time svm_classify ${output}/treino_svm ${output}/teste_svm ${output}/teste $i;

done
rm ${output}/treino* ${output}/teste* ${output}/dados*;
