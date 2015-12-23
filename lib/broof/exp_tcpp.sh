#!/bin/bash

export LANG=C
export OMP_NUM_THREADS=24

cur_path=`pwd`;
svm_path="/home/thiagosalles/liblinear-1.94";
svm_train="$svm_path/train";
svm_test="$svm_path/predict";
svm_scale="$svm_path/svm-scale";

function svm_classify {
  local treino=$1;
  local teste=$2;
  local testeOrig=$3;
  local round=$4;
  local s=$5;

#  $svm_scale -l 0 -u 1 -s ${output}/scale $treino > ${output}/tmp; mv ${output}/tmp $treino;
#  $svm_scale           -r ${output}/scale $teste  > ${output}/tmp; mv ${output}/tmp $teste ;

  local C=0; local G=0;
  cd $svm_path/tools;
    ./grid.py -log2c -5,15,2 -log2g null -svmtrain $svm_train -e 0.001 $cur_path/$treino > ${cur_path}/saidaGrid 2>/dev/null;
    C=`cat ${cur_path}/saidaGrid | tail -n 1 | awk '{print $1;}'`;
#    G=`cat ${cur_path}/saidaGrid | tail -n 1 | awk '{print $2;}'`;
    rm *.out;
  cd $cur_path;

  echo "iter=$round c=$C" >> ${output}/ParamsSVM_BL-$s;
  $svm_train -c $C -e 0.001 $treino ${output}/modelo_svm 2>/dev/null;
  $svm_test $teste ${output}/modelo_svm ${output}/predict_svm 2>/dev/null;
  ./createAffMatrix.pl $teste ${output}/predict_svm $round >> ${output}/ResSVM-$s;
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

#  cp ${output}/treino ${output}/treino.bkp;
#  for s in `seq 1.0 -0.1 1.0`;
#  do

#    ./undersample ${output}/treino $s > ${output}/treino_smp;
#    mv ${output}/treino_smp ${output}/treino;

    echo "#$i" >> ${output}/ResBRF;
    time ./tcpp -d ${output}/treino -t ${output}/teste -m rf-boosted -x 0.0005 -H 0 -n 100 >> ${output}/ResBRF 2> err$i;

#    echo "#$i" >> ${output}/ResRF;
#    time ./tcpp -d ${output}/treino -t ${output}/teste -m rf -x 0.0005 -H 0 -n 200 >> ${output}/ResRF;
  
    #./tsalles2libsvm ${output}/treino  > ${output}/treino_svm;
    #./tsalles2libsvm ${output}/teste   > ${output}/teste_svm; 
    #$svm_scale -l 0 -u 1 -s ${output}/scale ${output}/treino_svm > ${output}/tmp; mv ${output}/tmp ${output}/treino_svm;
    #$svm_scale           -r ${output}/scale ${output}/teste_svm  > ${output}/tmp; mv ${output}/tmp ${output}/teste_svm;
    #awk -f libsvm2tsalles.awk ${output}/treino_svm > ${output}/treino.svm;
    #awk -f libsvm2tsalles.awk ${output}/teste_svm  > ${output}/teste.svm;
  
#    echo "#$i" >> ${output}/ResLRF-$s-0.8;
#    time ./tcpp -d ${output}/treino -t ${output}/teste -m rf-krand -x 0.8 -k 30 -n 200 >> ${output}/ResLRF-$s-0.8;
  
#    echo "#$i" >> ${output}/ResDT-$s;
#    time ./tcpp -d ${output}/treino -t ${output}/teste -m rf     -x 0.5 -k 30 -n 1 >> ${output}/ResDT-$s;
  
    echo "#$i" >> ${output}/ResKNN;
    time ./tcpp -d ${output}/treino -t ${output}/teste -m knn -k 30 >> ${output}/ResKNN;
  
#    echo "#$i" >> ${output}/ResNB-$s;
#    time ./tcpp -d ${output}/treino -t ${output}/teste -m lnaivebayes -a 0.01 -l 0.001 >> ${output}/ResNB-$s;
  
#    ./tsalles2libsvm ${output}/treino  > ${output}/treino_svm;
#    ./tsalles2libsvm ${output}/teste   > ${output}/teste_svm; 
#    time svm_classify ${output}/treino_svm ${output}/teste_svm ${output}/teste $i $s;

#    cp ${output}/treino.bkp ${output}/treino;
#  done
  
done
rm ${output}/treino* ${output}/teste* ${output}/dados*;
