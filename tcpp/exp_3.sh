#!/bin/bash

dataSet=$1;
outputFold=$2;
numFolds=$3;
p_k=$4;

mkdir $outputFold;

rm $outputFold/_res* $outputFold/_dds*;
./crossValidation $dataSet $outputFold/_dds $numFolds `wc -l $dataSet`;

for (( j = 0; j < $numFolds; j++ ))
do
 rm $outputFold/_train $outputFold/_test;
 for (( k = 0; k < $numFolds; k++ ))
 do
   if [ $j -ne $k ]
   then
     cat $outputFold/_dds$k >> $outputFold/_train;
   else
     cat $outputFold/_dds$k > $outputFold/_test;
   fi
 done
 #echo "#$j" >> $outputFold/_res_knn;
 #./tcpp -d $outputFold/_train -t $outputFold/_test -m rf-knn -n 200 -x 0.03 -k $p_k >> $outputFold/_res_knn;
 echo "#$j" >> $outputFold/_res_kr;
 ./tcpp -d $outputFold/_train -t $outputFold/_test -m rf-krand -n 200 -x 0.03 -k $p_k >> $outputFold/_res_kr;
done
./stats $outputFold/_res_kr > ~/Dropbox/RandomProj_Victor/${outputFold}
