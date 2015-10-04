#!/bin/bash

m=$1;
dataSet=$2;
outputFold=$3;
numFolds=$4;
p_k=$5;

mkdir $outputFold;

rm $outputFold/_res $outputFold/_dds*;
./crossValidation $dataSet $outputFold/_dds $numFolds `wc -l $dataSet`;
for (( j = 0; j < 1; j++ ))
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
 echo "#$j" >> $outputFold/_res;
 time ./tcpp -d $outputFold/_train -t $outputFold/_test -m $m -n 200 -x 0.03 -k $p_k >> $outputFold/_res;
done
