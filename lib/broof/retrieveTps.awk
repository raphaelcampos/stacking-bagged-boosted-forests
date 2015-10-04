BEGIN {

  min=9999;
  max=-1;

}

{

  if ($2 < min) min = $2;
  if ($2 > max) max = $2;

}

END {

  print (max - min)

}
