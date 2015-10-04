BEGIN{
  id=1;
}

{
  l=id";"$1;
  for (i=2; i <= NF; i++) {
    split($i, a, ":");
    l=l";"a[1]";"a[2];
  }
  print l;
  id++;
}
