#!/usr/bin/perl -w
use strict;

my $arqTest = $ARGV[0];
my $arqPred = $ARGV[1];
my $round = $ARGV[2];

open(HANDLER, $arqPred);
my $docId=0;
my %hashPred = ();
foreach my $ln (<HANDLER>) {
  chomp $ln;
  $hashPred{$docId}=$ln;
  $docId++;
}
close HANDLER;

print "#$round\n";

open(HANDLER, $arqTest);
$docId=0;
foreach my $ln (<HANDLER>) {
  my @line = split(/ /, $ln);
  chomp @line;

  my $trueClass = $line[0];
  chomp $trueClass;

  print "$docId CLASS=$trueClass CLASS=$hashPred{$docId}:1\n";

  $docId++;
}
close HANDLER;

exit;
