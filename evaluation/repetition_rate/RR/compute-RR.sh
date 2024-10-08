#! /bin/bash
bindir=`dirname $0`

fn=$1
name=`basename $fn`
echo "$name RR: " | tr -d '\012'
sh $bindir/nonSingletonesRate.sh $fn > __tmp__$$
egrep '^[0-9]+ [0-9]+ [0-9\.]+' __tmp__$$ | awk '{print $NF}' | tr '\012' ' ' | perl $bindir/geomMean.pl

rm __tmp__$$
