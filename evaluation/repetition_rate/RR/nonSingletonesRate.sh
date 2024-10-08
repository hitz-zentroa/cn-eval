#! /bin/bash

# This script computes the rate of non-singletone events (N-grams with
# n=1,2,3,4) of the input textual file. Computation is averaged on
# equally-sized non-overlapped sub-samples, spanning the whole text
#
# Its results can be used to compute the repetition rate of the text
# as geometric mean by means of /hltsrv0/cettolo/bin/geomMean.pl


bindir=`dirname $0`
#irstlmdir=/ikerlariak/ragerri/experiments/conan/repetition_rate/RR/irstlm-5.80.01
irstlmdir=/ixadata/soft/rhel7/irstlm-6.00.01

# Size of subsamples:
sampleSize=1000

fn=$1
out=`basename $fn`
echo $out
for n in 1 2 3 4
do
ls $fn 
echo "n=$n"
totN=0
totC=0

offset=0
while [ $offset -gt -1 ]; do
offset=`perl $bindir/selectSubSample.pl $fn __subsample__$$ $sampleSize $offset`
cat __subsample__$$ | perl $bindir/word2ngrams.pl $n > __tmp__$$
$irstlmdir/bin/dict -cs=3 -f=yes -i=__tmp__$$ -sort=yes -c=yes >& __tmp.histo__$$
N=`egrep '^>0' __tmp.histo__$$ | awk '{print $2}'`
totN=`expr $totN + $N`
C=`egrep '^>1' __tmp.histo__$$ | awk '{print $2}'`
totC=`expr $totC + $C`
done
echo $totC $totN | awk '{print $0,100*$1/$2}'
echo ""
done

rm __tmp__$$ __tmp.histo__$$ __subsample__$$
exit

