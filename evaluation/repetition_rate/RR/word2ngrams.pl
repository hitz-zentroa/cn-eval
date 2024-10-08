#!/usr/bin/perl

exit print "usage: \n\n word2ngrams.pl N\n" if $#ARGV!=0;

$sep="_#_";

$N = $ARGV[0];
if (!($N>0)) {
    printf stderr "Error: N must be a positive integer (passed <%s>)\n", $N;
    exit(0);
}

while ($in=<STDIN>) {
    chop($in); $in=~s/^\s+|\s+$//g;
    @tmp=split(/\s+/, $in);
    if ($#tmp>=$N-1) {
	for ($i=0; $i<=$#tmp-$N+1; $i++) {
	    for ($j=0; $j<$N-1; $j++) {
		printf "%s%s", $tmp[$i+$j], $sep;
	    }
	    printf "%s ", $tmp[$i+$N-1];
	}
    } else {
	for ($i=0; $i<$#tmp; $i++) {
	    printf "%s%s", $tmp[$i], $sep;
	}
	printf "%s", $tmp[$#tmp];
    }

    printf "\n";
}
