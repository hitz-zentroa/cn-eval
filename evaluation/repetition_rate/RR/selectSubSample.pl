#!/usr/local/irst/bin/perl

# select sentences from offset up to wordN words from inFile and store them into outFile
# returns the index of the sentence following the last selected sentence, -1 if wordN words cannot be selected

exit print "usage: \n\n selectSubSample.pl inFile outFile wordN [offset]\n\n" if $#ARGV!=2 && $#ARGV!=3;
 

$err_msg = "\tErrore nell'apertura del file\n";
$|=1;

$file = $ARGV[0];
$ofile = $ARGV[1];
$N = $ARGV[2];
if ($#ARGV==3) {
    $OS= $ARGV[3];
} else {
    $OS= 0;
}

open(FH, "<$file") || die $err_msg;
chop(@lines=<FH>);
close(FH);
if ($OS>$#lines) {
    printf STDERR "\n\tWarning: offset (%d) exceeds lines in input file (%d). No selection\n\n", $OS, $#lines;
    exit(0);
}
open(FH, ">$ofile") || die $err_msg;
$wN=0;
for ($i=$OS; $i<=$#lines; $i++) {
	printf FH ("%s\n", $lines[$i]); 
	$lines[$i]=~s/^\s+|\s+$//g;
	$wN += scalar(split(/\s+/, $lines[$i]));
	if ($wN>=$N) {
	    close(FH);
	    printf "%d\n", $i+1; 
	    exit(0);
	}
}
close(FH);
printf "%d\n", -1; 
exit(0);
