while ($in=<STDIN>) {
    $in=~s/^\s+|\s+$//g;
    @n=split(/\s+/, $in);
    $c=0; $t=0;
    while ($n=shift(@n)) {
	$t+=log($n);
	$c++;
    }
    printf "%e\n", exp($t/$c);
}
