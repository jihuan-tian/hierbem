# Convert H-matrix testcase log to CSV format
# Usage:
#   ./lu-factoriazation-task-parallel | perl log2csv.pl
#   ./lu-factoriazation-serial | perl log2csv.pl

use strict;
my @times = ();
while (<>) {
    chomp;
    if (/^=== Mesh refinement/) {
        print( join( ",", @times ), "\n" );
        @times = ();
    }
    else {
        if (/^Elapsed wall time for\D+?(\d+(\.\d+)?([eE][+-](\d+))?)s/) {
            push( @times, $1 );
        }
    }
}
print( join( ",", @times ), "\n" );
