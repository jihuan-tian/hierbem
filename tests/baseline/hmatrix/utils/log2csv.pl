# Copyright (C) 2023 Xiaozhe Wang <chaoslawful@gmail.com>
#
# This file is part of the HierBEM library.
#
# HierBEM is free software: you can use it, redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your option)
# any later version. The full text of the license can be found in the file
# LICENSE at the top level directory of HierBEM.

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
