#!/bin/bash

cur_dir=`pwd`
tester_name=`basename $cur_dir`
check_build_type=`grep debug CTestTestfile.cmake`

if [ -n "$check_build_type" ]; then
    build_type=debug
else
    build_type=release
fi

tester_program="./$tester_name.$build_type/$tester_name.$build_type"

for n in 10 100 1000 10000
do
    $tester_program -d $n
done
