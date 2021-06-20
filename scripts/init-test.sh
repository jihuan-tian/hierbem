#!/bin/bash

# Enter the test folder to execute this script.

test_target_name="$1"
test_folder_name=$(basename $(pwd))

echo -e "SET(TEST_TARGET $test_target_name)\nDEAL_II_PICKUP_TESTS()" > CMakeLists.txt
touch $test_target_name.cc
touch $test_target_name.output

# Append the test project to CMakeLists.txt.
echo "ADD_SUBDIRECTORY(tests/$test_folder_name)" >> ../../CMakeLists.txt

# Rerun cmake.
cd ../../
cmake -DCMAKE_BUILD_TYPE=Debug -DDEAL_II_DIR=/home/jihuan/Projects/deal.ii/program/dealii-9.1.1 .
cd "tests/$test_folder_name"
