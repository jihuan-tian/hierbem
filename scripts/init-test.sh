#!/bin/bash

# In the folder of the created test case, this script can be called as below
#
# $ ../../scripts/init-test.sh $(basename $(pwd)) build_type
#
# This will create (touch) the source code file and output file with their base
# names being the folder name of the test case. 
# build_type can be either Debug or Release.

# Enter the test folder to execute this script.

test_target_name="$1"
test_folder_name=$(basename $(pwd))

cmakelist_template=$(cat <<EOF
SET(TEST_TARGET $test_target_name)

# Link TEST_TARGET to my generic_tools library.
SET(TEST_LIBRARIES generic_tools boost_program_options)

DEAL_II_PICKUP_TESTS()

# Post-build event
ADD_CUSTOM_COMMAND(TARGET $test_target_name.debug POST_BUILD
  COMMAND mplayer \${CMAKE_SOURCE_DIR}/media/Oxygen-Im-Nudge.ogg > /dev/null 2>&1
  VERBATIM)
ADD_CUSTOM_COMMAND(TARGET $test_target_name.release POST_BUILD
  COMMAND mplayer \${CMAKE_SOURCE_DIR}/media/Oxygen-Im-Nudge.ogg > /dev/null 2>&1
  VERBATIM)
EOF
)

echo "$cmakelist_template" > CMakeLists.txt
touch $test_target_name.cc
touch $test_target_name.output

# Insert template code into the file $test_target_name.cc.
date_string=$(date -I)
template_code=$(cat <<EOF
/**
 * \file $test_target_name.cc
 * \brief 
 * \ingroup testers
 * \author Jihuan Tian
 * \date $date_string
 */
EOF
)

echo "$template_code" > $test_target_name.cc

# Append the test project to CMakeLists.txt.
echo "ADD_SUBDIRECTORY(tests/$test_folder_name)" >> ../../CMakeLists.txt

# Add the newly created files to Git.
git add $test_target_name.cc
git add $test_target_name.output
git add CMakeLists.txt

# Rerun cmake.
cd ../../
# Here we add the command line argument -Wno-dev to suppress the
# developer warnings given by CMake.
cmake -Wno-dev -DCMAKE_BUILD_TYPE=$2 -DDEAL_II_DIR=/home/jihuan/Projects/deal.ii/program/dealii-9.1.1 .
cd "tests/$test_folder_name"
