#!/bin/bash

# N.B. This script should be called within the directory of the test to be initialized.
# The first argument is the build_type.
../../scripts/init-test.sh $(basename $(pwd)) $1
