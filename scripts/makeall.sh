#!/bin/bash

# @file makeall.sh
# @brief Make all targets matched by the regular expression pattern in parallel.
#
# This script should be run under the root binary directory.
# 
# @author Jihuan Tian
# @date 2024-03-18

usage_exit() {
    cat <<EOF
Usage: $0 [-h] [-j <THREAD_NUM>] [<TARGET_PATTERN>]
Make all targets matched by the regular expression pattern in parallel.

    -h	 	 	 Show this message.
    -j <THREAD_NUM> 	 Number of threads for the compilation. Default to 1.

The <TARGET_PATTERN> specifies the Perl-compatible regular expression used for
matching target names.
EOF
    exit 0
}

# Stop script when any error occurs
set -e

thread_num=1
list_targets=0

# Parse arguments
while getopts ':lj:' arg; do
    case "$arg" in
	j)
	    [ "$OPTARG" -gt 0 ] || usage_exit
	    thread_num="$OPTARG"
	    ;;
	l)
	    list_targets=1
	    ;;
	*)
	    usage_exit
	    ;;
    esac
done
shift $((OPTIND - 1))

target_pattern=".*"

if [ "$#" -gt 0 ]; then
    target_pattern="$1"
fi

if [ "$list_targets" = 1 ]; then
    make help | sed -E -e 's/^\.+\s*//' | grep -P "${target_pattern}"
else
    make -f CMakeFiles/Makefile2 -j${thread_num} `make help | sed -E -e 's/^\.+\s*//' | grep -P "${target_pattern}" | tr '\n' ' '`
fi

