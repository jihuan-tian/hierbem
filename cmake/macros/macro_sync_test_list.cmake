#
# Recursively scan all subdirectories containing CMakeLists.txt and append
# their realtive paths into list "${variable}"
#
# Usage:
#   SCAN_CMAKE_SUBDIRS(variable root_dir)
#
MACRO(SCAN_CMAKE_SUBDIRS _variable _root_dir)
    FILE(GLOB_RECURSE _cmake_file_paths ${_root_dir}/CMakeLists.txt)

    FOREACH(_cmake_file_path ${_cmake_file_paths})
        GET_FILENAME_COMPONENT(_case_dir ${_cmake_file_path} DIRECTORY)
        FILE(RELATIVE_PATH _rel_case_dir ${CMAKE_CURRENT_LIST_DIR} ${_case_dir})
        LIST(APPEND ${_variable} ${_rel_case_dir})
    ENDFOREACH()
ENDMACRO()

#
# Keep testcase list file content sync with testcases in the given directories
#
# Usage:
#   SYNC_TEST_LIST(test_list_file test_dir1 [test_dir2 ...])
#
MACRO(SYNC_TEST_LIST _test_list_file _test_dir1)
    SET(_test_suites ${_test_dir1})
    LIST(APPEND _test_suites ${ARGN})

    FOREACH(_test_suite ${_test_suites})
        MESSAGE(STATUS "Scanning test suites in ${_test_suite} ...")

        #
        # Traverse test-suite subdirectory recursively and add all nested
        # subdirectories containing CMakeLists.txt as test cases
        #
        SCAN_CMAKE_SUBDIRS(_cur_testcases ${_test_suite})
    ENDFOREACH()

    #
    # Update testcase listing file by the following steps:
    #   1. Read old test cases from the listing file, remove leading comment
    #   character and empty lines if necessary
    #   2. Calculate the difference list between old and current testcases
    #   3. If the difference list is not empty, append them into the listing
    #   file
    #
    FILE(STRINGS ${_test_list_file} _old_testcases)
    # Remove leading and tailing whitespaces
    LIST(TRANSFORM _old_testcases STRIP)
    # Remove leading comment characters (maybe whitespaced)
    LIST(TRANSFORM _old_testcases REPLACE "^[# \t\r\n]+" "")
    # Skip empty lines
    LIST(FILTER _old_testcases EXCLUDE REGEX "^$")
    # Remove surrounding ADD_SUBDIRECTORY(...) commands
    LIST(TRANSFORM _old_testcases REPLACE "^ADD_SUBDIRECTORY\\((.+)\\)$" "\\1")

    # XXX Debug only
    # MESSAGE(STATUS "+++ old: ${_old_testcases}")

    # Calculate difference of current and old testcase list
    LIST(APPEND _new_testcases ${_cur_testcases})
    IF(_old_testcases)
        LIST(REMOVE_ITEM _new_testcases ${_old_testcases})
    ENDIF()

    # XXX Debug only
    # MESSAGE(STATUS "+++ diff: ${_new_testcases}")

    # WARN deletion of testcase subdirectory will not be reflected in the listing file!

    # Append extra testcases if there are any differences
    IF(_new_testcases)
        FOREACH(_new_case ${_new_testcases})
            MESSAGE(STATUS "  Discovered new testcase ${_new_case}")

            #
            # Append newly found testcases into listing file
            #
            FILE(APPEND ${_test_list_file} "ADD_SUBDIRECTORY(${_new_case})\n")
        ENDFOREACH()
        LIST(LENGTH _new_testcases _num_new_testcases)
        MESSAGE(STATUS "Appended ${_num_new_testcases} newly found testcases into ${_test_list_file}")
    ENDIF()
ENDMACRO()
