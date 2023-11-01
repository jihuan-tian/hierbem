#
# Recursively scan all subdirectories containing CMakeLists.txt and append their
# realtive paths into list "${variable}"
#
# Usage: SCAN_CMAKE_SUBDIRS(variable root_dir)
#
macro(SCAN_CMAKE_SUBDIRS _variable _root_dir)
  file(GLOB_RECURSE _cmake_file_paths ${_root_dir}/CMakeLists.txt)

  foreach(_cmake_file_path ${_cmake_file_paths})
    get_filename_component(_case_dir ${_cmake_file_path} DIRECTORY)
    file(RELATIVE_PATH _rel_case_dir ${CMAKE_CURRENT_LIST_DIR} ${_case_dir})
    list(APPEND ${_variable} ${_rel_case_dir})
  endforeach()
endmacro()

#
# Keep testcase list file content sync with testcases in the given directories
#
# Usage: SYNC_TEST_LIST(test_list_file test_dir1 [test_dir2 ...])
#
macro(SYNC_TEST_LIST _test_list_file _test_dir1)
  set(_test_suites ${_test_dir1})
  list(APPEND _test_suites ${ARGN})

  foreach(_test_suite ${_test_suites})
    message(STATUS "Scanning test suites in ${_test_suite} ...")

    #
    # Traverse test-suite subdirectory recursively and add all nested
    # subdirectories containing CMakeLists.txt as test cases
    #
    scan_cmake_subdirs(_cur_testcases ${_test_suite})
  endforeach()

  #
  # Update testcase listing file by the following steps:
  #
  # 1. Read old test cases from the listing file, remove leading comment
  #    character and empty lines if necessary
  # 2. Calculate the difference list between old and current testcases
  # 3. If the difference list is not empty, append them into the listing file
  #
  file(STRINGS ${_test_list_file} _old_testcases)
  # Remove leading and tailing whitespaces
  list(TRANSFORM _old_testcases STRIP)
  # Remove leading comment characters (maybe whitespaced)
  list(TRANSFORM _old_testcases REPLACE "^[# \t\r\n]+" "")
  # Skip empty lines
  list(FILTER _old_testcases EXCLUDE REGEX "^$")
  # Remove surrounding ADD_SUBDIRECTORY(...) commands
  list(TRANSFORM _old_testcases REPLACE "^add_subdirectory\\((.+)\\)$" "\\1")

  # ~~~
  # XXX Debug only
  # MESSAGE(STATUS "+++ old: ${_old_testcases}")
  # ~~~

  # Calculate difference of current and old testcase list
  list(APPEND _new_testcases ${_cur_testcases})
  if(_old_testcases)
    list(REMOVE_ITEM _new_testcases ${_old_testcases})
  endif()

  # ~~~
  # XXX Debug only
  # MESSAGE(STATUS "+++ diff: ${_new_testcases}")
  # ~~~

  # WARN deletion of testcase subdirectory will not be reflected in the listing
  # file!

  # Append extra testcases if there are any differences
  if(_new_testcases)
    foreach(_new_case ${_new_testcases})
      message(STATUS "  Discovered new testcase ${_new_case}")

      #
      # Append newly found testcases into listing file
      #
      file(APPEND ${_test_list_file} "add_subdirectory(${_new_case})\n")
    endforeach()
    list(LENGTH _new_testcases _num_new_testcases)
    message(
      STATUS
        "Appended ${_num_new_testcases} newly found testcases into ${_test_list_file}"
    )
  endif()
endmacro()
