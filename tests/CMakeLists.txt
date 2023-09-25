SET(_test_suites units functions regressions)

FOREACH(_test_suite ${_test_suites})
    MESSAGE(STATUS "Scanning test suite tests/${_test_suite} ...")

    #
    # Traverse test-suite subdirectory recursively and add all nested
    # subdirectories containing CMakeLists.txt as test cases
    #
    FILE(GLOB_RECURSE _cmake_file_paths ${_test_suite}/CMakeLists.txt )
    FOREACH(_cmake_file_path ${_cmake_file_paths})
        GET_FILENAME_COMPONENT(_case_dir ${_cmake_file_path} DIRECTORY)
        FILE(RELATIVE_PATH _rel_case_dir ${CMAKE_CURRENT_LIST_DIR} ${_case_dir})
        ADD_SUBDIRECTORY(${_rel_case_dir})
        MESSAGE(STATUS "  Discovered test case ${_rel_case_dir}")
    ENDFOREACH()

ENDFOREACH()

