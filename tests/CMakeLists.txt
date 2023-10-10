INCLUDE(macro_load_test_deps)
INCLUDE(macro_sync_test_list)

SET(_test_list_file ${CMAKE_CURRENT_LIST_DIR}/TestCases.cmake)

# Update testcase list
SYNC_TEST_LIST(
    ${_test_list_file}
    units/
    functions/
    regressions/
    demos/
)

# Load testing dependencies
LOAD_TEST_DEPS()

# Generate testing config header files
SET(HBEM_ROOT_DIR ${CMAKE_SOURCE_DIR}/) # HierBEM source root
SET(HBEM_TEST_MODEL_DIR ${CMAKE_SOURCE_DIR}/tests/models/) # Testing models directory
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/include/hbem_test_config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/include/hbem_test_config.h)

# Add testing config header files to include path
INCLUDE_DIRECTORIES(APPEND ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add all testcase subdirectories
INCLUDE(${_test_list_file})
