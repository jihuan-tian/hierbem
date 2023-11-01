include(macro_load_test_deps)
include(macro_sync_test_list)
include(macro_hbem_std_compile_flags)

set(_test_list_file ${CMAKE_CURRENT_LIST_DIR}/TestCases.cmake)

# Update testcase list
sync_test_list(${_test_list_file} baseline/ units/ functions/ regressions/
               demos/)

# Load testing dependencies
load_test_deps()

# Generate testing config header files
set(HBEM_ROOT_DIR ${CMAKE_SOURCE_DIR}/) # HierBEM source root
set(HBEM_TEST_MODEL_DIR ${CMAKE_SOURCE_DIR}/tests/models/) # Testing models
                                                           # directory
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/hbem_test_config.h.in
               ${CMAKE_CURRENT_SOURCE_DIR}/include/hbem_test_config.h)

# Add testing config header files to include path
include_directories(APPEND ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Enable more strict compilation for C++ and CUDA
hbem_std_compile_flags()

add_subdirectory(src)

# Add all testcase subdirectories
include(${_test_list_file})

# Add custom target to cleanup deal.ii test output files
add_custom_target(
  clean_output
  COMMAND find -name output -delete
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  VERBATIM)
