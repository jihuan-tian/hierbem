include(macro_hbem_std_compile_flags)
hbem_std_compile_flags()

set(_target hbem_laplace)
add_executable(${_target} hbem_laplace.cc cu_related.cu)
target_link_libraries(${_target} generic_tools fmt::fmt reflectcpp::reflectcpp
                      PkgConfig::tomlplusplus cpptrace::cpptrace)
deal_ii_setup_target(${_target})

add_custom_command(
  TARGET ${_target}
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/config.default.toml
    ${CMAKE_CURRENT_BINARY_DIR}/config.default.toml)
