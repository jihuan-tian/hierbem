find_package(fmt CONFIG REQUIRED)

set(_target hierbem)
add_executable(${_target} hierbem.cc)
target_link_libraries(${_target} generic_tools fmt::fmt reflectcpp::reflectcpp
                      PkgConfig::tomlplusplus)

add_custom_command(
  TARGET ${_target}
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/config.default.toml
    ${CMAKE_CURRENT_BINARY_DIR}/config.default.toml)
