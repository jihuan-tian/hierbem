find_package(fmt CONFIG REQUIRED)

set(_target test-fmt-output)
add_executable(${_target} fmt-output.cc)
target_link_libraries(${_target} PRIVATE fmt::fmt Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(${_target})
