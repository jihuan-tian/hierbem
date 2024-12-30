find_package(fmt CONFIG REQUIRED)

add_library(sbb SHARED sbb.cc cu_sbb.cu)
target_link_libraries(sbb PRIVATE fmt::fmt)

set(_target test-template-pimpl)
add_executable(${_target} template-pimpl.cc)
target_link_libraries(${_target} PRIVATE sbb fmt::fmt Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(${_target})
