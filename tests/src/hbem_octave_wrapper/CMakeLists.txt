find_package(Octave REQUIRED)

set(_target hbem_octave_wrapper)

# Octave wrapper solib
add_library(${_target} SHARED ${_target}.cc)

# XXX Relax compiler's pedantic checks to include buggy Octave headers
set_target_properties(${_target} PROPERTIES HBEM_CHECK_RELAXED 1)

target_include_directories(${_target} PUBLIC ${OCTAVE_INCLUDE_DIRS})
target_link_libraries(${_target} PUBLIC ${OCTAVE_LIBRARIES})
