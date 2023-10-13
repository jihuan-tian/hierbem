FIND_PACKAGE(Octave REQUIRED)

SET(_target hbem_octave_wrapper)

# Octave wrapper solib
ADD_LIBRARY(${_target} SHARED ${_target}.cc)

# XXX Relax compiler's pedantic checks to include buggy Octave headers
SET_TARGET_PROPERTIES(${_target} PROPERTIES HBEM_CHECK_RELAXED 1)

TARGET_INCLUDE_DIRECTORIES(${_target} PUBLIC ${OCTAVE_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(${_target} PUBLIC ${OCTAVE_LIBRARIES})
