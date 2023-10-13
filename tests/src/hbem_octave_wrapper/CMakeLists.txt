FIND_PACKAGE(Octave REQUIRED)

SET(_target hbem_octave_wrapper)

# Octave wrapper solib
ADD_LIBRARY(${_target} SHARED ${_target}.cc)
TARGET_INCLUDE_DIRECTORIES(${_target} PUBLIC ${OCTAVE_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(${_target} PUBLIC ${OCTAVE_LIBRARIES})
