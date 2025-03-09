set(_target hbem_julia_wrapper)

add_library(${_target} SHARED ${_target}.cc)
target_include_directories(${_target} PUBLIC ${JULIA_DIR}/include/julia)
target_link_directories(${_target} PUBLIC ${JULIA_DIR}/lib)
target_link_libraries(${_target} PUBLIC julia)
