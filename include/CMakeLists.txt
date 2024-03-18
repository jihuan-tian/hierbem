#
# Set default config values
#
if(NOT HBEM_ENABLE_DEBUG)
  set(HBEM_ENABLE_DEBUG 1)
endif()

if(NOT HBEM_MESSAGE_LEVEL)
  set(HBEM_MESSAGE_LEVEL 1)
endif()

if(NOT HBEM_ENABLE_TIMER)
  set(HBEM_ENABLE_TIMER 1)
endif()

if(NOT HBEM_ENABLE_MATRIX_EXPORT)
  set(HBEM_ENABLE_MATRIX_EXPORT 0)
endif()

if(NOT HBEM_ENABLE_NVTX)
  set(HBEM_ENABLE_NVTX 0)
endif()

# If ACA runs in multiple threads, setting this flag to 0 will not make it
# really deterministic. To fully remove the randomness in ACA, TBB thread number
# should be set to 1 by calling `dealii::MultithreadInfo::set_thread_limit(1)`.
if(NOT HBEM_RANDOM_ACA)
  set(HBEM_RANDOM_ACA 0)
endif()

# Do not use TBB task arena or isolate in H-LU or H-Cholesky by default.
if(NOT HBEM_ARENA_OR_ISOLATE_IN_LU_AND_CHOL)
  set(HBEM_ARENA_OR_ISOLATE_IN_LU_AND_CHOL 3)
endif()

if(NOT HBEM_ARENA_MAX_CONCURRENCY_IN_LU_AND_CHOL)
  set(HBEM_ARENA_MAX_CONCURRENCY_IN_LU_AND_CHOL 1)
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
               ${CMAKE_CURRENT_SOURCE_DIR}/config.h)
