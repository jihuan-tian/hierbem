#
# Set syntax standard to C++20 and enable more strict compilation for C++ and
# CUDA
#
# Usage: HBEM_STD_COMPILE_FLAGS()
#
# After enabled strict compilation, targets can set global variable or their own
# property `HBEM_CHECK_RELAXED` to 1 to disable pedantic flags:
#
# SET(HBEM_CHECK_RELAXED 1) # or SET_TARGET_PROPERTIES(<target> PROPERTIES
# HBEM_CHECK_RELAXED 1)
#
macro(HBEM_STD_COMPILE_FLAGS)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)

  set(_common_flags -Wall -Wextra)
  set(_is_cxx "$<COMPILE_LANGUAGE:CXX>")
  set(_is_cuda "$<COMPILE_LANGUAGE:CUDA>")
  set(_is_gcc "$<STREQUAL:${CMAKE_CXX_COMPILER_ID},GNU>")
  set(_is_global_relaxed "$<BOOL:HBEM_CHECK_RELAXED>")
  set(_is_target_relaxed "$<BOOL:$<TARGET_PROPERTY:HBEM_CHECK_RELAXED>>")
  set(_is_relaxed "$<OR:${_is_global_relaxed},${_is_target_relaxed}>")

  # Enable precompiled headers for GCC
  set(_pch_if_gcc "$<IF:${_is_gcc},-fpch-preprocess,>")
  # Do not report pedantic errors if relaxed (such as trailing extra semicolon,
  # etc.)
  set(_pedantic_if_not_relaxed "$<IF:${_is_relaxed},,-pedantic-errors>")
  # Do not report deprecated warnings if relaxed, for example, implicit capture
  # of 'this' via [=] that is used in Octave.
  set(_disable_deprecated_warning "$<IF:${_is_relaxed},-Wno-deprecated,>")
  # ptxas in CUDA 11.4 has a bug that it will not honor stack size warning opts
  set(_is_buggy_ptxas "$<VERSION_LESS:${CUDAToolkit_VERSION},11.8>")
  # Suppress all warnings for buggy ptxas instead of just stack size warning
  set(_stack_size_opts
      "$<IF:${_is_buggy_ptxas},-Xptxas=-w,-Xptxas=-suppress-stack-size-warning;-Xnvlink=-suppress-stack-size-warning>"
  )

  add_compile_options(
    "${_common_flags}"
    "$<${_is_cxx}:-Werror;${_pedantic_if_not_relaxed};${_disable_deprecated_warning};${_pch_if_gcc}>"
    "$<${_is_cuda}:-Werror=all-warnings;${_stack_size_opts}>")

  #
  # Add bundled reflect-cpp library
  #
  include_directories("${CMAKE_SOURCE_DIR}/extern/reflect-cpp-0.9.0/include")

endmacro()
