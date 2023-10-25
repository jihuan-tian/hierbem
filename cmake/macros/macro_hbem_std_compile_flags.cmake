#
# Set syntax standard to C++17 and enable more strict compilation for
# C++ and CUDA
#
# Usage:
#   HBEM_STD_COMPILE_FLAGS()
#
# After enabled strict compilation, targets can set global variable or their
# own property `HBEM_CHECK_RELAXED` to 1 to disable pedantic flags:
#
#   SET(HBEM_CHECK_RELAXED 1) # or
#   SET_TARGET_PROPERTIES(<target> PROPERTIES HBEM_CHECK_RELAXED 1)
#
MACRO(HBEM_STD_COMPILE_FLAGS)
    SET(CMAKE_CXX_STANDARD 17)
    SET(CMAKE_CXX_STANDARD_REQUIRED ON)
    SET(CMAKE_CUDA_STANDARD 17)
    SET(CMAKE_CUDA_STANDARD_REQUIRED ON)

    SET(_common_flags -Wall -Wextra)
    SET(_is_cxx "$<COMPILE_LANGUAGE:CXX>")
    SET(_is_cuda "$<COMPILE_LANGUAGE:CUDA>")
    SET(_is_gcc "$<STREQUAL:${CMAKE_CXX_COMPILER_ID},GNU>")
    SET(_is_global_relaxed "$<BOOL:HBEM_CHECK_RELAXED>")
    SET(_is_target_relaxed "$<BOOL:$<TARGET_PROPERTY:HBEM_CHECK_RELAXED>>")
    SET(_is_relaxed "$<OR:${_is_global_relaxed},${_is_target_relaxed}>")

    # Enable precompiled headers for GCC
    SET(_pch_if_gcc "$<IF:${_is_gcc},-fpch-preprocess,>")
    # Do not report pedantic errors if relaxed (such as trailing extra semicolon, etc.)
    SET(_pedantic_if_not_relaxed "$<IF:${_is_relaxed},,-pedantic-errors>")
    # ptxas in CUDA 11.4 has a bug that it will not honor stack size warning opts
    SET(_is_buggy_ptxas "$<VERSION_LESS:${CUDAToolkit_VERSION},11.8>")
    # Suppress all warnings for buggy ptxas instead of just stack size warning
    SET(_stack_size_opts "$<IF:${_is_buggy_ptxas},-Xptxas=-w,-Xptxas=-suppress-stack-size-warning;-Xnvlink=-suppress-stack-size-warning>")

    ADD_COMPILE_OPTIONS(
        "${_common_flags}"
        "$<${_is_cxx}:-Werror;${_pedantic_if_not_relaxed};${_pch_if_gcc}>"
        "$<${_is_cuda}:-Werror=all-warnings;${_stack_size_opts}>"
    )
ENDMACRO()
