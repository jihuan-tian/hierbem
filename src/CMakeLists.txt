# Set the name of the project and target:
SET(TARGET "hierbem")

# Declare all source files the target consists of. Here, this is only
# the one step-X.cc file, but as you expand your project you may wish
# to add other source files as well. If your project becomes much larger,
# you may want to either replace the following statement by something like
#    FILE(GLOB_RECURSE TARGET_SRC  "source/*.cc")
#    FILE(GLOB_RECURSE TARGET_INC  "include/*.h")
#    SET(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC}) 
# or switch altogether to the large project CMakeLists.txt file discussed
# in the "CMake in user projects" page accessible from the "User info"
# page of the documentation.
SET(TARGET_SRC
  ${TARGET}.cc
  )

# This macro will further call the following macros:

# DEAL_II_SETUP_TARGET: it appends necessary include directories,
# linker flags, compile flags and compile definitions and the deal.II
# library link interface to the given target, which is stored in the
# variable TARGET, TARGET_SRC.
DEAL_II_INVOKE_AUTOPILOT()

# Add additional paths to link libraries
LINK_DIRECTORIES(${CMAKE_BINARY_DIR}/src /usr/lib/x86_64-linux-gnu)

# ##################################
# Library target: generic_tools
# ##################################
ADD_LIBRARY(generic_tools aca_plus.cc generic_functors.cc sauter_quadrature.cc regex_tools.cc hmatrix_support.cc)
# Because my library generic_tools has included deal.ii header files,
# dependent include paths and library paths need to be imported by
# calling this macro DEAL_II_SETUP_TARGET.
DEAL_II_SETUP_TARGET(generic_tools)

# Link TARGET to my generic_tools library.
TARGET_LINK_LIBRARIES(${TARGET} boost_program_options generic_tools)

# ###############################################
# Executable target: generate_block_cluster_tree
# ###############################################
ADD_EXECUTABLE(generate_block_cluster_tree generate_block_cluster_tree.cc)
DEAL_II_SETUP_TARGET(generate_block_cluster_tree)
TARGET_LINK_LIBRARIES(generate_block_cluster_tree boost_program_options generic_tools)
