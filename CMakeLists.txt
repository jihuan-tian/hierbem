CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)
FIND_PACKAGE(deal.II 9.1.0 REQUIRED
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

# Project name
PROJECT("laplace-bem")
# Folder for header files
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include)
# Folder for source code
ADD_SUBDIRECTORY(src) 

# Enable unit testing for this deal.ii user project. 
ENABLE_TESTING()
# Add unit test project folders 
ADD_SUBDIRECTORY(tests/erichsen1996efficient-example2)
ADD_SUBDIRECTORY(tests/cell-neighboring-type)
ADD_SUBDIRECTORY(tests/cell-distance)
ADD_SUBDIRECTORY(tests/surface-jacobian-det)
ADD_SUBDIRECTORY(tests/dof-forward-and-backward-permutation)
ADD_SUBDIRECTORY(tests/gauss-legendre-quad-4d)
ADD_SUBDIRECTORY(tests/point-4d)
ADD_SUBDIRECTORY(tests/sauter-quad-same-panel)
ADD_SUBDIRECTORY(tests/sauter-quad-common-edge)
ADD_SUBDIRECTORY(tests/sauter-quad-common-vertex)
ADD_SUBDIRECTORY(tests/sauter-quad-regular)
ADD_SUBDIRECTORY(tests/erichsen1996efficient-example2-analytical-solution)
ADD_SUBDIRECTORY(tests/erichsen1996efficient-example2-neumann-bc)
ADD_SUBDIRECTORY(tests/normal-vector)
ADD_SUBDIRECTORY(tests/planarity-of-gmsh-quad-cell)
ADD_SUBDIRECTORY(tests/determinant4x4)
ADD_SUBDIRECTORY(tests/gmsh-and-dealii-node-ordering)
ADD_SUBDIRECTORY(tests/simple-bounding-box)
ADD_SUBDIRECTORY(tests/binary-tree)
ADD_SUBDIRECTORY(tests/cluster-tree)
ADD_SUBDIRECTORY(tests/quad-tree)
ADD_SUBDIRECTORY(tests/cluster-diameter)
ADD_SUBDIRECTORY(tests/block-cluster-tree)
ADD_SUBDIRECTORY(tests/lapack-matrix-delete-rows-and-columns)
ADD_SUBDIRECTORY(tests/lapack-matrix-svd)
