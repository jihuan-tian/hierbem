# Preparation of Development Environment

## Compiler and build tools

* GCC >= 12.2.0
* CUDA >= 12.4
* CMake >= 3.25.2
* GNU Make or Ninja

## OpenCASCADE

* Install VTK 9. On Debian Linux, install the packages `vtk9` and `libvtk9-dev`.

* Download [OpenCASCADE 7.8.0](https://github.com/Open-Cascade-SAS/OCCT/archive/refs/tags/V7_8_0.tar.gz) from GitHub and decompress it.

  ```bash
  tar zxvf opencascade-7.8.0.tar.gz
  ```

* Because the `runpath` values for OpenCASCADE libraries are empty by default, we need to add the following CMake configurations to the top level `CMakeLists.txt`. See [here](https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling#always-full-rpath) for more information.

  ```cmake
  # use, i.e. don't skip the full RPATH for the build tree
  set(CMAKE_SKIP_BUILD_RPATH FALSE)
  
  # when building, don't use the install RPATH already
  # (but later on when installing)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
  
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  
  # add the automatically determined parts of the RPATH
  # which point to directories outside the build tree to the install RPATH
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  
  # the RPATH to be used when installing, but only if it's not a system directory
  list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
  if("${isSystemDir}" STREQUAL "-1")
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  endif("${isSystemDir}" STREQUAL "-1")
  ```

* Configure, build and install OpenCASCADE:

  ```bash
  mkdir occt-build && cd occt-build
  cmake -DCMAKE_INSTALL_PREFIX=<opencascade_install_prefix> \
    -DUSE_VTK=ON \
    -D3RDPARTY_VTK_INCLUDE_DIR=/usr/include/vtk-9.1 \
    -DCMAKE_BUILD_TYPE=Release
    ../OCCT-7_8_0
  make -j <n> all
  make install
  ```

  N.B. Do not enable `USE_TBB`.

## Gmsh

* Clone the modified Gmsh 4.14.0 from [here](https://github.com/jihuan-tian/gmsh):

  ```bash
  git clone https://github.com/jihuan-tian/gmsh.git
  ```

* Configure, build and install Gmsh:

  ```bash
  mkdir gmsh-build && cd gmsh-build
  CASROOT=<opencascade_install_prefix> \
    cmake -DCMAKE_INSTALL_PREFIX=<gmsh_install_prefix> \
      -DENABLE_OCC=1 \
      -DENABLE_MPI=0 \
      -DENABLE_MED=0 \
      -DENABLE_PETSC=0 \
      -DENABLE_BUILD_DYNAMIC=1 \
      -DENABLE_RPATH=1 \
      -DCMAKE_BUILD_TYPE=Release
      ../gmsh
  make -j <n> all
  make install
  ```

  In the above, the variable `CASROOT` is used to specify the installation path of OpenCASCADE.

## deal.II

* Install the following dependencies from Linux official sources:

  * OpenBLAS: `libopenblas-pthread-dev`
  * LAPACK: `liblapack-dev`
  * HDF5: `libhdf5-openmpi-dev`
  * OpenMPI: `libopenmpi-dev`
  * MuParser: `libmuparser-dev`
  * TBB: `libtbb-dev`

* Clone the modified deal.II 9.4.1 from [here](https://github.com/jihuan-tian/dealii):

  ```bash
  git clone https://github.com/jihuan-tian/dealii.git
  ```

* Configure, build and install deal.II:

  ```bash
  mkdir dealii-build && cd dealii-build
  cmake -DCMAKE_INSTALL_PREFIX=<deal_ii_install_prefix> \
    -DDEAL_II_ALLOW_AUTODETECTION=OFF \
    -DDEAL_II_COMPILE_EXAMPLES=OFF \
    -DDEAL_II_WITH_MPI=ON \
    -DDEAL_II_WITH_CUDA=ON \
    -DDEAL_II_WITH_COMPLEX_VALUES=ON \
    -DDEAL_II_WITH_LAPACK=ON \
    -DDEAL_II_WITH_MUPARSER=ON \
    -DDEAL_II_WITH_HDF5=ON \
    -DHDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/openmpi \
    -DDEAL_II_WITH_TBB=ON \
    -DDEAL_II_WITH_OPENCASCADE=ON \
    -DOPENCASCADE_DIR=<opencascade_install_prefix> \
    -DDEAL_II_WITH_GMSH=ON \
    -DGMSH_DIR=<gmsh_install_prefix> \
    -DCMAKE_BUILD_TYPE=Release \
    ../dealii
  make -j <n> all
  make install
  ```

## reflect-cpp

* Clone reflect-cpp from [here](https://github.com/getml/reflect-cpp):

  ```bash
  git clone https://github.com/getml/reflect-cpp.git
  ```

* Configure, build and install reflect-cpp:

  ```bash
  mkdir reflect-cpp-build && cd reflect-cpp-build
  cmake -DCMAKE_INSTALL_PREFIX=<reflect_cpp_install_prefix> \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_BUILD_TYPE=Release \
    ../reflect-cpp
  make -j <n> all
  make install
  ```

## Software package manager

* Clone `vcpkg` into some directory (assume the directory is `<SRC>`) and initialize it:

  ```bash
  cd <SRC>
  git clone https://github.com/microsoft/vcpkg.git
  cd vcpkg
  ./bootstrap-vcpkg.sh
  ```

  Then add the following content into your `.bashrc` file:

  ```bash
  export VCPKG_ROOT="<SRC>/vcpkg"
  export PATH="$VCPKG_ROOT:$PATH"
  ```

## Mathematical software tools used by HierBEM test cases

At the moment, Julia and GNU Octave are adopted to verify computation results in HierBEM test cases. Due to the limited performance of Octave, we will gradually convert all Octave scripts to Julia.

* Install Julia

  * Install `jill` via `pip`: `pip install jill`

  * Run `jill upstream` to check available Julia release sources and select the fastest one, such as

    ```text
    - TUNA: Tsinghua University TUNA Association
      * mirrors.tuna.tsinghua.edu.cn (33 ms)
      * opentuna.cn (20000 ms)
    ```

  * Install a specific version of Julia, such as 1.10.4

    ```text
    jill install 1.10.4 --upstream <source-name>
    ```

    where `<source-name>` is the short form of the release source name, such as `TUNA` as above.

  * Set the Julia package server in `$HOME/.julia/config/startup.jl`:

    ```text
    ENV["JULIA_PKG_SERVER"] = "https://mirrors.tuna.tsinghua.edu.cn/julia/"
    ```

  * Install the following packages in Julia to run HierBEM test cases

    * `CSV` and `DataFrames` for reading data files
    * `Gmsh` for reading Gmsh MSH files
    * `GLMakie` and `Colors` for visualization
    * `Meshes` for visualizing mesh

* Install GNU Octave

  * Download and build the source code of [GNU Octave](https://octave.org/download#source "GNU Octave"). Or you can also directly install the compiled packages `octave` and `liboctave-dev` in Debian Linux for example.

  * Install the `matgeom` package in Octave to run tests:

    ```octave
    octave> pkg install -forge matgeom
    ```

# Build HierBEM

* Clone HierBEM repository:

  ```bash
  git clone https://github.com/jihuan-tian/hierbem.git
  ```

* Enter `hierbem` source directory, and following the steps below:

  ```bash
  mkdir build && cd build
  cmake -DCMAKE_BUILD_TYPE={Debug|Release} \
    -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake \
    -DDEAL_II_DIR={deal_ii_install_prefix} \
    -DREFLECT_CPP_DIR={reflect_cpp_install_prefix} \
    -DOCTAVE_CONFIG_EXECUTABLE=<octave_install_prefix>/bin/octave-config \
    -DJULIA_DIR={julia_package_dir} \
    ../hierbem
  make -j <n> all
  ```

  There are also some configuration flags that can be specified on the `cmake` command line:

    * `HBEM_ENABLE_DEBUG` - `0`/`1`, default to `1`. Enable printing out debugging messages.
    * `HBEM_MESSAGE_LEVEL` - `int`, default to `1`. The higher the value, the more verbose the debugging messages.
    * `HBEM_ENABLE_TIMER` - `0`/`1`, default to `1`. Record elapsed wall time for key stages.
    * `HBEM_ENABLE_MATRIX_EXPORT` - `0`/`1`, default to `0`. When it is 1, enable exporting $\mathcal{H}$-matrices as full matrices to data files for debugging.
    * `HBEM_ENABLE_PRECONDITIONER_MATRIX_EXPORT` - `0`/`1`, default to `0`. When it is 1, enable exporting matrices built during the construction of the operator preconditioner for debugging.
    * `HBEM_ENABLE_NVTX` - `0`/`1`, default to `0`. Enable NVTX profile of key stages during $\mathcal{H}$-matrix assembly.
    * `HBEM_RANDOM_ACA` - `0`/`1`, default to `0`. Enable random number generation in the ACA+ (adaptive cross approximation) algorithm, which is used for randomly selecting reference rows and columns.
    * `HBEM_ARENA_OR_ISOLATE_IN_LU_AND_CHOL` - `1`/`2`/`3`, default to `3`. Used in `update` tasks in H-LU and H-Cholesky for restricting work stealing:
      * 1: use TBB arena
      * 2: use TBB isolate
      * 3: disable TBB arena and isolate
    * `HBEM_ARENA_MAX_CONCURRENCY_IN_LU_AND_CHOL` - `int`, default to `1`. When `HBEM_ARENA_OR_ISOLATE_IN_LU_AND_CHOL=2`, this specifies the maximum concurrency used in the local TBB task arena defined in the local scope of an `update` task.
    * `HBEM_NEUMANN_SOLUTION_SPACE` - `1`/`2`, default to 2. Select the solution space for Laplace problem with the Neumann boundary condition.
      * 1: $H_{\ast}^{1/2}(\Gamma)$: it is the subspace of $H^{1/2}(\Gamma)$, which is orthogonal to the natural density. Therefore, the natural density needs to be computed.
      * 2: $H_{\ast \ast}^{1/2}(\Gamma)$: it is the subspace of $H^{1/2}(\Gamma)$, which is orthogonal to the constant function $1$ on $\Gamma$.

  The default values of these configuration flags can be modified in `include/CMakeLists.txt`.

* After HierBEM is built, the library `libhierbem.so` will be generated in the folder `CMAKE_BINARY_DIR/src` and all test case executables will be generated in their corresponding folders.