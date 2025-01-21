# HierBEM

## Prepare Development Environment

* Install `matgeom` package in octave to run tests:
  ```octave
  octave> pkg install -forge matgeom
  ```
* Install Julia and the following packages to run tests:
  * `CSV` and `DataFrames` for reading data files
  * `Gmsh` for reading Gmsh MSH files
  * `GLMakie` and `Colors` for visualization
  * `Meshes` for visualizing mesh
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
* Run the following command-line in project repository directory to configure and install dependencies:
  ```bash
  # <build_type> could be 'Debug' or 'Release'
  cmake \
    -Bbuild/<build_type> \
    -DCMAKE_BUILD_TYPE=<build_type> \
    -DDEAL_II_DIR=<deal_ii_install_prefix> \
    -DOCTAVE_CONFIG_EXECUTABLE=<octave_install_prefix>/bin/octave-config \
    -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake \
    -DREFLECT_CPP_DIR=<reflect_cpp_install_dir> \
    -DJULIA_DIR=<julia_package_dir>
  ```
  Or add the following items into Visual Studio Code's 
  **Cmake: Configure Args** and configure/build using Cmake extension:
  ```
  -DDEAL_II_DIR=<deal_ii_install_prefix>
  -DOCTAVE_CONFIG_EXECUTABLE=<octave_install_prefix>/bin/octave-config
  -DCMAKE_TOOLCHAIN_FILE=<SRC>/vcpkg/scripts/buildsystems/vcpkg.cmake
  -DREFLECT_CPP_DIR=<reflect_cpp_install_dir>
  -DJULIA_DIR=<julia_package_dir>
  ```
