/**
 * @file create-efield-subdomain-hmatrices.cu
 * @brief
 *
 * @ingroup testers
 * @author
 * @date 2024-08-09
 */
#include <catch2/catch_all.hpp>
#include <cuda_runtime.h>

#include <fstream>

#include "debug_tools.hcu"
#include "electric_field/ddm_efield.h"
#include "grid_in_ext.h"
#include "grid_out_ext.h"
#include "hbem_test_config.h"

using namespace dealii;
using namespace HierBEM;
using namespace Catch::Matchers;

namespace HierBEM
{
  namespace CUDAWrappers
  {
    extern cudaDeviceProp device_properties;
  }
} // namespace HierBEM

TEST_CASE("Create subdomain H-hmatrices", "[ddm_efield]")
{
  deallog.pop();
  deallog.depth_console(0);
  deallog.depth_file(5);
  deallog.attach(std::cout);

  const size_t stack_size = 1024 * 10;
  AssertCuda(cudaDeviceSetLimit(cudaLimitStackSize, stack_size));
  deallog << "CUDA stack size has been set to " << stack_size << std::endl;

  /**
   * @internal Get GPU device properties.
   */
  AssertCuda(
    cudaGetDeviceProperties(&HierBEM::CUDAWrappers::device_properties, 0));

  DDMEfield<2, 3> efield(
    1,                           // fe order for dirichlet space
    0,                           // fe order for neumann space
    32,                          // n_min for cluster tree
    32,                          // n_min for block cluster tree
    0.8,                         // eta for H-matrix
    5,                           // max rank for H-matrix
    0.01,                        // aca epsilon for H-matrix
    1.0,                         // eta for preconditioner
    2,                           // max rank for preconditioner
    0.1,                         // aca epsilon for preconditioner
    MultithreadInfo::n_threads() // Number of threads used for ACA
  );
  efield.read_subdomain_topology(HBEM_TEST_MODEL_DIR
                                 "sphere-immersed-in-two-boxes.brep",
                                 HBEM_TEST_MODEL_DIR
                                 "sphere-immersed-in-two-boxes.msh");
  read_msh(HBEM_TEST_MODEL_DIR "sphere-immersed-in-two-boxes.msh",
           efield.get_triangulation(),
           false);
  // At the moment, we manually assign problem parameters.
  efield.initialize_parameters();
  efield.setup_system();
  efield.assemble_system();
  efield.output_results();
}
