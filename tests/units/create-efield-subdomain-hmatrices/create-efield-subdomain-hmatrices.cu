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
  const size_t stack_size = 1024 * 10;
  AssertCuda(cudaDeviceSetLimit(cudaLimitStackSize, stack_size));
  deallog << "CUDA stack size has been set to " << stack_size << std::endl;

  /**
   * @internal Get GPU device properties.
   */
  AssertCuda(
    cudaGetDeviceProperties(&HierBEM::CUDAWrappers::device_properties, 0));

  DDMEfield<2, 3> efield;
  efield.read_subdomain_topology(HBEM_TEST_MODEL_DIR
                                 "sphere-immersed-in-two-boxes.brep",
                                 HBEM_TEST_MODEL_DIR
                                 "sphere-immersed-in-two-boxes.msh");
  efield.read_skeleton_mesh(HBEM_TEST_MODEL_DIR
                            "sphere-immersed-in-two-boxes.msh");
  // At the moment, we manually assign problem parameters.
  efield.initialize_parameters();
  efield.setup_system();
  efield.output_results();
}
