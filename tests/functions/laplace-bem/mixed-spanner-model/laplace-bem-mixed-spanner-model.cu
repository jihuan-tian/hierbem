/**
 * @file laplace-bem-mixed-spanner-model.cu
 * @brief
 *
 * @ingroup testers
 * @author Jihuan Tian
 * @date 2023-05-24
 */

#include <deal.II/base/logstream.h>

#include <cuda_runtime.h>

#include <fstream>
#include <iostream>

#include "debug_tools.hcu"
#include "laplace_bem.h"
#include "hbem_test_config.h"


using namespace dealii;
using namespace HierBEM;

/**
 * Function object for the Dirichlet boundary condition data.
 */
class DirichletBC : public Function<3>
{
public:
  double
  value(const Point<3> &p, const unsigned int component = 0) const
  {
    (void)component;

    if (p(0) < 0)
      {
        return 1;
      }
    else
      {
        return 0;
      }
  }
};

/**
 * Function object for the Neumann boundary condition data.
 */
class NeumannBC : public Function<3>
{
public:
  double
  value(const Point<3> &p, const unsigned int component = 0) const
  {
    (void)component;
    (void)p;

    return 0;
  }
};

namespace HierBEM
{
  namespace CUDAWrappers
  {
    extern cudaDeviceProp device_properties;
  }
} // namespace HierBEM

int
main(int argc, char *argv[])
{
  /**
   * @internal Pop out the default "DEAL" prefix string.
   */
  // Write run-time logs to file
  std::ofstream ofs("hierbem.log");
  deallog.pop();
  deallog.depth_console(0);
  deallog.depth_file(5);
  deallog.attach(ofs);

  LogStream::Prefix prefix_string("HierBEM");

  /**
   * @internal Create and start the timer.
   */
  Timer timer;

  /**
   * @internal Initialize the CUDA device parameters.
   */
  //  cudaError_t error_code = cudaSetDevice(0);
  //  error_code =
  //    cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceScheduleBlockingSync);
  //  AssertCuda(error_code);

  const size_t stack_size = 1024 * 10;
  cudaError_t  error_code = cudaDeviceSetLimit(cudaLimitStackSize, stack_size);
  AssertCuda(error_code);
  deallog << "CUDA stack size has been set to " << stack_size << std::endl;

  /**
   * @internal Get GPU device properties.
   */
  error_code =
    cudaGetDeviceProperties(&HierBEM::CUDAWrappers::device_properties, 0);
  AssertCuda(error_code);

  /**
   * @internal Use 8-byte bank size in shared memory, since double value type is
   * used.
   */
  //  error_code = cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  //  AssertCuda(error_code);

  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  const bool                is_interior_problem = true;
  LaplaceBEM<dim, spacedim> bem(
    1, // fe order for dirichlet space
    0, // fe order for neumann space
    1, // mapping order for dirichlet domain
    1, // mapping order for neumann domain
    LaplaceBEM<dim, spacedim>::ProblemType::MixedBCProblem,
    is_interior_problem, // is interior problem
    64,                  // n_min for cluster tree
    64,                  // n_min for block cluster tree
    0.8,                 // eta for H-matrix
    5,                   // max rank for H-matrix
    0.01,                // aca epsilon for H-matrix
    1.0,                 // eta for preconditioner
    1,                   // max rank for preconditioner
    0.1,                 // aca epsilon for preconditioner
    MultithreadInfo::n_cores());

  timer.stop();
  print_wall_time(deallog, timer, "program preparation");

  timer.start();

  bem.set_dirichlet_boundary_ids({1, 2});
  bem.set_neumann_boundary_ids({0});

  if (argc > 1)
    {
      bem.read_volume_mesh(argv[1]);
    }
  else
    {
      bem.read_volume_mesh(HBEM_TEST_MODEL_DIR "spanner.msh");
    }

  timer.stop();
  print_wall_time(deallog, timer, "read mesh");

  timer.start();

  DirichletBC dirichlet_bc;
  NeumannBC   neumann_bc;

  bem.assign_dirichlet_bc(dirichlet_bc);
  bem.assign_neumann_bc(neumann_bc);

  timer.stop();
  print_wall_time(deallog, timer, "assign boundary conditions");

  timer.start();

  bem.run();

  timer.stop();
  print_wall_time(deallog, timer, "run the solver");

  deallog << "Program exits with a total wall time " << timer.wall_time() << "s"
          << std::endl;

  return 0;
}
