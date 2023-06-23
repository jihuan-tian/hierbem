/**
 * \file laplace-bem-dirichlet-hmatrix.cc
 * \brief
 * \ingroup testers
 * \author Jihuan Tian
 * \date 2022-09-23
 */

#include <deal.II/base/logstream.h>

#include <cuda_runtime.h>

#include <iostream>

#include "cu_profile.hcu"
#include "debug_tools.hcu"
#include "laplace_bem.h"

using namespace dealii;
using namespace IdeoBEM;

int
main(int argc, char *argv[])
{
  /**
   * @internal Pop out the default "DEAL" prefix string.
   */
  deallog.pop();
  deallog.depth_console(5);
  LogStream::Prefix                prefix_string("HierBEM");
  IdeoBEM::CUDAWrappers::NVTXRange nvtx_range("HierBEM");

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
   * @internal Use 8-byte bank size in shared memory, since double value type is
   * used.
   */
  //  error_code = cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  //  AssertCuda(error_code);

  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  const bool                is_interior_problem = false;
  LaplaceBEM<dim, spacedim> bem(
    1, // fe order for dirichlet space
    0, // fe order for neumann space
    1, // mapping order for dirichlet domain
    1, // mapping order for neumann domain
    LaplaceBEM<dim, spacedim>::ProblemType::DirichletBCProblem,
    is_interior_problem, // is interior problem
    4,                   // n_min for cluster tree
    8,                   // n_min for block cluster tree
    0.8,                 // eta for H-matrix
    5,                   // max rank for H-matrix
    0.01,                // aca epsilon for H-matrix
    1.0,                 // eta for preconditioner
    2,                   // max rank for preconditioner
    0.1,                 // aca epsilon for preconditioner
    MultithreadInfo::n_cores());

  timer.stop();
  print_wall_time(deallog, timer, "program preparation");

  timer.start();

  if (argc > 1)
    {
      bem.read_volume_mesh(std::string(argv[1]));
    }
  else
    {
      bem.read_volume_mesh(std::string("sphere-from-gmsh_hex.msh"));
    }

  timer.stop();
  print_wall_time(deallog, timer, "read mesh");

  timer.start();

  // Assign constant Dirichlet boundary conditions.
  Functions::ConstantFunction<3, double> dirichlet_bc(10);

  bem.assign_dirichlet_bc(dirichlet_bc);

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
