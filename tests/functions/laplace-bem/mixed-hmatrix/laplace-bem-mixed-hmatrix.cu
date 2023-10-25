/**
 * \file laplace-bem-mixed-hmatrix.cc
 * \brief Verify solve Laplace mixed boundary value problem using \hmat.
 * \ingroup testers
 * \author Jihuan Tian
 * \date 2022-11-21
 */

#include <deal.II/base/logstream.h>

#include <cuda_runtime.h>

#include <iostream>

#include "debug_tools.hcu"
#include "laplace_bem.h"
#include "hbem_test_config.h"

using namespace dealii;
using namespace HierBEM;

/**
 * Function object for the Dirichlet boundary condition data.
 *
 * On the surface at @p z=0, apply constant potential 1. On the surface at
 * @p z=6, apply constant potential 0.
 */
class DirichletBC : public Function<3>
{
public:
  double
  value(const Point<3> &p, const unsigned int component = 0) const
  {
    (void)component;

    if (p(2) < 3)
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
 *
 * For surfaces other than those at @p z=0 and @p z=6, apply homogeneous
 * Neumann boundary condition.
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
  deallog.pop();
  deallog.depth_console(5);
  LogStream::Prefix prefix_string("HierBEM");

  /**
   * @internal Create and start the timer.
   */
  Timer timer;

  /**
   * @internal Initialize the CUDA device parameters.
   */
  //  AssertCuda(cudaSetDevice(0));
  //  AssertCuda(cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceScheduleBlockingSync));

  const size_t stack_size = 1024 * 10;
  AssertCuda(cudaDeviceSetLimit(cudaLimitStackSize, stack_size));
  deallog << "CUDA stack size has been set to " << stack_size << std::endl;

  /**
   * @internal Get GPU device properties.
   */
  AssertCuda(
    cudaGetDeviceProperties(&HierBEM::CUDAWrappers::device_properties, 0));

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
    4,                   // n_min for cluster tree
    32,                  // n_min for block cluster tree
    0.8,                 // eta for H-matrix
    5,                   // max rank for H-matrix
    0.01,                // aca epsilon for H-matrix
    1.0,                 // eta for preconditioner
    2,                   // max rank for preconditioner
    0.1,                 // aca epsilon for preconditioner
    MultithreadInfo::n_cores());

  bem.set_dirichlet_boundary_ids({1, 2});
  bem.set_neumann_boundary_ids({3, 4, 5, 6});

  timer.stop();
  print_wall_time(deallog, timer, "program preparation");

  if (argc > 1)
    {
      bem.read_volume_mesh(argv[1]);
    }
  else
    {
      bem.read_volume_mesh(HBEM_TEST_MODEL_DIR "bar.msh");
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
