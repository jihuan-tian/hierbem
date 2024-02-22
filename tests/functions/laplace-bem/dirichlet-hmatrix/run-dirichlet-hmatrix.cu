#include <deal.II/base/logstream.h>

#include <cuda_runtime.h>
#include <openblas-pthread/cblas.h>

#include <fstream>
#include <iostream>

#include "cu_profile.hcu"
#include "debug_tools.hcu"
#include "hbem_test_config.h"
#include "laplace_bem.h"

using namespace dealii;
using namespace HierBEM;

/**
 * Function object for the Dirichlet boundary condition data, which is
 * also the solution of the Neumann problem. The analytical expression is:
 * \f[
 * u=\frac{1}{4\pi\norm{x-x_0}}
 * \f]
 */
class DirichletBC : public Function<3>
{
public:
  // N.B. This function should be defined outside class NeumannBC or class
  // Example2, if no inline.
  DirichletBC()
    : Function<3>()
    , x0(0.25, 0.25, 0.25)
  {}

  DirichletBC(const Point<3> &x0)
    : Function<3>()
    , x0(x0)
  {}

  double
  value(const Point<3> &p, const unsigned int component = 0) const
  {
    (void)component;
    return 1.0 / 4.0 / numbers::PI / (p - x0).norm();
  }

private:
  /**
   * Location of the Dirac point source \f$\delta(x-x_0)\f$.
   */
  Point<3> x0;
};


namespace HierBEM
{
  namespace CUDAWrappers
  {
    extern cudaDeviceProp device_properties;
  }
} // namespace HierBEM

void
run_dirichlet_hmatrix()
{
  /**
   * @internal Pop out the default "DEAL" prefix string.
   */
  // Write run-time logs to file
  std::ofstream ofs("dirichlet-hmatrix.log");
  deallog.pop();
  deallog.depth_console(0);
  deallog.depth_file(5);
  deallog.attach(ofs);

  LogStream::Prefix prefix_string("HierBEM");
#if ENABLE_NVTX == 1
  HierBEM::CUDAWrappers::NVTXRange nvtx_range("HierBEM");
#endif

  /**
   * @internal Create and start the timer.
   */
  Timer timer;

  /**
   * @internal Set number of threads used for OpenBLAS.
   */
  openblas_set_num_threads(1);

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

  /**
   * @internal Use 8-byte bank size in shared memory, since double value type is
   * used.
   */
  // AssertCuda(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));

  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  const bool                is_interior_problem = true;
  LaplaceBEM<dim, spacedim> bem(
    1, // fe order for dirichlet space
    0, // fe order for neumann space
    1, // mapping order for dirichlet domain
    1, // mapping order for neumann domain
    LaplaceBEM<dim, spacedim>::ProblemType::DirichletBCProblem,
    is_interior_problem,         // is interior problem
    4,                           // n_min for cluster tree
    4,                           // n_min for block cluster tree
    0.8,                         // eta for H-matrix
    5,                           // max rank for H-matrix
    0.01,                        // aca epsilon for H-matrix
    1.0,                         // eta for preconditioner
    2,                           // max rank for preconditioner
    0.1,                         // aca epsilon for preconditioner
    MultithreadInfo::n_threads() // Number of threads used for ACA
  );
  bem.set_project_name("dirichlet-hmatrix");

  timer.stop();
  print_wall_time(deallog, timer, "program preparation");

  timer.start();

  /**
   * @internal Set the Dirac source location according to interior or exterior
   * problem.
   */
  Point<spacedim> source_loc;

  if (is_interior_problem)
    {
      source_loc = Point<spacedim>(1, 1, 1);
    }
  else
    {
      source_loc = Point<spacedim>(0.25, 0.25, 0.25);
    }

  const Point<spacedim> center(0, 0, 0);
  const double          radius(1);

  Triangulation<spacedim> tria;
  // The manifold_id is set to 0 on the boundary faces in @p hyper_ball.
  GridGenerator::hyper_ball(tria, center, radius);
  tria.refine_global(1);

  bem.assign_volume_triangulation(std::move(tria), true);

  Triangulation<dim, spacedim>           surface_tria;
  const SphericalManifold<dim, spacedim> ball_surface_manifold(center);
  surface_tria.set_manifold(0, ball_surface_manifold);

  bem.assign_surface_triangulation(std::move(surface_tria), true);

  timer.stop();
  print_wall_time(deallog, timer, "read mesh");

  timer.start();

  DirichletBC dirichlet_bc(source_loc);
  bem.assign_dirichlet_bc(dirichlet_bc);

  timer.stop();
  print_wall_time(deallog, timer, "assign boundary conditions");

  timer.start();

  bem.run();

  timer.stop();
  print_wall_time(deallog, timer, "run the solver");

  deallog << "Program exits with a total wall time " << timer.wall_time() << "s"
          << std::endl;

  bem.print_memory_consumption_table(deallog.get_file_stream());
}
