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

class DirichletBC : public Function<3>
{
public:
  double
  value(const Point<3> &p, const unsigned int component = 0) const
  {
    (void)component;

    if (p(0) < 0)
      {
        return 10;
      }
    else
      {
        return -10;
      }
  }
};


namespace HierBEM
{
  namespace CUDAWrappers
  {
    extern cudaDeviceProp device_properties;
  }
} // namespace HierBEM

void
run_dirichlet_hmatrix_two_spheres()
{
  /**
   * @internal Pop out the default "DEAL" prefix string.
   */
  // Write run-time logs to file
  std::ofstream ofs("dirichlet-hmatrix-two-spheres.log");
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
  //  cudaError_t error_code = cudaSetDevice(0);
  //  error_code =
  //    cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceScheduleBlockingSync);
  //  AssertCuda(error_code);

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

  const bool                is_interior_problem = false;
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
  bem.set_project_name("dirichlet-hmatrix-two-spheres");

  timer.stop();
  print_wall_time(deallog, timer, "program preparation");

  timer.start();

  Triangulation<spacedim> left_ball, right_ball, tria;
  double                  inter_distance = 8;
  double                  radius         = 1.0;

  GridGenerator::hyper_ball(left_ball,
                            Point<spacedim>(-inter_distance / 2.0, 0, 0),
                            radius);
  GridGenerator::hyper_ball(right_ball,
                            Point<spacedim>(inter_distance / 2.0, 0, 0),
                            radius);

  /**
   * @internal Set different manifold ids and material ids to all the cells
   * in the two balls.
   */
  for (typename Triangulation<spacedim>::active_cell_iterator cell =
         left_ball.begin_active();
       cell != left_ball.end();
       cell++)
    {
      cell->set_all_manifold_ids(0);
      cell->set_material_id(0);
    }

  for (typename Triangulation<spacedim>::active_cell_iterator cell =
         right_ball.begin_active();
       cell != right_ball.end();
       cell++)
    {
      cell->set_all_manifold_ids(1);
      cell->set_material_id(1);
    }

  /**
   * @internal @p merge_triangulation can only operate on coarse mesh, i.e.
   * triangulations not refined. During the merging, the material ids are
   * copied. When the last argument is true, the manifold ids are copied.
   * Boundary ids will not be copied.
   */
  GridGenerator::merge_triangulations(left_ball, right_ball, tria, 1e-12, true);

  /**
   * @internal Assign manifold objects to the two balls in the merged mesh.
   */
  const SphericalManifold<spacedim> left_ball_manifold(
    Point<spacedim>(-inter_distance / 2.0, 0, 0));
  const SphericalManifold<spacedim> right_ball_manifold(
    Point<spacedim>(inter_distance / 2.0, 0, 0));

  tria.set_manifold(0, left_ball_manifold);
  tria.set_manifold(1, right_ball_manifold);

  // Refine the volume mesh.
  tria.refine_global(1);

  bem.assign_volume_triangulation(std::move(tria), true);

  // Extract the boundary mesh. N.B. Before the operation, the association
  // of manifold objects and manifold ids must also be set for the surface
  // triangulation. The manifold objects for the surface triangulation have
  // different dimension template paramreters as those for the volume
  // triangulation.
  Triangulation<dim, spacedim> surface_tria;

  const SphericalManifold<dim, spacedim> left_ball_surface_manifold(
    Point<spacedim>(-inter_distance / 2.0, 0, 0));
  const SphericalManifold<dim, spacedim> right_ball_surface_manifold(
    Point<spacedim>(inter_distance / 2.0, 0, 0));

  surface_tria.set_manifold(0, left_ball_surface_manifold);
  surface_tria.set_manifold(1, right_ball_surface_manifold);

  bem.assign_surface_triangulation(std::move(surface_tria), true);

  timer.stop();
  print_wall_time(deallog, timer, "read mesh");

  timer.start();

  // Assign constant Dirichlet boundary conditions.
  DirichletBC dirichlet_bc;
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
