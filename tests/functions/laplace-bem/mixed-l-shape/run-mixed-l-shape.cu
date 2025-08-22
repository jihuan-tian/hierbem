#include <deal.II/base/logstream.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>

#include <cuda_runtime.h>

#include <fstream>
#include <iostream>

#include "debug_tools.h"
#include "hbem_test_config.h"
#include "laplace_bem.h"

using namespace dealii;
using namespace HierBEM;

// Dirichlet boundary conditions on the left and top surface of the L-shape
class DirichletBC : public Function<3>
{
public:
  double
  value(const Point<3> &p, const unsigned int component = 0) const
  {
    (void)component;

    if (p(0) <= 1e-6)
      {
        // left surface
        return 0.0;
      }
    else
      {
        // top surface
        return 10.0;
      }
  }
};

// Neumann boundary conditions on the other surfaces of the L-shape
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

void
run_mixed_l_shape()
{
  // Write run-time logs to file
  std::ofstream ofs("mixed-l-shape.log");
  deallog.pop();
  deallog.depth_console(0);
  deallog.depth_file(5);
  deallog.attach(ofs);

  LogStream::Prefix prefix_string("HierBEM");

  /**
   * @internal Create and start the timer.
   */
  Timer timer;

  const size_t stack_size = 1024 * 10;
  AssertCuda(cudaDeviceSetLimit(cudaLimitStackSize, stack_size));
  deallog << "CUDA stack size has been set to " << stack_size << std::endl;

  /**
   * @internal Get GPU device properties.
   */
  AssertCuda(
    cudaGetDeviceProperties(&HierBEM::CUDAWrappers::device_properties, 0));

  const unsigned int dim                 = 2;
  const unsigned int spacedim            = 3;
  const bool         is_interior_problem = true;

  LaplaceBEM<dim, spacedim, double, double> bem(
    1, // fe order for dirichlet space
    0, // fe order for neumann space
    LaplaceBEM<dim, spacedim, double, double>::ProblemType::MixedBCProblem,
    is_interior_problem,         // is interior problem
    4,                           // n_min for cluster tree
    32,                          // n_min for block cluster tree
    0.8,                         // eta for H-matrix
    5,                           // max rank for H-matrix
    0.01,                        // aca epsilon for H-matrix
    1.0,                         // eta for preconditioner
    2,                           // max rank for preconditioner
    0.1,                         // aca epsilon for preconditioner
    MultithreadInfo::n_threads() // Number of threads used for ACA
  );
  bem.set_project_name("mixed-l-shape");

  timer.stop();
  print_wall_time(deallog, timer, "program preparation");

  timer.start();

  std::ifstream           mesh_file(HBEM_TEST_MODEL_DIR "l-shape.msh");
  Triangulation<spacedim> tria;
  GridIn<spacedim>        grid_in;
  grid_in.attach_triangulation(tria);
  grid_in.read_msh(mesh_file);

  // Create the map from material ids to manifold ids.
  bem.get_manifold_description()[1] = 0;
  bem.get_manifold_description()[2] = 0;
  for (types::material_id i = 19; i <= 24; i++)
    {
      bem.get_manifold_description()[i] = 0;
    }

  FlatManifold<dim, spacedim> *flat_manifold =
    new FlatManifold<dim, spacedim>();
  bem.get_manifolds()[0] = flat_manifold;

  Triangulation<dim, spacedim> surface_tria;
  surface_tria.set_manifold(0, *flat_manifold);
  bem.extract_surface_triangulation(tria, std::move(surface_tria), true);

  // Create the map from manifold id to mapping order.
  bem.get_manifold_id_to_mapping_order()[0] = 1;

  // Build surface-to-volume and volume-to-surface relationship.
  bem.get_subdomain_topology().generate_single_domain_topology_for_dealii_model(
    {1, 2, 19, 20, 21, 22, 23, 24});

  timer.stop();
  print_wall_time(deallog, timer, "read mesh");

  timer.start();

  DirichletBC dirichlet_bc;
  NeumannBC   neumann_bc;

  bem.assign_dirichlet_bc(dirichlet_bc, {1, 2});
  bem.assign_neumann_bc(neumann_bc, {19, 20, 21, 22, 23, 24});

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
