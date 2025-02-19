#include <deal.II/base/logstream.h>

#include <deal.II/grid/manifold_lib.h>

#include <cuda_runtime.h>
#include <openblas-pthread/cblas.h>

#include <fstream>
#include <iostream>

#include "debug_tools.h"
#include "grid_in_ext.h"
#include "hbem_test_config.h"
#include "laplace_bem.h"

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

void
run_mixed_hmatrix_op_precond()
{
  /**
   * @internal Pop out the default "DEAL" prefix string.
   */
  // Write run-time logs to file
  std::ofstream ofs("mixed-hmatrix-op-precond.log");
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
    LaplaceBEM<dim, spacedim>::ProblemType::MixedBCProblem,
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
  bem.set_project_name("mixed-hmatrix-op-precond");
  bem.set_preconditioner_type(
    LaplaceBEM<dim, spacedim>::PreconditionerType::OperatorPreconditioning);

  timer.stop();
  print_wall_time(deallog, timer, "program preparation");

  timer.start();

  std::ifstream mesh_in(HBEM_TEST_MODEL_DIR "bar.msh");
  read_msh(mesh_in, bem.get_triangulation());
  bem.get_subdomain_topology().generate_topology(HBEM_TEST_MODEL_DIR "bar.brep",
                                                 HBEM_TEST_MODEL_DIR "bar.msh");

  // Generate flat manifold.
  FlatManifold<dim, spacedim> *flat_manifold =
    new FlatManifold<dim, spacedim>();
  bem.get_manifolds()[0] = flat_manifold;

  // Create the map from material ids to manifold ids.
  for (types::material_id i = 1; i <= 6; i++)
    bem.get_manifold_description()[i] = 0;

  // Create the map from manifold id to mapping order.
  bem.get_manifold_id_to_mapping_order()[0] = 1;

  timer.stop();
  print_wall_time(deallog, timer, "read mesh");

  timer.start();

  DirichletBC dirichlet_bc;
  NeumannBC   neumann_bc;

  bem.assign_dirichlet_bc(dirichlet_bc, {5, 6});
  bem.assign_neumann_bc(neumann_bc, {1, 2, 3, 4});

  timer.stop();
  print_wall_time(deallog, timer, "assign boundary conditions");

  if (bem.validate_subdomain_topology())
    {
      timer.start();

      bem.run();

      timer.stop();
      print_wall_time(deallog, timer, "run the solver");

      deallog << "Program exits with a total wall time " << timer.wall_time()
              << "s" << std::endl;

      bem.print_memory_consumption_table(deallog.get_file_stream());
    }
  else
    {
      deallog << "Invalid subdomains!" << std::endl;
    }

  ofs.close();
}
