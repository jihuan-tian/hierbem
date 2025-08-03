#include <deal.II/base/logstream.h>

#include <deal.II/grid/manifold_lib.h>

#include <cuda_runtime.h>
#include <openblas-pthread/cblas.h>

#include <fstream>
#include <iostream>

#include "cu_profile.hcu"
#include "debug_tools.h"
#include "grid_in_ext.h"
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

  const bool                                is_interior_problem = false;
  LaplaceBEM<dim, spacedim, double, double> bem(
    1, // fe order for dirichlet space
    0, // fe order for neumann space
    LaplaceBEM<dim, spacedim, double, double>::ProblemType::DirichletBCProblem,
    is_interior_problem,         // is interior problem
    16,                          // n_min for cluster tree
    16,                          // n_min for block cluster tree
    0.8,                         // eta for H-matrix
    10,                          // max rank for H-matrix
    0.01,                        // aca epsilon for H-matrix
    1.0,                         // eta for preconditioner
    5,                           // max rank for preconditioner
    0.1,                         // aca epsilon for preconditioner
    MultithreadInfo::n_threads() // Number of threads used for ACA
  );
  bem.set_project_name("dirichlet-hmatrix-two-spheres");

  timer.stop();
  print_wall_time(deallog, timer, "program preparation");

  timer.start();

  std::ifstream mesh_in(HBEM_TEST_MODEL_DIR "two-spheres.msh");
  read_msh(mesh_in, bem.get_triangulation());
  bem.get_subdomain_topology().generate_topology(HBEM_TEST_MODEL_DIR
                                                 "two-spheres.brep",
                                                 HBEM_TEST_MODEL_DIR
                                                 "two-spheres.msh");

  // Generate two sphere manifolds.
  double                   inter_distance = 8.0;
  Manifold<dim, spacedim> *left_sphere_manifold =
    new SphericalManifold<dim, spacedim>(
      Point<spacedim>(-inter_distance / 2.0, 0, 0));
  Manifold<dim, spacedim> *right_sphere_manifold =
    new SphericalManifold<dim, spacedim>(
      Point<spacedim>(inter_distance / 2.0, 0, 0));
  bem.get_manifolds()[0] = left_sphere_manifold;
  bem.get_manifolds()[1] = right_sphere_manifold;

  // Create the map from manifold id to mapping order.
  bem.get_manifold_id_to_mapping_order()[0] = 1;
  bem.get_manifold_id_to_mapping_order()[1] = 1;

  // Assign manifolds to surface entities.
  bem.get_manifold_description()[1] = 0;
  bem.get_manifold_description()[2] = 1;

  timer.stop();
  print_wall_time(deallog, timer, "read mesh");

  timer.start();

  // Assign constant Dirichlet boundary conditions.
  DirichletBC dirichlet_bc;
  bem.assign_dirichlet_bc(dirichlet_bc);

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
