/**
 * \file laplace-bem-dirichlet-hmatrix.cc
 * \brief
 * \ingroup testers
 * \author Jihuan Tian
 * \date 2022-09-23
 */

#include <deal.II/base/logstream.h>

#include <boost/program_options.hpp>

#include <cuda_runtime.h>

#include <fstream>
#include <iostream>

#include "cu_profile.hcu"
#include "cu_debug_tools.hcu"
#include "grid_in_ext.h"
#include "hbem_test_config.h"
#include "laplace_bem.hcu"

using namespace dealii;
using namespace HierBEM;
namespace po = boost::program_options;

struct CmdOpts
{
  unsigned int dirichlet_space_fe_order;
  unsigned int neumann_space_fe_order;
  unsigned int mapping_order;
};

CmdOpts
parse_cmdline(int argc, char *argv[])
{
  CmdOpts                 opts;
  po::options_description desc("Allowed options");

  // clang-format off
  desc.add_options()
    ("help,h", "show help message")
    ("dirichlet-order,d", po::value<unsigned int>()->default_value(1), "Finite element space order for the Dirichlet data")
    ("neumann-order,n", po::value<unsigned int>()->default_value(0), "Finite element space order for the Neumann data")
    ("mapping-order,m", po::value<unsigned int>()->default_value(1), "Mapping order for the sphere");
  // clang-format on

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help"))
    {
      std::cout << desc << std::endl;
      std::exit(EXIT_SUCCESS);
    }

  opts.dirichlet_space_fe_order = vm["dirichlet-order"].as<unsigned int>();
  opts.neumann_space_fe_order   = vm["neumann-order"].as<unsigned int>();
  opts.mapping_order            = vm["mapping-order"].as<unsigned int>();

  return opts;
}

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

int
main(int argc, char *argv[])
{
  CmdOpts opts = parse_cmdline(argc, argv);

  /**
   * @internal Pop out the default "DEAL" prefix string.
   */
  // Write run-time logs to file
  std::ofstream ofs("laplace-bem-dirichlet-hmatrix-two-spheres.log");
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
    opts.dirichlet_space_fe_order, // fe order for dirichlet space
    opts.neumann_space_fe_order,   // fe order for neumann space
    LaplaceBEM<dim, spacedim>::ProblemType::DirichletBCProblem,
    is_interior_problem,         // is interior problem
    64,                          // n_min for cluster tree
    64,                          // n_min for block cluster tree
    0.8,                         // eta for H-matrix
    5,                           // max rank for H-matrix
    0.01,                        // aca epsilon for H-matrix
    1.0,                         // eta for preconditioner
    2,                           // max rank for preconditioner
    0.1,                         // aca epsilon for preconditioner
    MultithreadInfo::n_threads() // Number of threads used for ACA
  );
  bem.set_project_name("laplace-bem-dirichlet-hmatrix-two-spheres");

  timer.stop();
  print_wall_time(deallog, timer, "program preparation");

  timer.start();

  read_skeleton_mesh(HBEM_TEST_MODEL_DIR "two-spheres-fine.msh",
                     bem.get_triangulation());
  bem.get_subdomain_topology().generate_topology(HBEM_TEST_MODEL_DIR
                                                 "two-spheres.brep",
                                                 HBEM_TEST_MODEL_DIR
                                                 "two-spheres-fine.msh");

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
  bem.get_manifold_id_to_mapping_order()[0] = opts.mapping_order;
  bem.get_manifold_id_to_mapping_order()[1] = opts.mapping_order;

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

  timer.start();

  bem.run();

  timer.stop();
  print_wall_time(deallog, timer, "run the solver");

  deallog << "Program exits with a total wall time " << timer.wall_time() << "s"
          << std::endl;

  bem.print_memory_consumption_table(deallog.get_file_stream());

  return 0;
}
