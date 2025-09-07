/**
 * \file dirichlet-hmatrix.cc
 * \brief Verify solving the Laplace problem with Dirichlet boundary condition
 * using H-matrix based BEM.
 *
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2022-09-23
 */

#include <deal.II/base/logstream.h>

#include <deal.II/fe/mapping_manifold.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <boost/program_options.hpp>

#include <cuda_runtime.h>

#include <fstream>
#include <iostream>

#include "hbem_test_config.h"
#include "hmatrix/hmatrix.h"
#include "hmatrix/hmatrix_vmult_strategy.h"
#include "laplace/laplace_bem.h"
#include "preconditioners/preconditioner_type.h"
#include "utilities/cu_profile.hcu"
#include "utilities/debug_tools.h"

using namespace dealii;
using namespace HierBEM;
namespace po = boost::program_options;

struct CmdOpts
{
  unsigned int             dirichlet_space_fe_order;
  unsigned int             neumann_space_fe_order;
  unsigned int             mapping_order;
  unsigned int             refinement;
  PreconditionerType       precond_type;
  IterativeSolverVmultType vmult_type;
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
    ("mapping-order,m", po::value<unsigned int>()->default_value(1), "Mapping order for the sphere")
    ("refinement,r", po::value<unsigned int>()->default_value(5), "Number of global refinement when deal.ii is used to generate the mesh")
    ("precond-type,p", po::value<unsigned int>()->default_value(0), "Preconditioner for iterative solver: 0:H-Cholesky, 1:operator preconditioner, 2:identity")
    ("vmult-type,v", po::value<unsigned int>()->default_value(0), "H-matrix vmult type: 0:serial recursive, 1:serial iterative, 2:task parallel");
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
  opts.refinement               = vm["refinement"].as<unsigned int>();

  switch (vm["precond-type"].as<unsigned int>())
    {
        case 0: {
          opts.precond_type = PreconditionerType::HMatrixFactorization;
          break;
        }
        case 1: {
          opts.precond_type = PreconditionerType::OperatorPreconditioning;
          break;
        }
        case 2: {
          opts.precond_type = PreconditionerType::Identity;
          break;
        }
        default: {
          opts.precond_type = PreconditionerType::HMatrixFactorization;
          break;
        }
    }

  switch (vm["vmult-type"].as<unsigned int>())
    {
        case 0: {
          opts.vmult_type = IterativeSolverVmultType::SerialRecursive;
          break;
        }
        case 1: {
          opts.vmult_type = IterativeSolverVmultType::SerialIterative;
          break;
        }
        case 2: {
          opts.vmult_type = IterativeSolverVmultType::TaskParallel;
          break;
        }
        default: {
          opts.vmult_type = IterativeSolverVmultType::SerialRecursive;
          break;
        }
    }

  return opts;
}

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

int
main(int argc, char *argv[])
{
  CmdOpts opts = parse_cmdline(argc, argv);

  /**
   * @internal Pop out the default "DEAL" prefix string.
   */
  // Write run-time logs to file
  std::ofstream ofs("laplace-bem-dirichlet-hmatrix.log");
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
   * @internal Use 8-byte bank size in shared memory, since double value type
   * is used.
   */
  // AssertCuda(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));

  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  const bool                is_interior_problem = true;
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
  bem.set_project_name("laplace-bem-dirichlet-hmatrix");
  bem.set_preconditioner_type(opts.precond_type);
  bem.set_iterative_solver_vmult_type(opts.vmult_type);
  HMatrix<spacedim, double>::set_leaf_set_traversal_method(
    HMatrix<spacedim, double>::SpaceFillingCurveType::Hilbert);

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
  tria.refine_global(opts.refinement);

  Triangulation<dim, spacedim> surface_tria(
    Triangulation<dim,
                  spacedim>::MeshSmoothing::limit_level_difference_at_vertices);

  // Create the map from material ids to manifold ids. By default, the
  // material ids of all cells are zero, if the triangulation is created by a
  // deal.ii function in GridGenerator.
  bem.get_manifold_description()[0] = 0;

  // Create the map from manifold ids to manifold objects. Because in the
  // destructor of LaplaceBEM the manifold objects will be released, the
  // manifold object here is created on the heap.
  SphericalManifold<dim, spacedim> *ball_surface_manifold =
    new SphericalManifold<dim, spacedim>(center);
  bem.get_manifolds()[0] = ball_surface_manifold;

  // We should first assign manifold objects to the empty surface
  // triangulation, then perform surface mesh extraction.
  surface_tria.set_manifold(0, *ball_surface_manifold);
  bem.extract_surface_triangulation(tria, std::move(surface_tria), true);

  // Create the map from manifold id to mapping order.
  bem.get_manifold_id_to_mapping_order()[0] = opts.mapping_order;

  timer.stop();
  print_wall_time(deallog, timer, "read mesh");

  // Build surface-to-volume and volume-to-surface relationship.
  bem.get_subdomain_topology().generate_single_domain_topology_for_dealii_model(
    {0});

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

  return 0;
}
