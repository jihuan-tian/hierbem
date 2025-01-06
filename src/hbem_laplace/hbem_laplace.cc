/**
 * HierBEM Laplace solver workbench
 */

#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>

#include <deal.II/grid/grid_in.h>

#include <cpptrace/from_current.hpp>
#include <fmt/core.h>
#include <openblas-pthread/cblas.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <system_error>

#include "config_file/config_file.h"
#include "cu_related.h"
#include "debug_tools.h"
#include "laplace_bem.h"

using namespace dealii;
using namespace HierBEM;

class LaplaceWorkbench
{
public:
  using SolverType = LaplaceBEM<2, 3>;
  using TriaType = Triangulation<3>;
  using GridInType = GridIn<3>;

  static constexpr const char *LOG_PREFIX       = "HierBEM";
  static constexpr const char *RUNTIME_LOG_FILE = "hierbem_laplace.log";

  LaplaceWorkbench()
    : log_file_os_()
    , bem_(nullptr)
  {}

  void
  initWorkDir()
  {
    const auto &conf_inst  = ConfigFile::instance().getConfig();
    const auto &output_dir = conf_inst.project.output_dir;
    const auto &proj_name  = conf_inst.project.project_name.value();
    const auto &work_dir   = std::filesystem::path(output_dir) / proj_name;

    // Create working directory if it doesn't exist
    std::error_code ec;
    std::filesystem::create_directories(work_dir, ec);
    if (ec)
      {
        throw fmt::system_error(ec.value(),
                                "Failed to create working directory: {}",
                                work_dir.string());
      }

    // Change current working directory to the project directory
    std::filesystem::current_path(work_dir);
  }

  void
  initLogger()
  {
    // Write run-time logs to file
    log_file_os_ = std::make_shared<std::ofstream>(RUNTIME_LOG_FILE);

    deallog.pop();
    deallog.depth_console(0);
    deallog.depth_file(5);
    deallog.attach(*log_file_os_);
  }

  void
  initRuntime()
  {
    /**
     * @internal Set number of threads used for OpenBLAS.
     */
    openblas_set_num_threads(1);

    initCudaRuntime();
  }

  void
  setupHierBEMSolver()
  {
    const auto &conf_inst = ConfigFile::instance().getConfig();

    std::int32_t num_threads = conf_inst.misc.aca_thread_num;
    if (num_threads < 0)
      {
        num_threads = MultithreadInfo::n_threads();
      }

    // Initialize HierBEM Laplace solver
    bem_ =
      std::make_unique<SolverType>(conf_inst.bem.fe_order_for_dirichlet_space,
                                   conf_inst.bem.fe_order_for_neumann_space,
                                   problemTypeLiteralToEnum(
                                     conf_inst.bem.problem_type),
                                   conf_inst.bem.is_interior_problem,
                                   conf_inst.cluster_tree.n_min_for_ct,
                                   conf_inst.cluster_tree.n_min_for_bct,
                                   conf_inst.hmatrix.eta,
                                   conf_inst.hmatrix.max_rank,
                                   conf_inst.hmatrix.aca_relative_err,
                                   conf_inst.precond.eta,
                                   conf_inst.precond.max_rank,
                                   conf_inst.precond.aca_relative_err,
                                   num_threads);

    // Set project name
    bem_->set_project_name(conf_inst.project.project_name.value());
  }

  void
  setupMeshAndManifold()
  {
    // Read mesh file
    const auto &conf_inst = ConfigFile::instance().getConfig();
    std::ifstream mesh_file(conf_inst.project.mesh_file);

  }

  void
  setupBoundaryConds()
  {
    // TBD
  }

  void
  runSolverAndOutput()
  {
    // TBD
  }

  void
  startup()
  {
    initWorkDir(); // Make working directory hierarchies if it doesn't
                   // exist
    initLogger();  // Initialize deal.ii logger
    LogStream::Prefix prefix_string(LOG_PREFIX);

    initRuntime(); // Initialize OpenBLAS/CUDA runtimes

    Timer timer; // Create and start the timer

    setupHierBEMSolver(); // Prepare HierBEM Laplace solver
    timer.stop();
    print_wall_time(deallog, timer, "program preparation");

    timer.start();
    setupMeshAndManifold(); // Read in mesh and assign manifolds if any
    timer.stop();
    print_wall_time(deallog, timer, "read mesh");

    timer.start();
    setupBoundaryConds(); // Setup boundary conditions if any
    timer.stop();
    print_wall_time(deallog, timer, "assign boundary conditions");

    timer.start();
    runSolverAndOutput(); // Assemble and solve BEM system, output
                          // results
    timer.stop();
    print_wall_time(deallog, timer, "run the solver");

    // Final summary log
    deallog << "Program exits with a total wall time " << timer.wall_time()
            << "s" << std::endl;
    bem_->print_memory_consumption_table(deallog.get_file_stream());
  }

protected:
  SolverType::ProblemType
  problemTypeLiteralToEnum(const ProblemType &literal)
  {
    switch (literal.value())
      {
        case ProblemType::value_of<"neumann">():
          return SolverType::ProblemType::NeumannBCProblem;
        case ProblemType::value_of<"dirichlet">():
          return SolverType::ProblemType::DirichletBCProblem;
        case ProblemType::value_of<"mixed">():
          return SolverType::ProblemType::MixedBCProblem;
        default:
          throw std::runtime_error("Unknown problem type");
      }
  }

  std::shared_ptr<std::ofstream> log_file_os_;
  std::unique_ptr<SolverType>    bem_;
};



int
main(int argc, char **argv)
{
  CPPTRACE_TRY
  {
    if (argc != 2)
      {
        std::cerr << "Usage: " << argv[0] << " <config file>" << std::endl;
        return 1;
      }

    ConfigFile::instance().initialize(argv[1]); // Load configuration file

    LaplaceWorkbench workbench;
    workbench.startup();

    return 0;
  }
  CPPTRACE_CATCH(const std::exception &e)
  {
    std::cerr << "Exception: " << e.what() << std::endl;
    cpptrace::from_current_exception().print();
    return 1;
  }
}
