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
  LaplaceWorkbench()
    : log_file_os_()
    , bem_(nullptr)
  {}

  void
  init_workdir()
  {
    const auto &conf_inst  = ConfigFile::instance().get_config();
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
  init_logger()
  {
    // Write run-time logs to file
    log_file_os_ = std::make_shared<std::ofstream>("hierbem_laplace.log");

    deallog.pop();
    deallog.depth_console(0);
    deallog.depth_file(5);
    deallog.attach(*log_file_os_);
  }

  void
  init_runtime()
  {
    /**
     * @internal Set number of threads used for OpenBLAS.
     */
    openblas_set_num_threads(1);

    init_cuda_runtime();
  }

  void
  setup_hierbem_solver()
  {
    // TBD
  }

  void
  setup_mesh_and_manifold()
  {
    // TBD
  }

  void
  setup_boundary_conditions()
  {
    // TBD
  }

  void
  run_solver_and_output()
  {
    // TBD
  }

  void
  startup()
  {
    init_workdir(); // Make working directory hierarchies if it doesn't
                    // exist
    init_logger();  // Initialize deal.ii logger
    LogStream::Prefix prefix_string("HierBEM");

    init_runtime(); // Initialize OpenBLAS/CUDA runtimes

    Timer timer; // Create and start the timer

    setup_hierbem_solver(); // Prepare HierBEM Laplace solver
    timer.stop();
    print_wall_time(deallog, timer, "program preparation");

    timer.start();
    setup_mesh_and_manifold(); // Read in mesh and assign manifolds if any
    timer.stop();
    print_wall_time(deallog, timer, "read mesh");

    timer.start();
    setup_boundary_conditions(); // Setup boundary conditions if any
    timer.stop();
    print_wall_time(deallog, timer, "assign boundary conditions");

    timer.start();
    run_solver_and_output(); // Assemble and solve BEM system, output
                             // results
    timer.stop();
    print_wall_time(deallog, timer, "run the solver");

    // Final summary log
    deallog << "Program exits with a total wall time " << timer.wall_time()
            << "s" << std::endl;
    bem_->print_memory_consumption_table(deallog.get_file_stream());
  }

protected:
  std::shared_ptr<std::ofstream>    log_file_os_;
  std::unique_ptr<LaplaceBEM<2, 3>> bem_;
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
