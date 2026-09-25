// Copyright (C) 2025 Xiaozhe Wang <chaoslawful@gmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * Pure BEM workbench for real valued and complex valued Laplace solvers
 */

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/types.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/opencascade/manifold_lib.h>

#include <BRep_Builder.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_Face.hxx>
#include <cpptrace/from_current.hpp>
#include <fmt/core.h>
#include <fmt/format.h>
#include <gmsh.h>
#include <rfl.hpp>

#include <complex>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <variant>
#include <vector>

#include "bem/types.h"
#include "cad_mesh/gmsh_manipulation.h"
#include "config_file/complex_function_parser.h"
#include "config_file/config_file.h"
#include "config_file/config_structs.h"
#include "config_file/cu_related.h"
#include "grid/grid_in_ext.h"
#include "laplace/laplace_bem.h"
#include "utilities/debug_tools.h"

using namespace dealii;
using namespace HierBEM;

class LaplaceWorkbench
{
public:
  using RealSolverType = LaplaceBEM<2, 3, double, double>;
  // In the complex valued case, boundary conditions take complex value, while
  // BEM kernels are still real valued.
  using ComplexSolverType = LaplaceBEM<2, 3, std::complex<double>, double>;
  using SolverPtrType     = std::variant<std::monostate,
                                     std::unique_ptr<RealSolverType>,
                                     std::unique_ptr<ComplexSolverType>>;
  using TriaType          = Triangulation<2, 3>;
  using GridInType        = GridIn<2, 3>;

  static constexpr const char *LOG_PREFIX       = "HierBEM";
  static constexpr const char *RUNTIME_LOG_FILE = "hierbem_laplace.log";

  LaplaceWorkbench()
    : log_file_os_()
    , bem_()
  {}

  void
  initWorkDir()
  {
    // If @p output_dir is a relative path, it is with respect to @p work_dir.
    // This also holds for the paths for mesh file and cad file.
    const auto       &conf_inst  = ConfigFile::instance().getConfig();
    const std::string output_dir = conf_inst.project.output_dir;
    const std::string proj_name  = conf_inst.project.project_name.value();
    const std::filesystem::path work_dir =
      std::filesystem::path(output_dir) / proj_name;

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
  initRuntime(const ConfParallelization &parallel_params)
  {
    initCudaRuntime(parallel_params);

    // Set TBB thread num.
    if (parallel_params.tbb_thread_num == -1)
      MultithreadInfo::set_thread_limit(MultithreadInfo::n_threads());
    else
      MultithreadInfo::set_thread_limit(parallel_params.tbb_thread_num);
  }

  void
  setupHierBEMSolver()
  {
    const auto &conf_inst = ConfigFile::instance().getConfig();

    // Initialize HierBEM Laplace solver
    if (is_solver_complex_valued())
      {
        bem_ = std::make_unique<ComplexSolverType>(
          ConfLaplaceBEM{conf_inst.bem.mesh_refinement,
                         conf_inst.bem.fe_order_for_dirichlet_space,
                         conf_inst.bem.fe_order_for_neumann_space,
                         problemTypeLiteralToEnum(conf_inst.bem.problem_type),
                         conf_inst.bem.is_interior_problem},
          conf_inst.hmatrix,
          conf_inst.hmatrix_precond,
          conf_inst.sauter_quad,
          conf_inst.sauter_quad_precond,
          conf_inst.linear_solver,
          conf_inst.op_precond,
          conf_inst.parallel);
      }
    else
      {
        bem_ = std::make_unique<RealSolverType>(
          ConfLaplaceBEM{conf_inst.bem.mesh_refinement,
                         conf_inst.bem.fe_order_for_dirichlet_space,
                         conf_inst.bem.fe_order_for_neumann_space,
                         problemTypeLiteralToEnum(conf_inst.bem.problem_type),
                         conf_inst.bem.is_interior_problem},
          conf_inst.hmatrix,
          conf_inst.hmatrix_precond,
          conf_inst.sauter_quad,
          conf_inst.sauter_quad_precond,
          conf_inst.linear_solver,
          conf_inst.op_precond,
          conf_inst.parallel);
      }

    std::visit(
      [&conf_inst, this](auto &solver_ptr) {
        using T = std::decay_t<decltype(solver_ptr)>;
        if constexpr (!std::is_same_v<T, std::monostate>)
          {
            solver_ptr->set_project_name(
              conf_inst.project.project_name.value());
            solver_ptr->set_preconditioner_type(
              this->preconditionerTypeLiteralToEnum(
                conf_inst.bem.precond_type));
            solver_ptr->set_iterative_solver_vmult_type(
              this->vmultTypeLiteralToEnum(conf_inst.bem.vmult_type));
          }
        else
          throw std::runtime_error("LaplaceBEM solver is not created yet");
      },
      bem_);
  }

  void
  setupMeshAndManifold()
  {
    // Load mesh file
    const auto   &conf_inst = ConfigFile::instance().getConfig();
    std::ifstream mesh_file(conf_inst.project.mesh_file);
    auto         &tria = std::visit(
      [](auto &solver_ptr) -> TriaType         &{
        using T = std::decay_t<decltype(solver_ptr)>;
        if constexpr (!std::is_same_v<T, std::monostate>)
          return solver_ptr->get_triangulation();
        else
          throw std::runtime_error("LaplaceBEM solver is not created yet");
      },
      bem_);
    read_msh(mesh_file, tria);

    if (conf_inst.project.cad_file.empty())
      throw std::runtime_error(
        "'project.cad_file' should not be empty. It is needed for acquiring surface normal vector direction and constructing surface-to-subdomain incidence.");

    // Initialize Gmsh then read CAD and mesh files.
    gmsh::initialize();
    gmsh::option::setNumber("General.Verbosity", 0);
    gmsh::open(conf_inst.project.cad_file);
    gmsh::merge(conf_inst.project.mesh_file);

    // Check if the format of the CAD file is geo.
    const bool is_geo_cad = conf_inst.project.cad_file.rfind(
                              std::string(".geo")) != std::string::npos;

    if (is_geo_cad)
      gmsh::model::geo::synchronize();
    else
      gmsh::model::occ::synchronize();

    std::visit(
      [is_geo_cad](auto &solver_ptr) {
        using T = std::decay_t<decltype(solver_ptr)>;
        if constexpr (!std::is_same_v<T, std::monostate>)
          solver_ptr->get_subdomain_topology().generate_topology(is_geo_cad);
        else
          throw std::runtime_error("LaplaceBEM solver is not created yet");
      },
      bem_);
    auto &manifold_description = std::visit(
      [](auto &solver_ptr) -> std::map<EntityTag, types::manifold_id> & {
        using T = std::decay_t<decltype(solver_ptr)>;
        if constexpr (!std::is_same_v<T, std::monostate>)
          return solver_ptr->get_manifold_description();
        else
          throw std::runtime_error("LaplaceBEM solver is not created yet");
      },
      bem_);
    auto &manifold_id_to_manifold = std::visit(
      [](auto &solver_ptr) -> std::map<types::manifold_id, Manifold<2, 3> *> & {
        using T = std::decay_t<decltype(solver_ptr)>;
        if constexpr (!std::is_same_v<T, std::monostate>)
          return solver_ptr->get_manifolds();
        else
          throw std::runtime_error("LaplaceBEM solver is not created yet");
      },
      bem_);
    auto &manifold_id_to_mapping_order = std::visit(
      [](auto &solver_ptr) -> std::map<types::manifold_id, unsigned int> & {
        using T = std::decay_t<decltype(solver_ptr)>;
        if constexpr (!std::is_same_v<T, std::monostate>)
          return solver_ptr->get_manifold_id_to_mapping_order();
        else
          throw std::runtime_error("LaplaceBEM solver is not created yet");
      },
      bem_);

    unsigned int manifold_id = 0;
    for (const auto &manifold : conf_inst.manifolds)
      {
        manifold.visit([&](const auto &alternative) {
          using T = std::decay_t<decltype(alternative)>;
          if constexpr (std::is_same_v<T, ConfOCCNURBSPatchManifold>)
            {
              // In this branch, a manifold object should be created for each
              // surface tag.
              for (const EntityTag tag : alternative.surface_tags)
                {
                  manifold_description[tag] = manifold_id;
                  TopoDS_Face face = gmsh::model::occ::getTopoDSFace(tag);
                  manifold_id_to_manifold[manifold_id] =
                    new OpenCASCADE::NURBSPatchManifold<2, 3>(
                      face, alternative.tolerance);
                  manifold_id_to_mapping_order[manifold_id] =
                    alternative.mapping_order;

                  manifold_id++;
                }
            }
          else
            {
              // In this branch, only a single manifold object is needed for all
              // surface tags.
              for (const EntityTag tag : alternative.surface_tags)
                manifold_description[tag] = manifold_id;

              manifold_id_to_mapping_order[manifold_id] =
                alternative.mapping_order;

              // Create the manifold.
              if constexpr (std::is_same_v<T, ConfSphericalManifold>)
                {
                  manifold_id_to_manifold[manifold_id] =
                    new SphericalManifold<2, 3>(alternative.center);
                }
              else if constexpr (std::is_same_v<
                                   T,
                                   ConfCylindricalManifoldCartesianAxis>)
                {
                  manifold_id_to_manifold[manifold_id] =
                    new CylindricalManifold<2, 3>(alternative.axis,
                                                  alternative.tolerance);
                }
              else if constexpr (std::is_same_v<T, ConfCylindricalManifold>)
                {
                  manifold_id_to_manifold[manifold_id] =
                    new CylindricalManifold<2, 3>(alternative.direction,
                                                  alternative.point_on_axis,
                                                  alternative.tolerance);
                }
              else if constexpr (std::is_same_v<T, ConfTorusManifold>)
                {
                  manifold_id_to_manifold[manifold_id] =
                    new TorusManifold<2>(alternative.R, alternative.r);
                }
              else if constexpr (
                std::is_same_v<T, ConfOCCNormalProjectionManifold> ||
                std::is_same_v<T, ConfOCCDirectionalProjectionManifold> ||
                std::is_same_v<T, ConfOCCNormalToMeshProjectionManifold>)
                {
                  // Create a shape containing all surfaces.
                  TopoDS_Compound compound;
                  BRep_Builder    builder;
                  builder.MakeCompound(compound);

                  for (const EntityTag tag : alternative.surface_tags)
                    builder.Add(compound, gmsh::model::occ::getTopoDSFace(tag));

                  TopoDS_Shape shape = compound;

                  if constexpr (std::is_same_v<T,
                                               ConfOCCNormalProjectionManifold>)
                    {
                      manifold_id_to_manifold[manifold_id] =
                        new OpenCASCADE::NormalProjectionManifold<2, 3>(
                          shape, alternative.tolerance);
                    }
                  else if constexpr (std::is_same_v<
                                       T,
                                       ConfOCCDirectionalProjectionManifold>)
                    {
                      manifold_id_to_manifold[manifold_id] =
                        new OpenCASCADE::DirectionalProjectionManifold<2, 3>(
                          shape, alternative.direction, alternative.tolerance);
                    }
                  else if constexpr (std::is_same_v<
                                       T,
                                       ConfOCCNormalToMeshProjectionManifold>)
                    {
                      manifold_id_to_manifold[manifold_id] =
                        new OpenCASCADE::NormalToMeshProjectionManifold<2, 3>(
                          shape, alternative.tolerance);
                    }
                }
              else
                throw std::runtime_error("Unrecognized manifold");

              manifold_id++;
            }
        });
      }

    gmsh::clear();
    gmsh::finalize();
  }

  void
  setupBoundaryConds()
  {
    std::visit(
      [&](auto &solver_ptr) {
        using T_solver_ptr = std::decay_t<decltype(solver_ptr)>;
        if constexpr (!std::is_same_v<T_solver_ptr, std::monostate>)
          {
            using T_solver  = typename T_solver_ptr::element_type;
            using real_type = typename T_solver::real_type;
            constexpr bool is_complex =
              std::is_same_v<T_solver, ComplexSolverType>;
            using Number = std::
              conditional_t<is_complex, std::complex<real_type>, real_type>;

            for (const auto &bc :
                 ConfigFile::instance().getConfig().boundary_conditions)
              {
                bc.visit([&](const auto &alternative) {
                  using T_bc = std::decay_t<decltype(alternative)>;
                  if (is_complex_value_spec(alternative.value_spec) !=
                      is_complex)
                    {
                      throw std::runtime_error(
                        "Boundary condition value type does not match the solver type");
                    }

                  auto func = create_function_from_value_spec<3, Number>(
                    alternative.value_spec);

                  if constexpr (std::is_same_v<T_bc, ConfBEMDirichletBC>)
                    {
                      solver_ptr->assign_dirichlet_bc(*func,
                                                      alternative.surface_tags);
                    }
                  else if constexpr (std::is_same_v<T_bc, ConfBEMNeumannBC>)
                    {
                      solver_ptr->assign_neumann_bc(*func,
                                                    alternative.surface_tags);
                    }
                  else
                    {
                      throw std::runtime_error(
                        "Not implemented, cannot setup boundary condition");
                    }

                  if constexpr (is_complex)
                    bc_complex_functions_.push_back(std::move(func));
                  else
                    bc_real_functions_.push_back(std::move(func));
                });
              }
          }
        else
          throw std::runtime_error("LaplaceBEM solver is not created yet");
      },
      bem_);
  }

  void
  runSolverAndOutput()
  {
    std::visit(
      [](auto &ptr) {
        using T = std::decay_t<decltype(ptr)>;
        if constexpr (!std::is_same_v<T, std::monostate>)
          ptr->run();
        else
          throw std::runtime_error("LaplaceBEM solver is not created yet");
      },
      bem_);
  }

  void
  startup()
  {
    initWorkDir(); // Make working directory hierarchies if it doesn't
                   // exist
    initLogger();  // Initialize deal.ii logger
    LogStream::Prefix prefix_string(LOG_PREFIX);

    // Initialize TBB/OpenBLAS/CUDA runtimes
    initRuntime(ConfigFile::instance().getConfig().parallel);

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
    std::visit(
      [](auto &ptr) {
        using T = std::decay_t<decltype(ptr)>;
        if constexpr (!std::is_same_v<T, std::monostate>)
          ptr->print_memory_consumption_table(deallog.get_file_stream());
        else
          throw std::runtime_error("LaplaceBEM solver is not created yet");
      },
      bem_);
  }

protected:
  ProblemType
  problemTypeLiteralToEnum(const ProblemTypeLiteral &literal)
  {
    switch (literal.value())
      {
        case ProblemTypeLiteral::value_of<"neumann">():
          return ProblemType::NeumannBCProblem;
        case ProblemTypeLiteral::value_of<"dirichlet">():
          return ProblemType::DirichletBCProblem;
        case ProblemTypeLiteral::value_of<"mixed">():
          return ProblemType::MixedBCProblem;
        case ProblemTypeLiteral::value_of<"robin">():
          return ProblemType::RobinBCProblem;
        default:
          throw std::runtime_error("Unknown problem type");
      }
  }

  PreconditionerType
  preconditionerTypeLiteralToEnum(const PreconditionerTypeLiteral &literal)
  {
    switch (literal.value())
      {
        case PreconditionerTypeLiteral::value_of<"factorization">():
          return PreconditionerType::HMatrixFactorization;
        case PreconditionerTypeLiteral::value_of<"operator">():
          return PreconditionerType::OperatorPreconditioning;
        case PreconditionerTypeLiteral::value_of<"identity">():
          return PreconditionerType::Identity;
        case PreconditionerTypeLiteral::value_of<"jacobi">():
          return PreconditionerType::Jacobi;
        case PreconditionerTypeLiteral::value_of<"block_jacobi">():
          return PreconditionerType::BlockJacobi;
        default:
          throw std::runtime_error("Unknown preconditioner type");
      }
  }

  IterativeSolverVmultType
  vmultTypeLiteralToEnum(const VmultTypeLiteral &literal)
  {
    switch (literal.value())
      {
        case VmultTypeLiteral::value_of<"serial_recursive">():
          return IterativeSolverVmultType::SerialRecursive;
        case VmultTypeLiteral::value_of<"serial_iterative">():
          return IterativeSolverVmultType::SerialIterative;
        case VmultTypeLiteral::value_of<"task_parallel">():
          return IterativeSolverVmultType::TaskParallel;
        default:
          throw std::runtime_error("Unknown vmult type");
      }
  }

  std::shared_ptr<std::ofstream>                    log_file_os_;
  SolverPtrType                                     bem_;
  std::vector<std::unique_ptr<Function<3, double>>> bc_real_functions_;
  std::vector<std::unique_ptr<Function<3, std::complex<double>>>>
    bc_complex_functions_;
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
