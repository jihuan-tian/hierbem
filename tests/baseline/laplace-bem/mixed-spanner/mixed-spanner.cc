// Copyright (C) 2023-2026 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file mixed-spanner.cc
 * @brief Baseline test for solving the Laplace mixed boundary value problem
 * using \hmat.
 *
 * @ingroup test_cases
 * @author Jihuan Tian
 * @date 2023-05-24
 */

#include <deal.II/base/logstream.h>
#include <deal.II/base/multithread_info.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>

#include <boost/program_options.hpp>

#include <cpptrace/from_current.hpp>
#include <fmt/core.h>
#include <fmt/format.h>

#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "bem/types.h"
#include "config_file/config_file.h"
#include "config_file/config_structs.h"
#include "config_file/cu_related.h"
#include "hbem_test_config.h"
#include "hmatrix/hmatrix.h"
#include "hmatrix/hmatrix_vmult_strategy.h"
#include "laplace/laplace_bem.h"
#include "preconditioners/preconditioner_type.h"
#include "utilities/debug_tools.h"

using namespace dealii;
using namespace HierBEM;

/**
 * Function object for the Dirichlet boundary condition data.
 */
class DirichletBC : public Function<3>
{
public:
  double
  value(const Point<3> &p, const unsigned int component = 0) const
  {
    (void)component;

    if (p(0) < 0)
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

ProblemType
problemTypeLiteralToEnum(const ProblemTypeLiteral &literal)
{
  switch (literal.value())
    {
      case ProblemTypeLiteral::value_of<"dirichlet">():
        return ProblemType::DirichletBCProblem;
      case ProblemTypeLiteral::value_of<"neumann">():
        return ProblemType::NeumannBCProblem;
      case ProblemTypeLiteral::value_of<"mixed">():
        return ProblemType::MixedBCProblem;
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

void
initWorkDir()
{
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

int
main(int argc, char *argv[])
{
  CPPTRACE_TRY
  {
    if (argc != 2)
      {
        std::cerr << "Usage: " << argv[0] << " <config file>" << std::endl;
        return 1;
      }

    ConfigFile::instance().initialize(argv[1]); // Load configuration file
    initWorkDir();
    const auto       &conf_inst    = ConfigFile::instance().getConfig();
    const std::string project_name = conf_inst.project.project_name.value();

    std::ofstream ofs(project_name + std::string(".log"));
    deallog.pop();
    deallog.depth_console(0);
    deallog.depth_file(5);
    deallog.attach(ofs);

    LogStream::Prefix prefix_string("HierBEM");

    /**
     * @internal Create and start the timer.
     */
    Timer timer;

    const unsigned int dim      = 2;
    const unsigned int spacedim = 3;

    ConfLaplaceBEM bem_params{conf_inst.bem.mesh_refinement,
                              conf_inst.bem.fe_order_for_dirichlet_space,
                              conf_inst.bem.fe_order_for_neumann_space,
                              problemTypeLiteralToEnum(
                                conf_inst.bem.problem_type),
                              conf_inst.bem.is_interior_problem};

    // Set TBB thread num.
    if (conf_inst.parallel.tbb_thread_num == -1)
      MultithreadInfo::set_thread_limit(MultithreadInfo::n_threads());
    else
      MultithreadInfo::set_thread_limit(conf_inst.parallel.tbb_thread_num);

    // Initialize CUDA stack size and device properties.
    initCudaRuntime(conf_inst.parallel);

    LaplaceBEM<dim, spacedim> bem(bem_params,
                                  conf_inst.hmatrix,
                                  conf_inst.hmatrix_precond,
                                  conf_inst.sauter_quad,
                                  conf_inst.sauter_quad_precond,
                                  conf_inst.linear_solver,
                                  conf_inst.op_precond,
                                  conf_inst.parallel);
    bem.set_project_name(project_name);
    bem.set_preconditioner_type(
      preconditionerTypeLiteralToEnum(conf_inst.bem.precond_type));
    bem.set_iterative_solver_vmult_type(
      vmultTypeLiteralToEnum(conf_inst.bem.vmult_type));

    timer.stop();
    print_wall_time(deallog, timer, "program preparation");

    timer.start();

    std::ifstream mesh_in(std::string(HBEM_TEST_MODEL_DIR) +
                          conf_inst.project.mesh_file);
    // Use @p dealii::GridIn to read the mesh, because there are physical groups
    // defined in Gmsh for the spanner model and we need to read physical group
    // ids as material ids. @p read_msh in HierBEM ignore physical group ids and
    // directly read elementary entity tags as material ids.
    GridIn<dim, spacedim> grid_in;
    grid_in.attach_triangulation(bem.get_triangulation());
    grid_in.read_msh(mesh_in);
    // Build surface-to-volume and volume-to-surface relationship.
    bem.get_subdomain_topology()
      .generate_single_domain_topology_for_dealii_model({0, 1, 2});

    // Create a flat manifold.
    FlatManifold<dim, spacedim> *flat_manifold =
      new FlatManifold<dim, spacedim>();
    bem.get_manifolds()[0] = flat_manifold;

    // Create the map from manifold id to mapping order.
    bem.get_manifold_id_to_mapping_order()[0] = 1;

    // Create the map from material ids to manifold ids.
    bem.get_manifold_description()[0] = 0;
    bem.get_manifold_description()[1] = 0;
    bem.get_manifold_description()[2] = 0;

    timer.stop();
    print_wall_time(deallog, timer, "read mesh");

    timer.start();

    // Assign boundary conditions.
    DirichletBC dirichlet_bc;
    NeumannBC   neumann_bc;

    bem.assign_dirichlet_bc(dirichlet_bc, {1, 2});
    bem.assign_neumann_bc(neumann_bc, 0);

    timer.stop();
    print_wall_time(deallog, timer, "assign boundary conditions");

    timer.start();

    bem.run();

    timer.stop();
    print_wall_time(deallog, timer, "run the solver");

    deallog << "Program exits with a total wall time " << timer.wall_time()
            << "s" << std::endl;

    bem.print_memory_consumption_table(deallog.get_file_stream());

    return 0;
  }
  CPPTRACE_CATCH(const std::exception &e)
  {
    std::cerr << "Exception: " << e.what() << std::endl;
    cpptrace::from_current_exception().print();
    return 1;
  }
}
