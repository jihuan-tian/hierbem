// Copyright (C) 2024-2026 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * \file dirichlet-hmatrix-two-spheres.cc
 * \brief Baseline test for solving the Laplace problem with Dirichlet boundary
 * condition using H-matrix based BEM. The two sphere model is solved.
 *
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2022-09-23
 */

#include <deal.II/base/logstream.h>
#include <deal.II/base/multithread_info.h>

#include <deal.II/grid/manifold_lib.h>

#include <cpptrace/from_current.hpp>
#include <cuda_runtime.h>
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
#include "grid/grid_in_ext.h"
#include "hbem_test_config.h"
#include "hmatrix/hmatrix.h"
#include "hmatrix/hmatrix_vmult_strategy.h"
#include "laplace/laplace_bem.h"
#include "preconditioners/preconditioner_type.h"
#include "utilities/debug_tools.h"

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

    ConfLaplaceBEM bem_params{conf_inst.bem.fe_order_for_dirichlet_space,
                              conf_inst.bem.fe_order_for_neumann_space,
                              problemTypeLiteralToEnum(
                                conf_inst.bem.problem_type),
                              conf_inst.bem.is_interior_problem};
    // ConfHMatrix                hmat_params{64, 64, 8, 4, 0.8, 5, 0.01,
    // false}; ConfHMatrix                hmat_preconditioner_params{64, 64, 8,
    // 4, 1.0, 2, 0.1};

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
    std::ifstream mesh_in(HBEM_TEST_MODEL_DIR "two-spheres-fine.msh");
    read_msh(mesh_in, bem.get_triangulation());
    bem.get_subdomain_topology().generate_topology(HBEM_TEST_MODEL_DIR
                                                   "two-spheres.brep",
                                                   HBEM_TEST_MODEL_DIR
                                                   "two-spheres-fine.msh");

    if (conf_inst.bem.mesh_refinement > 0)
      bem.get_triangulation().refine_global(conf_inst.bem.mesh_refinement);

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
    bem.get_manifold_id_to_mapping_order()[0] = 2;
    bem.get_manifold_id_to_mapping_order()[1] = 2;

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
