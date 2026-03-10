// Copyright (C) 2023-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file laplace-bem-mixed-spanner-model.cc
 * @brief Verify solve Laplace mixed boundary value problem using \hmat.
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

#include <fstream>
#include <iostream>

#include "bem/types.h"
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

namespace po = boost::program_options;

struct CmdOpts
{
  unsigned int             dirichlet_space_fe_order;
  unsigned int             neumann_space_fe_order;
  unsigned int             mapping_order;
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

int
main(int argc, char *argv[])
{
  CmdOpts opts = parse_cmdline(argc, argv);

  /**
   * @internal Pop out the default "DEAL" prefix string.
   */
  // Write run-time logs to file
  std::ofstream ofs("hierbem.log");
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

  ConfLaplaceBEM             bem_params{opts.dirichlet_space_fe_order,
                            opts.neumann_space_fe_order,
                            ProblemType::MixedBCProblem,
                            true};
  ConfHMatrix                hmat_params{64, 64, 0.8, 5, 0.01};
  ConfHMatrix                hmat_preconditioner_params{64, 64, 1.0, 1, 0.1};
  ConfSauterQuad             sauter_quad_params;
  ConfSauterQuad             sauter_quad_precond_params;
  ConfLinearSolver           linear_solver_params;
  ConfOperatorPreconditioner op_precond_params;
  ConfParallelization        parallel_params;

  // Set TBB thread num.
  if (parallel_params.tbb_thread_num == -1)
    MultithreadInfo::set_thread_limit(MultithreadInfo::n_threads());
  else
    MultithreadInfo::set_thread_limit(parallel_params.tbb_thread_num);

  // Initialize CUDA stack size and device properties.
  initCudaRuntime(parallel_params);

  LaplaceBEM<dim, spacedim> bem(bem_params,
                                hmat_params,
                                hmat_preconditioner_params,
                                sauter_quad_params,
                                sauter_quad_precond_params,
                                linear_solver_params,
                                op_precond_params,
                                parallel_params);
  bem.set_project_name("laplace-bem-mixed-spanner");
  bem.set_preconditioner_type(opts.precond_type);
  bem.set_iterative_solver_vmult_type(opts.vmult_type);
  if (opts.vmult_type == IterativeSolverVmultType::TaskParallel)
    {
      HMatrix<spacedim, double>::set_leaf_set_traversal_method(
        HMatrix<spacedim, double>::SpaceFillingCurveType::Hilbert);
    }

  timer.stop();
  print_wall_time(deallog, timer, "program preparation");

  timer.start();

  std::ifstream           mesh_file(HBEM_TEST_MODEL_DIR "spanner.msh");
  Triangulation<spacedim> tria;
  GridIn<spacedim>        grid_in;
  grid_in.attach_triangulation(tria);
  grid_in.read_msh(mesh_file);

  // Create the map from material ids to manifold ids.
  bem.get_manifold_description()[0] = 0;
  bem.get_manifold_description()[1] = 0;
  bem.get_manifold_description()[2] = 0;

  FlatManifold<dim, spacedim> *flat_manifold =
    new FlatManifold<dim, spacedim>();
  bem.get_manifolds()[0] = flat_manifold;

  Triangulation<dim, spacedim> surface_tria;
  surface_tria.set_manifold(0, *flat_manifold);
  bem.extract_surface_triangulation(tria, std::move(surface_tria), true);

  // Create the map from manifold id to mapping order.
  bem.get_manifold_id_to_mapping_order()[0] = opts.mapping_order;

  // Build surface-to-volume and volume-to-surface relationship.
  bem.get_subdomain_topology().generate_single_domain_topology_for_dealii_model(
    {0, 1, 2});

  timer.stop();
  print_wall_time(deallog, timer, "read mesh");

  timer.start();

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

  deallog << "Program exits with a total wall time " << timer.wall_time() << "s"
          << std::endl;

  bem.print_memory_consumption_table(deallog.get_file_stream());

  return 0;
}
