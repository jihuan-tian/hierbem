// Copyright (C) 2025-2026 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

#include <deal.II/base/function.h>
#include <deal.II/base/multithread_info.h>

#include <deal.II/grid/manifold_lib.h>

#include <fstream>
#include <iostream>

#include "bem/types.h"
#include "config_file/config_structs.h"
#include "config_file/cu_related.h"
#include "grid/grid_in_ext.h"
#include "hbem_test_config.h"
#include "hmatrix/hmatrix_vmult_strategy.h"
#include "laplace/laplace_bem.h"
#include "preconditioners/preconditioner_type.h"

using namespace dealii;
using namespace HierBEM;

int
main()
{
  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  ConfLaplaceBEM bem_params;
  bem_params.problem_type        = ProblemType::MixedBCProblem;
  bem_params.is_interior_problem = true;
  ConfHMatrix      hmat_params{32, 32, 8, 4, 0.8, 10, 0.01, false};
  ConfHMatrix      hmat_preconditioner_params{32, 32, 8, 4, 1.0, 5, 0.1, false};
  ConfSauterQuad   sauter_quad_params;
  ConfSauterQuad   sauter_quad_precond_params;
  ConfLinearSolver linear_solver_params;
  ConfOperatorPreconditioner op_precond_params;
  ConfParallelization        parallel_params;

  // Set TBB thread num.
  if (parallel_params.tbb_thread_num == -1)
    MultithreadInfo::set_thread_limit(MultithreadInfo::n_threads());
  else
    MultithreadInfo::set_thread_limit(parallel_params.tbb_thread_num);

  // Initialize CUDA stack size and device properties.
  initCudaRuntime(parallel_params);

  LaplaceBEM<dim, spacedim, double, double> bem(bem_params,
                                                hmat_params,
                                                hmat_preconditioner_params,
                                                sauter_quad_params,
                                                sauter_quad_precond_params,
                                                linear_solver_params,
                                                op_precond_params,
                                                parallel_params);
  bem.set_project_name("solve-laplace-mixed");
  bem.set_preconditioner_type(PreconditionerType::OperatorPreconditioning);
  bem.set_iterative_solver_vmult_type(
    IterativeSolverVmultType::SerialIterative);

  std::ifstream mesh_in(HBEM_TEST_MODEL_DIR "cylinder.msh");
  read_msh(mesh_in, bem.get_triangulation());
  bem.get_subdomain_topology().generate_topology(HBEM_TEST_MODEL_DIR
                                                 "cylinder.brep",
                                                 HBEM_TEST_MODEL_DIR
                                                 "cylinder.msh");
  bem.get_triangulation().refine_global(1);

  // Create the map from manifold ids to manifold objects. Because in the
  // destructor of LaplaceBEM the manifold objects will be released, the
  // manifold object here is created on the heap.
  FlatManifold<dim, spacedim> *flat_manifold =
    new FlatManifold<dim, spacedim>();
  CylindricalManifold<dim, spacedim> *cyl_manifold =
    new CylindricalManifold<dim, spacedim>(2);
  bem.get_manifolds()[0] = flat_manifold;
  bem.get_manifolds()[1] = cyl_manifold;

  // Create the map from material ids to manifold ids.
  bem.get_manifold_description()[1] = 1;
  bem.get_manifold_description()[2] = 0;
  bem.get_manifold_description()[3] = 0;

  // Create the map from manifold id to mapping order.
  bem.get_manifold_id_to_mapping_order()[0] = 1;
  bem.get_manifold_id_to_mapping_order()[1] = 2;

  Functions::ConstantFunction<spacedim> f1(1.0);
  Functions::ConstantFunction<spacedim> f2(0.0);
  bem.assign_neumann_bc(f2, 1);
  bem.assign_dirichlet_bc(f1, 2);
  bem.assign_dirichlet_bc(f2, 3);

  bem.run();
}
