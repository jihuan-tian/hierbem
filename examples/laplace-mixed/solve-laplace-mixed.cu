// Copyright (C) 2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

#include <deal.II/base/function.h>

#include <deal.II/grid/manifold_lib.h>

#include <cuda_runtime.h>

#include <fstream>
#include <iostream>

#include "grid/grid_in_ext.h"
#include "hbem_test_config.h"
#include "hmatrix/hmatrix_vmult_strategy.h"
#include "laplace/laplace_bem.h"
#include "preconditioners/preconditioner_type.h"

using namespace dealii;
using namespace HierBEM;

namespace HierBEM
{
  namespace CUDAWrappers
  {
    extern cudaDeviceProp device_properties;
  }
} // namespace HierBEM

int
main()
{
  const size_t stack_size = 1024 * 10;
  AssertCuda(cudaDeviceSetLimit(cudaLimitStackSize, stack_size));
  AssertCuda(
    cudaGetDeviceProperties(&HierBEM::CUDAWrappers::device_properties, 0));

  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  LaplaceBEM<dim, spacedim, double, double> bem(
    1, // fe order for dirichlet space
    0, // fe order for neumann space
    LaplaceBEM<dim, spacedim, double, double>::ProblemType::MixedBCProblem,
    true,                        // is interior problem
    32,                          // n_min for cluster tree
    32,                          // n_min for block cluster tree
    0.8,                         // eta for H-matrix
    10,                          // max rank for H-matrix
    0.01,                        // aca epsilon for H-matrix
    1.0,                         // eta for preconditioner
    5,                           // max rank for preconditioner
    0.1,                         // aca epsilon for preconditioner
    MultithreadInfo::n_threads() // Number of threads used for ACA
  );
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
