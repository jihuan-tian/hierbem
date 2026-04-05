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
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include "bem/types.h"
#include "cad_mesh/gmsh_manipulation.h"
#include "config_file/config_structs.h"
#include "config_file/cu_related.h"
#include "hmatrix/hmatrix_vmult_strategy.h"
#include "laplace/laplace_bem.h"
#include "preconditioners/preconditioner_type.h"
#include "utilities/debug_tools.h"

using namespace dealii;
using namespace HierBEM;

// Function object for the Dirichlet boundary condition data.
class DirichletBC : public Function<3>
{
public:
  DirichletBC()
    : Function<3>()
    , x0(0., 0., 0.)
  {}

  DirichletBC(const Point<3> &x0_)
    : Function<3>()
    , x0(x0_)
  {}

  double
  value(const Point<3> &p, const unsigned int component = 0) const
  {
    (void)component;
    return 1.0 / 4.0 / numbers::PI / (p - x0).norm();
  }

private:
  // Location of the unit Dirac source.
  Point<3> x0;
};


// Function object for the analytical solution.
class AnalyticalSolution : public Function<3>
{
public:
  AnalyticalSolution()
    : Function<3>()
    , x0(0., 0., 0.)
    , center(0.0, 0.0, 0.0)
    , radius(1.0)
  {}

  AnalyticalSolution(const Point<3> &x0_,
                     const Point<3> &center_,
                     double          radius_)
    : Function<3>()
    , x0(x0_)
    , center(center_)
    , radius(radius_)
  {}

  double
  value(const Point<3> &p, const unsigned int component = 0) const
  {
    (void)component;

    Tensor<1, 3> diff_vector = x0 - p;

    return ((p - center) * diff_vector) / 4.0 / numbers::PI /
           std::pow(diff_vector.norm(), 3) / radius;
  }

private:
  // Location of the unit Dirac source.
  Point<3> x0;
  Point<3> center;
  double   radius;
};

template <int dim,
          int spacedim,
          typename RangeNumberType,
          typename KernelNumberType>
void
output_analytical_solution(
  const LaplaceBEM<dim, spacedim, RangeNumberType, KernelNumberType> &bem,
  const Point<spacedim> &source_loc,
  const Point<spacedim> &center,
  const double           radius)
{
  const DoFHandler<dim, spacedim> &dof_handler = bem.get_dof_handler_neumann();
  Vector<double>                   analytical_solution(dof_handler.n_dofs());
  // We use the 2nd order mapping to interpolate the analytical solution.
  VectorTools::interpolate(bem.get_mappings()[1]->get_mapping(),
                           dof_handler,
                           AnalyticalSolution(source_loc, center, radius),
                           analytical_solution);
  std::ofstream out("analytical-solution.output");
  print_vector_to_mat(out, "analytical_solution", analytical_solution);
  out.close();
}

int
main()
{
  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  ConfLaplaceBEM bem_params;
  bem_params.problem_type        = ProblemType::DirichletBCProblem;
  bem_params.is_interior_problem = true;
  ConfHMatrix      hmat_params{32, 32, 0.8, 10, 0.01, false};
  ConfHMatrix      hmat_preconditioner_params{32, 32, 1.0, 5, 0.1, false};
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
  bem.set_project_name("solve-laplace-dirichlet");
  bem.set_preconditioner_type(PreconditionerType::OperatorPreconditioning);
  bem.set_iterative_solver_vmult_type(
    IterativeSolverVmultType::SerialIterative);

  const Point<spacedim> center(0, 0, 0);
  const double          radius(1);
  const unsigned int    refinement = 4;

  Triangulation<dim, spacedim> tria;
  GridGenerator::hyper_sphere(tria, center, radius);
  tria.refine_global(refinement);
  bem.get_triangulation() = std::move(tria);

  // Create the map from material ids to manifold ids. By default, the material
  // ids of all cells are zero, if the triangulation is created by a deal.ii
  // function in GridGenerator.
  bem.get_manifold_description()[0] = 0;

  // Create the map from manifold ids to manifold objects. Because in the
  // destructor of LaplaceBEM the manifold objects will be released, the
  // manifold object here is created on the heap.
  SphericalManifold<dim, spacedim> *spherical_manifold =
    new SphericalManifold<dim, spacedim>(center);
  bem.get_manifolds()[0] = spherical_manifold;

  // Create the map from manifold id to mapping order.
  bem.get_manifold_id_to_mapping_order()[0] = 2;

  // Build surface-to-volume and volume-to-surface relationship.
  bem.get_subdomain_topology().generate_single_domain_topology_for_dealii_model(
    {0});

  Point<spacedim> source_loc(1., 1., 1.);
  DirichletBC     dirichlet_bc(source_loc);
  bem.assign_dirichlet_bc(dirichlet_bc);

  bem.run();

  output_analytical_solution(bem, source_loc, center, radius);
}
