// Copyright (C) 2023-2026 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

#include <deal.II/base/logstream.h>
#include <deal.II/base/multithread_info.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <cmath>
#include <complex>
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

// Dirichlet boundary conditions on the left and top surface of the L-shape
class DirichletBC : public Function<3, std::complex<double>>
{
public:
  std::complex<double>
  value(const Point<3> &p, const unsigned int component = 0) const
  {
    (void)component;

    if (p(0) <= 1e-6)
      {
        // left surface
        return std::complex<double>(0.0);
      }
    else
      {
        // top surface
        const double angle = numbers::PI / 3.0;
        return std::complex(10.0 * std::cos(angle), 10.0 * std::sin(angle));
      }
  }
};

// Neumann boundary conditions on the other surfaces of the L-shape
class NeumannBC : public Function<3, std::complex<double>>
{
public:
  std::complex<double>
  value(const Point<3> &p, const unsigned int component = 0) const
  {
    (void)component;
    (void)p;

    return std::complex<double>(0.);
  }
};

void
run_mixed_l_shape_op_precond_complex(const IterativeSolverVmultType vmult_type)
{
  // Write run-time logs to file
  std::ofstream ofs(std::string("mixed-l-shape-op-precond-complex-vmult-") +
                    std::string(vmult_type_name(vmult_type)) +
                    std::string(".log"));
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

  ConfLaplaceBEM bem_params;
  bem_params.problem_type        = ProblemType::MixedBCProblem;
  bem_params.is_interior_problem = true;
  ConfHMatrix hmat_params{4, 32, 8, 4, 0.8, 5, 5, 0.01, false, 10};
  ConfHMatrix hmat_preconditioner_params{
    4, 32, 8, 4, 1.0, 2, 2, 0.1, false, 10};
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

  LaplaceBEM<dim, spacedim, std::complex<double>, double> bem(
    bem_params,
    hmat_params,
    hmat_preconditioner_params,
    sauter_quad_params,
    sauter_quad_precond_params,
    linear_solver_params,
    op_precond_params,
    parallel_params);
  bem.set_project_name("mixed-l-shape-op-precond-complex");
  bem.set_preconditioner_type(PreconditionerType::OperatorPreconditioning);
  bem.set_iterative_solver_vmult_type(vmult_type);
  if (vmult_type == IterativeSolverVmultType::TaskParallel)
    {
      HMatrix<spacedim, std::complex<double>>::set_leaf_set_traversal_method(
        HMatrix<spacedim,
                std::complex<double>>::SpaceFillingCurveType::Hilbert);
    }

  timer.stop();
  print_wall_time(deallog, timer, "program preparation");

  timer.start();

  // Read the 3D mesh.
  std::ifstream           mesh_file(HBEM_TEST_MODEL_DIR "l-shape.msh");
  Triangulation<spacedim> tria;
  GridIn<spacedim>        grid_in;
  grid_in.attach_triangulation(tria);
  grid_in.read_msh(mesh_file);

  // Create the map from material ids to manifold ids.
  bem.get_manifold_description()[1] = 0;
  bem.get_manifold_description()[2] = 0;
  for (types::material_id i = 19; i <= 24; i++)
    {
      bem.get_manifold_description()[i] = 0;
    }

  FlatManifold<dim, spacedim> *flat_manifold =
    new FlatManifold<dim, spacedim>();
  bem.get_manifolds()[0] = flat_manifold;

  // Extract the surface mesh.
  Triangulation<dim, spacedim> surface_tria(
    Triangulation<dim,
                  spacedim>::MeshSmoothing::limit_level_difference_at_vertices);
  surface_tria.set_manifold(0, *flat_manifold);
  bem.extract_surface_triangulation(tria, std::move(surface_tria), true);

  // Create the map from manifold id to mapping order.
  bem.get_manifold_id_to_mapping_order()[0] = 1;

  // Build surface-to-volume and volume-to-surface relationship.
  bem.get_subdomain_topology().generate_single_domain_topology_for_dealii_model(
    {1, 2, 19, 20, 21, 22, 23, 24});

  timer.stop();
  print_wall_time(deallog, timer, "read mesh");

  timer.start();

  DirichletBC dirichlet_bc;
  NeumannBC   neumann_bc;

  bem.assign_dirichlet_bc(dirichlet_bc, {1, 2});
  bem.assign_neumann_bc(neumann_bc, {19, 20, 21, 22, 23, 24});

  timer.stop();
  print_wall_time(deallog, timer, "assign boundary conditions");

  if (bem.validate_subdomain_topology())
    {
      timer.start();

      bem.run();

      timer.stop();
      print_wall_time(deallog, timer, "run the solver");

      deallog << "Program exits with a total wall time " << timer.wall_time()
              << "s" << std::endl;

      bem.print_memory_consumption_table(deallog.get_file_stream());
    }
  else
    {
      deallog << "Invalid subdomains!" << std::endl;
    }

  ofs.close();
}
