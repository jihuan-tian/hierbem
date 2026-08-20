// Copyright (C) 2022-2026 Jihuan Tian <jihuan_tian@hotmail.com>
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
#include <deal.II/base/numbers.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <string>

#include "bem/types.h"
#include "config_file/config_structs.h"
#include "config_file/cu_related.h"
#include "grid/grid_in_ext.h"
#include "grid/grid_out_ext.h"
#include "hmatrix/hmatrix.h"
#include "hmatrix/hmatrix_vmult_strategy.h"
#include "laplace/laplace_bem.h"
#include "preconditioners/preconditioner_type.h"
#include "utilities/debug_tools.h"

using namespace dealii;
using namespace HierBEM;

/**
 * Function object for the Dirichlet boundary condition data, which is
 * also the solution of the Neumann problem. The analytical expression is:
 * \f[
 * u=\frac{1}{4\pi\norm{x-x_0}}
 * \f]
 */
class DirichletBC : public Function<3, std::complex<double>>
{
public:
  // N.B. This function should be defined outside class NeumannBC or class
  // Example2, if no inline.
  DirichletBC()
    : Function<3, std::complex<double>>()
    , x0(0.25, 0.25, 0.25)
  {}

  DirichletBC(const Point<3> &x0)
    : Function<3, std::complex<double>>()
    , x0(x0)
  {}

  std::complex<double>
  value(const Point<3> &p, const unsigned int component = 0) const
  {
    (void)component;
    const double amplitude = 1.0 / 4.0 / numbers::PI / (p - x0).norm();
    // In the complex valued case, we assign a fixed phase angle to the
    // potential distribution.
    const double angle = numbers::PI / 3.0;
    return std::complex<double>(amplitude * std::cos(angle),
                                amplitude * std::sin(angle));
  }

private:
  /**
   * Location of the Dirac point source \f$\delta(x-x_0)\f$.
   */
  Point<3> x0;
};

void
run_dirichlet_hmatrix_op_precond_complex(
  const unsigned int             refinement,
  const IterativeSolverVmultType vmult_type,
  const bool                     cpu_serial_without_producer_consumer)
{
  /**
   * @internal Pop out the default "DEAL" prefix string.
   */
  // Write run-time logs to file
  std::ofstream ofs(std::string("dirichlet-hmatrix-op-precond-complex-vmult-") +
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
  bem_params.problem_type        = ProblemType::DirichletBCProblem;
  bem_params.is_interior_problem = true;
  ConfHMatrix hmat_params{
    4, 4, 8, 4, 0.8, 5, 5, 0.01, cpu_serial_without_producer_consumer, 10};
  ConfHMatrix hmat_preconditioner_params{
    4, 4, 8, 4, 1.0, 2, 2, 0.1, cpu_serial_without_producer_consumer, 10};
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
  bem.set_project_name("dirichlet-hmatrix-op-precond-complex");
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

  /**
   * @internal Set the Dirac source location according to interior or exterior
   * problem.
   */
  Point<spacedim> source_loc;

  if (bem_params.is_interior_problem)
    {
      source_loc = Point<spacedim>(1, 1, 1);
    }
  else
    {
      source_loc = Point<spacedim>(0.25, 0.25, 0.25);
    }

  const Point<spacedim> center(0, 0, 0);
  const double          radius(1);

  Triangulation<dim, spacedim> tria;
  GridGenerator::hyper_sphere(tria, center, radius);
  tria.refine_global(refinement);
  std::string   mesh_file("surface-mesh.msh");
  std::ofstream mesh_out(mesh_file);
  write_msh_correct(tria, mesh_out);
  mesh_out.close();

  // Reread the mesh as a single level triangulation.
  std::ifstream mesh_in(mesh_file);
  read_msh(mesh_in, bem.get_triangulation(), false, true, false);
  mesh_in.close();

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
  bem.get_manifold_id_to_mapping_order()[0] = 1;

  // Build surface-to-volume and volume-to-surface relationship.
  bem.get_subdomain_topology().generate_single_domain_topology_for_dealii_model(
    {0});

  timer.stop();
  print_wall_time(deallog, timer, "read mesh");

  timer.start();

  DirichletBC dirichlet_bc(source_loc);
  bem.assign_dirichlet_bc(dirichlet_bc);

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
