// Copyright (C) 2026 Jihuan Tian <jihuan_tian@hotmail.com>
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
#include "hbem_test_config.h"
#include "helmholtz/helmholtz_acoustic_bem.h"
#include "hmatrix/hmatrix.h"
#include "hmatrix/hmatrix_vmult_strategy.h"
#include "preconditioners/preconditioner_type.h"
#include "utilities/debug_tools.h"

using namespace dealii;
using namespace HierBEM;
using namespace std::literals::complex_literals;

/**
 * Function object for the Neumann boundary condition data. The analytical
 * expression is:
 * \f[ u(x) = 4 x_1 (1 + 2\sqrt{3}x_2 + 4ix_3) \exp(2\sqrt{3} x_2) \exp(i4x_3)
 * \f].
 */
class NeumannBC : public Function<3, std::complex<double>>
{
public:
  NeumannBC(const std::complex<double> &k2_,
            const std::complex<double> &k3_,
            const double                a_,
            const double                b_)
    : Function<3, std::complex<double>>()
    , k2(k2_)
    , k3(k3_)
    , a(a_)
    , b(b_)
  {}

  std::complex<double>
  value(const Point<3> &p, const unsigned int component = 0) const
  {
    (void)component;
    std::complex<double> ik2 = std::complex<double>(0., 1.0) * k2;
    std::complex<double> ik3 = std::complex<double>(0., 1.0) * k3;
    return (p(0) * b + (p(1) * ik2 + p(2) * ik3) * (a + b * p(0))) *
           std::exp(ik2 * p(1)) * std::exp(ik3 * p(2));
  }

private:
  /**
   * The second component of the wave vector.
   */
  std::complex<double> k2;
  /**
   * The third component of the wave vector.
   */
  std::complex<double> k3;
  double               a;
  double               b;
};

void
run_helmholtz_neumann_hmatrix_op_precond(
  const unsigned int             refinement,
  const IterativeSolverVmultType vmult_type)
{
  /**
   * @internal Pop out the default "DEAL" prefix string.
   */
  // Write run-time logs to file
  std::ofstream ofs(std::string("helmholtz-neumann-hmatrix-op-precond-vmult-") +
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

  ConfHelmholtzAcousticBEM bem_params;
  bem_params.kappa               = std::complex<double>(2.0, 0.);
  bem_params.problem_type        = ProblemType::NeumannBCProblem;
  bem_params.is_interior_problem = true;
  ConfHMatrix                hmat_params{4, 4, 0.8, 5, 0.01};
  ConfHMatrix                hmat_preconditioner_params{4, 4, 1.0, 2, 0.1};
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

  HelmholtzAcousticBEM<dim, spacedim> bem(bem_params,
                                          hmat_params,
                                          hmat_preconditioner_params,
                                          sauter_quad_params,
                                          sauter_quad_precond_params,
                                          linear_solver_params,
                                          op_precond_params,
                                          parallel_params);
  bem.set_project_name("helmholtz-neumann-hmatrix-op-precond");
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

  const Point<spacedim> center(0, 0, 0);
  const double          radius(1);
  GridGenerator::hyper_sphere(bem.get_triangulation(), center, radius);
  bem.get_triangulation().refine_global(refinement);
  std::string   mesh_file("surface-mesh.msh");
  std::ofstream mesh_out(mesh_file);
  write_msh_correct(bem.get_triangulation(), mesh_out);
  mesh_out.close();

  // Create the map from material ids to manifold ids. By default, the material
  // ids of all cells are zero, if the triangulation is created by a deal.ii
  // function in GridGenerator.
  bem.get_manifold_description()[0] = 0;

  // Create the map from manifold ids to manifold objects. Because in the
  // destructor of HelmholtzAcousticBEM the manifold objects will be released,
  // the manifold object here is created on the heap.
  SphericalManifold<dim, spacedim> *spherical_manifold =
    new SphericalManifold<dim, spacedim>(center);
  bem.get_manifolds()[0] = spherical_manifold;

  // Create the map from manifold id to mapping order.
  bem.get_manifold_id_to_mapping_order()[0] = 2;

  // Build surface-to-volume and volume-to-surface relationship.
  bem.get_subdomain_topology().generate_single_domain_topology_for_dealii_model(
    {0});

  timer.stop();
  print_wall_time(deallog, timer, "read mesh");

  timer.start();

  NeumannBC neumann_bc(-std::sqrt(3.0) * 2i, 4.0, 0., 4.0);
  bem.assign_neumann_bc(neumann_bc);

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
