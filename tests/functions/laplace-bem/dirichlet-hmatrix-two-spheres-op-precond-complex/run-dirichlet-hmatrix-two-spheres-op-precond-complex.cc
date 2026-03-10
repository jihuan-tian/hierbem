// Copyright (C) 2025 Jihuan Tian <jihuan_tian@hotmail.com>
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

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/numerics/data_out.h>

#include <boost/math/constants/constants.hpp>

#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>

#include "bem/types.h"
#include "config_file/config_structs.h"
#include "config_file/cu_related.h"
#include "grid/grid_in_ext.h"
#include "hbem_test_config.h"
#include "hmatrix/hmatrix.h"
#include "hmatrix/hmatrix_vmult_strategy.h"
#include "laplace/laplace_bem.h"
#include "platform_shared/laplace_kernels.h"
#include "postprocessing/data_out_ext.h"
#include "preconditioners/preconditioner_type.h"
#include "utilities/debug_tools.h"

using namespace dealii;
using namespace HierBEM;

class DirichletBC : public Function<3, std::complex<double>>
{
public:
  std::complex<double>
  value(const Point<3> &p, const unsigned int component = 0) const
  {
    (void)component;

    const double angle = numbers::PI / 3.0;
    if (p(0) < 0)
      {
        return std::complex<double>(10 * std::cos(angle), 10 * std::sin(angle));
      }
    else
      {
        return std::complex<double>(-10 * std::cos(angle),
                                    -10 * std::sin(angle));
      }
  }
};

/**
 * Output the results of potential and conormal trace at a plane and results
 * of potential for a volume.
 */
void
output_results_at_targets(LaplaceBEM<2, 3, std::complex<double>, double> &bem)
{
  Triangulation<2, 3> plane;
  GridGenerator::subdivided_hyper_rectangle(plane,
                                            {5, 5},
                                            Point<2>(-4, -4),
                                            Point<2>(4, 4));
  GridTools::rotate(boost::math::constants::pi<double>() / 2, 1, plane);
  GridOut().write_msh(plane, "plane.msh");
  bem.output_results_on_target_tria("plane.vtk", plane, 1);

  Triangulation<3, 3> cube;
  GridGenerator::subdivided_hyper_cube(cube, 5, 3., 6.);
  GridOut().write_msh(cube, "cube.msh");
  bem.output_results_on_target_tria("cube.vtk", cube, 1);
}

void
run_dirichlet_hmatrix_two_spheres_op_precond_complex(
  const IterativeSolverVmultType vmult_type)
{
  /**
   * @internal Pop out the default "DEAL" prefix string.
   */
  // Write run-time logs to file
  std::ofstream ofs(
    std::string("dirichlet-hmatrix-two-spheres-op-precond-complex-vmult-") +
    std::string(vmult_type_name(vmult_type)) + std::string(".log"));
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
  bem_params.is_interior_problem = false;
  ConfHMatrix                hmat_params{16, 16, 0.8, 10, 0.01};
  ConfHMatrix                hmat_preconditioner_params{16, 16, 1.0, 5, 0.1};
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
  bem.set_project_name("dirichlet-hmatrix-two-spheres-op-precond-complex");
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

  std::ifstream mesh_in(HBEM_TEST_MODEL_DIR "two-spheres.msh");
  read_msh(mesh_in, bem.get_triangulation());
  bem.get_subdomain_topology().generate_topology(HBEM_TEST_MODEL_DIR
                                                 "two-spheres.brep",
                                                 HBEM_TEST_MODEL_DIR
                                                 "two-spheres.msh");

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
  bem.get_manifold_id_to_mapping_order()[0] = 1;
  bem.get_manifold_id_to_mapping_order()[1] = 1;

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

  if (bem.validate_subdomain_topology())
    {
      timer.start();

      bem.run();
      output_results_at_targets(bem);

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
