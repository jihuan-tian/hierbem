// Copyright (C) 2022-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

#include <deal.II/base/logstream.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>

#include "hbem_test_config.h"
#include "laplace/laplace_bem.h"

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
run_dirichlet_full_matrix_complex()
{
  // Write run-time logs to file
  std::ofstream ofs("dirichlet-full-matrix-complex.log");
  deallog.pop();
  deallog.depth_console(0);
  deallog.depth_file(5);
  deallog.attach(ofs);

  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  const bool is_interior_problem = true;
  LaplaceBEM<dim, spacedim, std::complex<double>, double> bem(
    1,
    0,
    LaplaceBEM<dim, spacedim, std::complex<double>, double>::ProblemType::
      DirichletBCProblem,
    is_interior_problem,
    MultithreadInfo::n_threads());
  bem.set_project_name("dirichlet-full-matrix-complex");

  // When the problem type is interior, the source point charge should be placed
  // outside the sphere.
  Point<spacedim> source_loc;

  if (is_interior_problem)
    {
      source_loc = Point<spacedim>(1, 1, 1);
    }
  else
    {
      source_loc = Point<spacedim>(0.25, 0.25, 0.25);
    }

  const Point<spacedim> center(0, 0, 0);
  const double          radius(1);

  Triangulation<spacedim> tria;
  // The manifold_id is set to 0 on the boundary faces in @p hyper_ball.
  GridGenerator::hyper_ball(tria, center, radius);
  tria.refine_global(1);

  Triangulation<dim, spacedim> surface_tria;

  // Create the map from material ids to manifold ids. By default, the material
  // ids of all cells are zero, if the triangulation is created by a deal.ii
  // function in GridGenerator.
  bem.get_manifold_description()[0] = 0;

  // Create the map from manifold ids to manifold objects. Because in the
  // destructor of LaplaceBEM the manifold objects will be released, the
  // manifold object here is created on the heap.
  SphericalManifold<dim, spacedim> *ball_surface_manifold =
    new SphericalManifold<dim, spacedim>(center);
  bem.get_manifolds()[0] = ball_surface_manifold;

  // We should first assign manifold objects to the empty surface triangulation,
  // then perform surface mesh extraction.
  surface_tria.set_manifold(0, *ball_surface_manifold);
  bem.extract_surface_triangulation(tria, std::move(surface_tria), true);

  // Create the map from manifold id to mapping order.
  bem.get_manifold_id_to_mapping_order()[0] = 1;

  // Build surface-to-volume and volume-to-surface relationship.
  bem.get_subdomain_topology().generate_single_domain_topology_for_dealii_model(
    {0});

  DirichletBC dirichlet_bc(source_loc);
  bem.assign_dirichlet_bc(dirichlet_bc);

  if (bem.validate_subdomain_topology())
    {
      bem.run();

      bem.print_memory_consumption_table(deallog.get_file_stream());
    }
  else
    {
      deallog << "Invalid subdomains!" << std::endl;
    }

  ofs.close();
}
