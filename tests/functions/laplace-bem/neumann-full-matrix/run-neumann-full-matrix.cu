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

#include <fstream>
#include <iostream>

#include "hbem_test_config.h"
#include "laplace/laplace_bem.h"
#include "preconditioners/preconditioner_type.h"

using namespace dealii;
using namespace HierBEM;

/**
 * Function object for the Neumann boundary condition data, which is also
 * the solution of the Dirichlet problem. The analytical expression is
 * \f[
 * \frac{\pdiff u}{\pdiff n}\Big\vert_{\Gamma} = \frac{\langle x-x_c,x_0-x
 * \rangle}{4\pi\norm{x_0-x}^3\rho}
 * \f]
 */
class NeumannBC : public Function<3>
{
public:
  // N.B. This function should be defined outside class NeumannBC and
  // class Example2, if not inline.
  NeumannBC()
    : Function<3>()
    , x0(0.25, 0.25, 0.25)
    , model_sphere_center(0.0, 0.0, 0.0)
    , model_sphere_radius(1.0)
  {}

  NeumannBC(const Point<3> &x0, const Point<3> &center, double radius)
    : Function<3>()
    , x0(x0)
    , model_sphere_center(center)
    , model_sphere_radius(radius)
  {}

  double
  value(const Point<3> &p, const unsigned int component = 0) const
  {
    (void)component;

    Tensor<1, 3> diff_vector = x0 - p;

    return ((p - model_sphere_center) * diff_vector) / 4.0 / numbers::PI /
           std::pow(diff_vector.norm(), 3) / model_sphere_radius;
  }

private:
  /**
   * Location of the Dirac point source \f$\delta(x-x_0)\f$.
   */
  Point<3> x0;
  Point<3> model_sphere_center;
  double   model_sphere_radius;
};

void
run_neumann_full_matrix()
{
  // Write run-time logs to file
  std::ofstream ofs("neumann-full-matrix.log");
  deallog.pop();
  deallog.depth_console(0);
  deallog.depth_file(5);
  deallog.attach(ofs);

  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  const bool                                is_interior_problem = false;
  LaplaceBEM<dim, spacedim, double, double> bem(
    1,
    0,
    LaplaceBEM<dim, spacedim, double, double>::ProblemType::NeumannBCProblem,
    is_interior_problem,
    MultithreadInfo::n_threads());
  bem.set_project_name("neumann-full-matrix");
  bem.set_preconditioner_type(PreconditionerType::Identity);

  /**
   * @internal Set the Dirac source location according to interior or exterior
   * problem.
   */
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

  NeumannBC neumann_bc(source_loc, center, radius);
  bem.assign_neumann_bc(neumann_bc);

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
