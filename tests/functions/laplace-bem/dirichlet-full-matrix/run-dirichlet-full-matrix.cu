#include <deal.II/base/logstream.h>

#include <fstream>
#include <iostream>

#include "hbem_test_config.h"
#include "laplace_bem.h"

using namespace dealii;
using namespace HierBEM;

/**
 * Function object for the Dirichlet boundary condition data, which is
 * also the solution of the Neumann problem. The analytical expression is:
 * \f[
 * u=\frac{1}{4\pi\norm{x-x_0}}
 * \f]
 */
class DirichletBC : public Function<3>
{
public:
  // N.B. This function should be defined outside class NeumannBC or class
  // Example2, if no inline.
  DirichletBC()
    : Function<3>()
    , x0(0.25, 0.25, 0.25)
  {}

  DirichletBC(const Point<3> &x0)
    : Function<3>()
    , x0(x0)
  {}

  double
  value(const Point<3> &p, const unsigned int component = 0) const
  {
    (void)component;
    return 1.0 / 4.0 / numbers::PI / (p - x0).norm();
  }

private:
  /**
   * Location of the Dirac point source \f$\delta(x-x_0)\f$.
   */
  Point<3> x0;
};

void
run_dirichlet_full_matrix()
{
  // Write run-time logs to file
  std::ofstream ofs("dirichlet-full-matrix.log");
  deallog.pop();
  deallog.depth_console(0);
  deallog.depth_file(5);
  deallog.attach(ofs);

  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  const bool                is_interior_problem = true;
  LaplaceBEM<dim, spacedim> bem(
    1,
    0,
    1,
    1,
    LaplaceBEM<dim, spacedim>::ProblemType::DirichletBCProblem,
    is_interior_problem,
    MultithreadInfo::n_threads());
  bem.set_project_name("dirichlet-full-matrix");

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

  bem.assign_volume_triangulation(std::move(tria), true);

  Triangulation<dim, spacedim>           surface_tria;
  const SphericalManifold<dim, spacedim> ball_surface_manifold(center);
  surface_tria.set_manifold(0, ball_surface_manifold);

  bem.assign_surface_triangulation(std::move(surface_tria), true);

  DirichletBC dirichlet_bc(source_loc);
  bem.assign_dirichlet_bc(dirichlet_bc);

  bem.run();

  bem.print_memory_consumption_table(deallog.get_file_stream());
}
