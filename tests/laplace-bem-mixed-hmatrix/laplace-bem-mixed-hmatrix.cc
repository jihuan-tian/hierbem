/**
 * \file laplace-bem-mixed-hmatrix.cc
 * \brief Verify solve Laplace mixed boundary value problem using \hmat.
 * \ingroup testers
 * \author Jihuan Tian
 * \date 2022-11-21
 */

#include <iostream>

#include "laplace_bem.h"

using namespace dealii;
using namespace IdeoBEM;

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

int
main(int argc, char *argv[])
{
  deallog.depth_console(2);

  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  LaplaceBEM<dim, spacedim> bem(
    1, // fe order for dirichlet space
    0, // fe order for neumann space
    1, // mapping order for dirichlet domain
    1, // mapping order for neumann domain
    LaplaceBEM<dim, spacedim>::ProblemType::MixedBCProblem,
    true, // is interior problem
    4,    // n_min for cluster tree
    10,   // n_min for block cluster tree
    0.8,  // eta for H-matrix
    5,    // max rank for H-matrix
    0.01, // aca epsilon for H-matrix
    1.0,  // eta for preconditioner
    2,    // max rank for preconditioner
    0.1,  // aca epsilon for preconditioner
    MultithreadInfo::n_cores());

  bem.set_dirichlet_boundary_ids({1});
  bem.set_neumann_boundary_ids({2});

  if (argc > 1)
    {
      bem.read_volume_mesh(std::string(argv[1]));
    }
  else
    {
      bem.read_volume_mesh(std::string("sphere-split-into-halves_hex.msh"));
    }

  bem.setup_system();
  bem.assemble_hmatrix_system();

  //  const Point<3> source_loc(1, 1, 1);
  //  const Point<3> center(0, 0, 0);
  //  const double   radius(1);
  //
  //  DirichletBC dirichlet_bc(source_loc);
  //  NeumannBC   neumann_bc(source_loc, center, radius);
  //
  //  bem.assign_dirichlet_bc(dirichlet_bc);
  //  bem.assign_neumann_bc(neumann_bc);
  //
  //  bem.run();

  return 0;
}
