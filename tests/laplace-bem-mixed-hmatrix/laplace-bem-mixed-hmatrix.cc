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
 * Function object for the Dirichlet boundary condition data.
 */
class DirichletBC : public Function<3>
{
public:
  double
  value(const Point<3> &p, const unsigned int component = 0) const
  {
    (void)component;

    if (p(2) < 3)
      // Apply constant 1 volt on the plane at @p z=0.
      return 1;
    else
      // Apply constant 0 volt on the plane at @p z=6.
      return 0;
  }
};

/**
 * Function object for the Neumann boundary condition data.
 */
class NeumannBC : public Function<3>
{
public:
  // Apply homogeneous Neumann boundary condition on the four sides of the bar.
  double
  value(const Point<3> &p, const unsigned int component = 0) const
  {
    (void)p;
    (void)component;

    return 0;
  }
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

  bem.set_dirichlet_boundary_ids({1, 2});
  bem.set_neumann_boundary_ids({3, 4, 5, 6});

  if (argc > 1)
    {
      bem.read_volume_mesh(std::string(argv[1]));
    }
  else
    {
      bem.read_volume_mesh(std::string("bar-coarse_hex.msh"));
    }

  DirichletBC dirichlet_bc;
  NeumannBC   neumann_bc;

  bem.assign_dirichlet_bc(dirichlet_bc);
  bem.assign_neumann_bc(neumann_bc);

  try
    {
      bem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
