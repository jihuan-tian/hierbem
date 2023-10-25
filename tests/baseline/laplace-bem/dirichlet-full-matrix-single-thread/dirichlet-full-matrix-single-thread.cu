/**
 * @file test-dirichlet-full-matrix-single-thread.cu
 * @brief Baseline test for solving Laplace problem with Dirichlet boundary
 * condition based on full matrix BEM, which runs in a single thread.
 *
 * @ingroup testers
 * @author Jihuan Tian
 * @date 2023-10-24
 */
#include <catch2/catch_all.hpp>

#include <iostream>

#include "hbem_test_config.h"
#include "laplace_bem.h"

using namespace Catch::Matchers;
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

TEST_CASE(
  "Solve Laplace Dirichlet problem using full matrix in a single thread",
  "[baseline][laplace]")
{
  INFO("*** test start");
  CHECK(1 == 1);

  /**
   * @internal Pop out the default "DEAL" prefix string.
   */
  deallog.pop();
  deallog.depth_console(5);
  LogStream::Prefix prefix_string("HierBEM");

  Timer timer;

  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  LaplaceBEM<dim, spacedim> bem(
    1,
    0,
    1,
    1,
    LaplaceBEM<dim, spacedim>::ProblemType::DirichletBCProblem,
    true,
    MultithreadInfo::n_cores());
  bem.set_cpu_serial(true);

  timer.stop();
  print_wall_time(deallog, timer, "program preparation");

  timer.start();
  bem.read_volume_mesh(HBEM_TEST_MODEL_DIR "sphere-refine-1.msh");
  timer.stop();
  print_wall_time(deallog, timer, "read mesh");

  timer.start();

  const Point<3> source_loc(1, 1, 1);
  const Point<3> center(0, 0, 0);

  DirichletBC dirichlet_bc(source_loc);
  bem.assign_dirichlet_bc(dirichlet_bc);

  timer.stop();
  print_wall_time(deallog, timer, "assign boundary conditions");

  timer.start();

  bem.run();

  timer.stop();
  print_wall_time(deallog, timer, "run the solver");

  deallog << "Program exits with a total wall time " << timer.wall_time() << "s"
          << std::endl;

  double result   = 0.5;
  double expected = 0.5;
  REQUIRE_THAT(result, WithinAbs(expected, 1e-6) || WithinRel(expected, 1e-8));
  INFO("*** test end");
}
