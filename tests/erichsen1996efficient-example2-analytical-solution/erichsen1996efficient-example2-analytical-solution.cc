#include <deal.II/base/logstream.h>

#include <erichsen1996efficient_example2.h>

#include <iostream>


int
main()
{
  deallog.depth_console(2);
  deallog.pop();

  IdeoBEM::Erichsen1996Efficient::Example2::AnalyticalSolution
    analytical_solution;

  Point<3> p1(0.1, 0.2, 0.3);
  Point<3> p2(-0.25, 1., -3.);

  deallog << "Analytical solution at (0.1, 0.2, 0.3) is "
          << analytical_solution.value(p1) << std::endl;
  deallog << "Analytical solution at (-0.25, 1., -3.) is "
          << analytical_solution.value(p2) << std::endl;
}
