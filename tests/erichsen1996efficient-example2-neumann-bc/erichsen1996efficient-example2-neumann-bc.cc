#include <deal.II/base/logstream.h>
#include <erichsen1996efficient_example2.h>

#include <iostream>


int
main()
{
  deallog.depth_console(2);
  deallog.pop();

  IdeoBEM::Erichsen1996Efficient::Example2::NeumannBC neumann_bc;

  double theta = 60. * numbers::PI / 180.;
  double phi   = 75. * numbers::PI / 180.;

  Point<3> p1(std::sin(theta) * std::cos(phi),
              std::sin(theta) * std::sin(phi),
              std::cos(theta));
  theta = 135. * numbers::PI / 180.;
  phi   = 210. * numbers::PI / 180.;
  Point<3> p2(std::sin(theta) * std::cos(phi),
              std::sin(theta) * std::sin(phi),
              std::cos(theta));

  deallog << "Neumann BC at (60, 75) is " << neumann_bc.value(p1) << std::endl;
  deallog << "Neumann BC at (135, 210) is " << neumann_bc.value(p2)
          << std::endl;
}
