#include <deal.II/base/logstream.h>
#include <deal.II/base/point.h>

#include <algorithm>
#include <vector>

#include "quadrature.templates.h"

using namespace dealii;

int
main()
{
  deallog.depth_console(2);
  deallog.pop();

  const unsigned int dim = 4;
  const unsigned int n   = 5;

  QGauss<dim> quad_rule(n);

  const std::vector<Point<dim>> &qpts = quad_rule.get_points();
  const std::vector<double> &    qwts = quad_rule.get_weights();

  deallog << n << "th order quadrature points:\n";
  deallog << "Number of points: " << qpts.size() << "\n";
  for (auto point : qpts)
    {
      // Make sure all point coordinates are in the range [0, 1].
      Assert(point(0) >= 0 && point(0) <= 1, ExcInternalError());
      Assert(point(1) >= 0 && point(1) <= 1, ExcInternalError());
      Assert(point(2) >= 0 && point(2) <= 1, ExcInternalError());
      Assert(point(3) >= 0 && point(3) <= 1, ExcInternalError());

      deallog << point << "\n";
    }

  deallog << n << "th order quadrature weights:\n";
  deallog << "Number of weights: " << qwts.size() << "\n";
  double sum_of_weights = 0.;
  for (auto weight : qwts)
    {
      deallog << weight << "\n";
      sum_of_weights += weight;
    }
  deallog << "Sum of all weights: " << sum_of_weights << std::endl;

  return 0;
}
