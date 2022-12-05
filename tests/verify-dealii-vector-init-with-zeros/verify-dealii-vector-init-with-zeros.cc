/**
 * @file verify-dealii-vector-init-with-zeros.cc
 * @brief Verify if a @p dealii::Vector is assigned with zeros after its
 * construction or reinitialization.
 *
 * @ingroup testers
 * @author Jihuan Tian
 * @date 2022-12-04
 */

#include <deal.II/lac/vector.h>

#include <iostream>

#include "debug_tools.h"

using namespace dealii;

int
main()
{
  const unsigned int N = 10;
  Vector<double>     a(N);

  print_vector_values(std::cout, a, ",", false);
  std::cout << "\n";

  for (unsigned int i = 1; i <= N; i++)
    {
      a(i - 1) = i;
    }

  a.reinit(20);
  print_vector_values(std::cout, a, ",", false);
  std::cout << std::endl;
}
