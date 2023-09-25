/**
 * @file verify-dealii-vector-memory.cc
 * @brief Verify the internal memory of @p dealii::Vector.
 *
 * @ingroup testers
 * @author Jihuan Tian
 * @date 2022-12-04
 */

#include <deal.II/lac/vector.h>

#include <iostream>

using namespace dealii;

int
main()
{
  const unsigned int N = 10;
  Vector<double>     a(N);

  for (unsigned int i = 1; i <= N; i++)
    {
      a(i - 1) = i;
    }

  // Direct memory access of the vector.
  typename dealii::Vector<double>::pointer p = a.data();
  for (unsigned int i = 0; i < N; i++)
    {
      std::cout << *(p + i) << " ";
    }
  std::cout << std::endl;

  // Check the memory consumption.
  std::cout << "Memory consumption: " << a.memory_consumption() << std::endl;
  a.reinit(200);
  std::cout << "Memory consumption after reinit: " << a.memory_consumption()
            << std::endl;

  return 0;
}
