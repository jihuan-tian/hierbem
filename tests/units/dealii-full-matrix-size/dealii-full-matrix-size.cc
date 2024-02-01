/**
 * @file dealii-full-matrix-size.cc
 * @brief
 *
 * @ingroup testers
 * @author
 * @date 2024-01-30
 */

#include <deal.II/base/table.h>

#include <deal.II/lac/full_matrix.h>

#include <iostream>

using namespace dealii;
using namespace std;

int
main()
{
  cout << "sizeof(Table<2, int>): " << sizeof(Table<2, int>) << endl;
  cout << "sizeof(Table<2, double>): " << sizeof(Table<2, double>) << endl;
  cout << "sizeof(FullMatrix<int>): " << sizeof(FullMatrix<int>) << endl;
  cout << "sizeof(FullMatrix<double>): " << sizeof(FullMatrix<double>) << endl;

  Table<2, int>    t1;
  Table<2, double> t2;
  cout << "t1: " << t1.memory_consumption() << endl;
  cout << "t2: " << t2.memory_consumption() << endl;

  FullMatrix<double> m1;
  FullMatrix<double> m2(10, 10);
  cout << "m1: " << m1.memory_consumption() << endl;
  cout << "m2: " << m2.memory_consumption() << endl;

  return 0;
}
