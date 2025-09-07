/**
 * @file verify-memory-continuity-for-point-table.cc
 * @brief
 *
 * @ingroup test_cases
 * @author Jihuan Tian
 * @date 2023-02-07
 */

#include <deal.II/base/point.h>
#include <deal.II/base/table.h>

#include <iostream>

using namespace dealii;
using namespace std;

int
main()
{
  const unsigned int                spacedim = 3;
  const unsigned int                rows     = 50;
  const unsigned int                cols     = 50;
  Table<2, Point<spacedim, double>> point_table(rows, cols);

  // Get the pointer to the first element.
  const Point<spacedim, double> *start_point_ptr = &(point_table(0, 0));
  const double                  *start_value_ptr = &(point_table(0, 0)[0]);
  // This assertion confirms the starting memory address of the tensor table is
  // the same as the address of its first component.
  Assert((void *)start_point_ptr == (void *)start_value_ptr,
         ExcInternalError());

  for (unsigned int i = 0; i < rows; i++)
    {
      for (unsigned int j = 0; j < cols; j++)
        {
          point_table(i, j)[0] = 1 + i * 10;
          point_table(i, j)[1] = 2 + i * 10;
          point_table(i, j)[2] = 3 + i * 10;
        }
    }

  // Get the size for a point type, which is @p 8*spacedim. From this we can see
  // that the actual data in a point are just all the point components,
  // without any other data stored.
  cout << "sizeof(Point<spacedim, double>)=" << sizeof(Point<spacedim, double>)
       << endl;

  return 0;
}
