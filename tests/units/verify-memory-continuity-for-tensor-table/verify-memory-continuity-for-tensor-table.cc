/**
 * @file verify-memory-continuity-for-tensor-table.cc
 * @brief
 *
 * @ingroup test_cases
 * @author Jihuan Tian
 * @date 2023-02-06
 */

#include <deal.II/base/table.h>
#include <deal.II/base/tensor.h>

#include <iostream>

using namespace dealii;
using namespace std;

int
main()
{
  const unsigned int                    spacedim = 3;
  const unsigned int                    rows     = 50;
  const unsigned int                    cols     = 50;
  Table<2, Tensor<1, spacedim, double>> tensor_table(rows, cols);

  // Get the pointer to the first element.
  const Tensor<1, spacedim, double> *start_tensor_ptr = &(tensor_table(0, 0));
  const double                      *start_value_ptr = &(tensor_table(0, 0)[0]);
  // This assertion confirms the starting memory address of the tensor table is
  // the same as the address of its first component.
  Assert((void *)start_tensor_ptr == (void *)start_value_ptr,
         ExcInternalError());

  for (unsigned int i = 0; i < rows; i++)
    {
      for (unsigned int j = 0; j < cols; j++)
        {
          tensor_table(i, j)[0] = 1 + i * 10;
          tensor_table(i, j)[1] = 2 + i * 10;
          tensor_table(i, j)[2] = 3 + i * 10;
        }
    }

  // Get the size for a tensor type, which is @p 8*spacedim. From this we can see
  // that the actual data in a tensor are just all the tensor components,
  // without any other data stored.
  cout << "sizeof(Tensor<1, spacedim, double>)="
       << sizeof(Tensor<1, spacedim, double>) << endl;

  return 0;
}
