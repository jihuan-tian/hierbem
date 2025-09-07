/**
 * \file verify-copy-vector.cc
 * \brief Verify copy deal.ii Vector via memcpy
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2022-11-25
 */

#include <deal.II/lac/vector.h>

#include <cstring>
#include <iostream>

using namespace dealii;

template <typename number>
void
copy_vector(Vector<number>                          &dst_vec,
            const typename Vector<number>::size_type dst_start_index,
            const Vector<number>                    &src_vec,
            const typename Vector<number>::size_type src_start_index,
            const typename Vector<number>::size_type number_of_data)
{
  std::memcpy(dst_vec.data() + dst_start_index,
              src_vec.data() + src_start_index,
              number_of_data * sizeof(number));
}

int
main()
{
  Vector<double> a({1, 2, 3, 4, 5, 6});
  Vector<double> a1(2), a2(4);

  // Split the vector.
  copy_vector(a1, 0, a, 0, 2);
  copy_vector(a2, 0, a, 2, 4);

  std::cout << "a1=";
  a1.print(std::cout, 5, false);
  std::cout << "a2=";
  a2.print(std::cout, 5, false);

  // Assemble the vector
  Vector<double> a3({3, 5, 9}), a4({6, 2, 0});
  copy_vector(a, 3, a3, 0, 3);
  copy_vector(a, 0, a4, 0, 3);
  std::cout << "new_a=";
  a.print(std::cout, 5, false);
}
