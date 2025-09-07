/**
 * \file dealii-vector-argmax-argmin.cc
 * \brief
 * \ingroup test_cases
 *
 * \author Jihuan Tian
 * \date 2022-03-10
 */

#include <deal.II/lac/vector.h>

#include <algorithm>
#include <forward_list>
#include <iostream>
#include <iterator>
#include <vector>

using namespace dealii;

int
main()
{
  std::vector<int> a{3, 2, 5, 7, 1, 9, 0};
  Vector<int>      v(a.begin(), a.end());

  auto iter1 = std::max_element(v.begin(), v.end());
  std::cout << "Max element index: " << iter1 - v.begin() << std::endl;
  iter1 = std::min_element(v.begin(), v.end());
  std::cout << "Min element index: " << iter1 - v.begin() << std::endl;

  std::forward_list<int> b{3, 2, 5, 7, 1, 9, 0};
  auto                   iter2 = std::max_element(b.begin(), b.end());
  std::cout << "Max element index in forward list: "
            << std::distance(b.begin(), iter2) << std::endl;
  iter2 = std::min_element(b.begin(), b.end());
  std::cout << "Min element index in forward list: "
            << std::distance(b.begin(), iter2) << std::endl;

  return 0;
}
