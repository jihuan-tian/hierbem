/**
 * \file stl-gen-linear-indices.cc
 * \brief
 * \ingroup testers stl
 * \author Jihuan Tian
 * \date 2022-03-10
 */

#include "debug_tools.h"
#include "generic_functors.h"
#include "unary_template_arg_containers.h"

using namespace std;

int
main()
{
  vector<int> a(10);
  //! The two template argument should be given explicitly.
  gen_linear_indices<vector_uta, int>(a, 0, 2);
  print_vector_values(std::cout, a, " ", false);
  cout << endl;

  forward_list<unsigned int> b(10);
  gen_linear_indices<forward_list_uta, unsigned int>(b, 1, 1);
  print_vector_values(std::cout, b, " ", false);
  cout << endl;

  list<unsigned int> c(10);
  gen_linear_indices<list_uta, unsigned int>(c, 1, 2);
  print_vector_values(std::cout, c, " ", false);
  cout << endl;

  return 0;
}
