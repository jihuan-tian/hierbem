/**
 * \file verify-get-dpo-vector.cc
 * \brief Verify the function @p get_dpo_vector for generating the numbering
 * from lexicographic to hierarchic order.
 *
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2022-07-18
 */

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>

#include <iostream>

#include "mapping/mapping_q_ext.h"
#include "utilities/debug_tools.h"

using namespace dealii;

int
main()
{
  FE_Q<2, 3> fe(4);

  print_vector_values(std::cout,
                      fe.get_poly_space_numbering_inverse(),
                      ",",
                      true);
  print_vector_values(std::cout,
                      FETools::lexicographic_to_hierarchic_numbering<2>(4),
                      ",",
                      true);
}
