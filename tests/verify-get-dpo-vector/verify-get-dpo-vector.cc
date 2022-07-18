/**
 * \file verify-get-dpo-vector.cc
 * \brief Verify the function @p get_dpo_vector for generating the numbering
 * from lexicographic to hierarchic order.
 *
 * \ingroup testers
 * \author Jihuan Tian
 * \date 2022-07-18
 */

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>

#include <iostream>

#include "debug_tools.h"
#include "mapping_q_generic_ext.h"

using namespace dealii;

int
main()
{
  FE_Q<2, 3> fe(4);

  print_vector_values(std::cout,
                      FETools::lexicographic_to_hierarchic_numbering(fe),
                      ",",
                      true);
  print_vector_values(
    std::cout,
    FETools::lexicographic_to_hierarchic_numbering(FiniteElementData<2>(
      ::internal::MappingQGenericImplementation::get_dpo_vector<2>(4), 1, 4)),
    ",",
    true);
}
