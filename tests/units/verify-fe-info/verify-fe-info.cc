/**
 * \file verify-fe-info.cc
 * \brief Verify the information for typical finite elements.
 *
 * \ingroup test_cases dealii_verify
 * \author Jihuan Tian
 * \date 2022-06-11
 */

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>

#include <iostream>

#include "utilities/debug_tools.h"

using namespace dealii;

int
main()
{
  FE_Q<2, 3> fe_q(2);
  print_fe_info(std::cout, fe_q);

  FE_DGQ<2, 3> fe_dgq(2);
  print_fe_info(std::cout, fe_dgq);
}
