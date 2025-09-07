/**
 * \file get_dofs_per_face_for_fe.cc
 * \brief Verify calculating the number of DoFs per face for a finite element.
 *
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2022-06-14
 */

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>

#include <iostream>

#include "bem/bem_tools.h"
#include "utilities/debug_tools.h"

int
main()
{
  {
    FE_Q<2, 3> fe(2);
    std::cout << fe.get_name() << ":"
              << HierBEM::BEMTools::get_dofs_per_face_for_fe(fe) << std::endl;
  }

  {
    FE_DGQ<2, 3> fe(2);
    std::cout << fe.get_name() << ":"
              << HierBEM::BEMTools::get_dofs_per_face_for_fe(fe) << std::endl;
  }

  {
    FE_DGQ<3, 3> fe(2);
    std::cout << fe.get_name() << ":"
              << HierBEM::BEMTools::get_dofs_per_face_for_fe(fe) << std::endl;
  }

  return 0;
}
