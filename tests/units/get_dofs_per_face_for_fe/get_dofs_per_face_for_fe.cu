// Copyright (C) 2022-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

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
