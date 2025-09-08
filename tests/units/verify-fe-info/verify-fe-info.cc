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
