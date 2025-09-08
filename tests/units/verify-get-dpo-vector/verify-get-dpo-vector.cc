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
