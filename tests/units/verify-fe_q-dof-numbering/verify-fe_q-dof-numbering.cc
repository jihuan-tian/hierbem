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
 * @file verify-fe-poly-dof-numbering.cc
 * @brief Verify different orderings of dof support points related to finite
 * element polynomials.
 *
 * @date 2022-02-14
 * @author Jihuan Tian
 */

// Log handling
#include <deal.II/base/logstream.h>
// H1-conforming finite element
#include <deal.II/fe/fe_q.h>
// L2-conforming finite element
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_tools.h>

// Linear algebra related
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <fstream>

#include "utilities/debug_tools.h"

using namespace dealii;

int
main()
{
  deallog.pop();
  deallog.depth_console(2);

  const unsigned int    spacedim = 3;
  const unsigned int    dim      = 2;
  const unsigned int    fe_order = 3;
  FE_Q<dim, spacedim>   fe_q(fe_order);
  FE_DGQ<dim, spacedim> fe_dgq(fe_order);
  FE_DGQ<dim, spacedim> fe_dgq0(0);

  print_polynomial_space_numbering(std::cout, fe_q, "fe_q");
  print_mapping_between_lexicographic_and_hierarchic_numberings(std::cout,
                                                                fe_q,
                                                                "fe_q");

  // N.B. The default DoF ordering for the discontinuous finite element
  // @p FE_DGQ has already been lexicographic. Therefore, the numbering
  // returned from @p print_polynomial_space_numbering is just [0, 1, ...].
  // However, @p FETools::hierarchic_to_lexicographic_numbering disregards this.
  print_polynomial_space_numbering(std::cout, fe_dgq, "fe_dgq");
  print_mapping_between_lexicographic_and_hierarchic_numberings(std::cout,
                                                                fe_dgq,
                                                                "fe_dgq");

  print_polynomial_space_numbering(std::cout, fe_dgq0, "fe_dgq0");
  print_mapping_between_lexicographic_and_hierarchic_numberings(std::cout,
                                                                fe_dgq0,
                                                                "fe_dgq0");
}
