// Copyright (C) 2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file verify-dealii-traverse-cell-iterators-in-two-dofhandlers.cc
 * @brief Verify synchronous traversing cell iterators in two DoF handlers on a
 * same triangulation.
 * @details When the two iterators advance synchronously, we expect that the
 * cells they are visiting are a same cell by checking the cell indices. Since
 * the two DoF handlers are created on a same triangulation, the two cell
 * indices must be the same.
 *
 * @ingroup dealii_verify
 * @author Jihuan Tian
 * @date 2025-01-08
 */
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <catch2/catch_all.hpp>

using namespace Catch::Matchers;
using namespace dealii;

TEST_CASE("Traverse cell iterators in two DoF handlers", "[dealii]")
{
  INFO("*** test start");

  Triangulation<2, 3> tria;
  GridGenerator::hyper_sphere(tria);
  tria.refine_global(3);

  FE_Q<2, 3>   fe1(1);
  FE_DGQ<2, 3> fe2(0);

  DoFHandler<2, 3> dof_handler1(tria);
  DoFHandler<2, 3> dof_handler2(tria);

  dof_handler1.distribute_dofs(fe1);
  dof_handler2.distribute_dofs(fe2);

  auto cell_iter1 = dof_handler1.begin_active();
  auto cell_iter2 = dof_handler2.begin_active();

  for (; cell_iter1 != dof_handler1.end(); cell_iter1++, cell_iter2++)
    REQUIRE(cell_iter1->index() == cell_iter2->index());

  INFO("*** test end");
}
