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
 * \file verify-jxw-on-fe_q-fe_dgq.cc
 * \brief Compare the values of @p JxW obtained from @p FEValues on each
 * quadrature point by using two finite elements @p FE_Q and @p FE_DGQ.
 *
 * \ingroup test_cases dealii_verify
 * \author Jihuan Tian
 * \date 2022-05-27
 */

#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_in.h>

#include <fstream>
#include <iostream>

using namespace dealii;

int
main()
{
  deallog.pop();
  deallog.depth_console(2);

  /**
   * Generate a single-cell mesh.
   */
  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  Triangulation<dim, spacedim> triangulation;

  GridIn<dim, spacedim> grid_in;
  grid_in.attach_triangulation(triangulation);
  std::fstream mesh_file("sphere-from-gmsh_hex.msh");
  grid_in.read_msh(mesh_file);

  DoFHandler<dim, spacedim> dof_handler_fe_q;
  DoFHandler<dim, spacedim> dof_handler_fe_dgq;

  const unsigned int fe_q_order   = 3;
  const unsigned int fe_dgq_order = 0;

  FE_Q<dim, spacedim>   fe_q(fe_q_order);
  FE_DGQ<dim, spacedim> fe_dgq(fe_dgq_order);

  dof_handler_fe_q.initialize(triangulation, fe_q);
  dof_handler_fe_dgq.initialize(triangulation, fe_dgq);

  /**
   * Iterate over each active cell.
   */
  typename DoFHandler<dim, spacedim>::cell_iterator fe_q_iter =
    dof_handler_fe_q.active_cell_iterators().begin();
  typename DoFHandler<dim, spacedim>::cell_iterator fe_dgq_iter =
    dof_handler_fe_dgq.active_cell_iterators().begin();

  /**
   * Declare the Gauss quadrature object.
   */
  QGauss<dim>        quad(fe_q_order + 1);
  const unsigned int n_q_points = quad.size();

  /**
   * Declare the @p FEValues objects which are used for calculating cell-wise
   * data. When iterating to a new cell, the @p FEValues objects should be
   * reinitialized.
   */
  FEValues<dim, spacedim> fe_q_values(fe_q, quad, update_JxW_values);
  FEValues<dim, spacedim> fe_dgq_values(fe_dgq, quad, update_JxW_values);

  deallog << "# JxW of fe_q and fe_dgq on each quad point in each cell"
          << std::endl;

  for (; fe_q_iter != dof_handler_fe_q.active_cell_iterators().end();
       fe_q_iter++, fe_dgq_iter++)
    {
      fe_q_values.reinit(fe_q_iter);
      fe_dgq_values.reinit(fe_dgq_iter);

      for (unsigned int q = 0; q < n_q_points; q++)
        {
          deallog << fe_q_values.JxW(q) << "," << fe_dgq_values.JxW(q) << ","
                  << fe_q_values.JxW(q) - fe_dgq_values.JxW(q) << std::endl;
        }
    }


  return 0;
}
