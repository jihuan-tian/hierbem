// Copyright (C) 2023-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file verify-integration-on-sphere-manifold.cc
 * @brief Verify the accuracy of high order mapping used for computing the
 * surface area of a sphere.
 *
 * @ingroup test_cases
 * @author Jihuan Tian
 * @date 2023-09-23
 */

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <fstream>
#include <iomanip>
#include <iostream>

#include "grid/grid_out_ext.h"
#include "linear_algebra/lapack_full_matrix_ext.h"
#include "utilities/debug_tools.h"

using namespace dealii;

int
main(int argc, char *argv[])
{
  deallog.pop();
  deallog.depth_console(3);

  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  /**
   * Generate a unit sphere.
   */
  Triangulation<dim, spacedim> sphere_mesh;
  GridGenerator::hyper_sphere(sphere_mesh);

  std::ofstream mesh_out("sphere.msh");
  HierBEM::write_msh_correct(sphere_mesh, mesh_out);

  /**
   * Define 0-th order finite element space.
   */
  FE_DGQ<dim, spacedim>     fe(0);
  DoFHandler<dim, spacedim> dof_handler;
  dof_handler.initialize(sphere_mesh, fe);

  /**
   * Function to be integrated on the sphere. At the moment, it is simply a
   * constant function.
   */
  Functions::ConstantFunction<spacedim> func(1.0);

  const unsigned int refinement_num    = 5;
  const unsigned int max_mapping_order = 5;

  /**
   * Result matrix: first dimension is mapping order, second dimension is
   * refinement.
   */
  HierBEM::LAPACKFullMatrixExt<double> result_mat(max_mapping_order,
                                                  refinement_num + 1);

  for (unsigned int refinement = 0; refinement <= refinement_num; refinement++)
    {
      if (refinement > 0)
        {
          sphere_mesh.refine_global(1);
          std::ofstream mesh_out(std::string("sphere-refine-") +
                                 std::to_string(refinement) +
                                 std::string(".msh"));
          HierBEM::write_msh_correct(sphere_mesh, mesh_out);
        }

      dof_handler.distribute_dofs(fe);

      for (unsigned int mapping_order = 1; mapping_order <= max_mapping_order;
           mapping_order++)
        {
          double sphere_area = 0.;

          MappingQ<dim, spacedim> mapping(mapping_order);
          QGauss<dim>             quad(std::ceil((mapping_order + 1) / 2.0));
          FEValues<dim, spacedim> fe_values(mapping,
                                            fe,
                                            quad,
                                            UpdateFlags::update_JxW_values |
                                              update_quadrature_points |
                                              update_values);

          for (const auto &cell : sphere_mesh.active_cell_iterators())
            {
              fe_values.reinit(cell);
              for (unsigned int q = 0; q < quad.size(); q++)
                {
                  sphere_area += func.value(fe_values.quadrature_point(q)) *
                                 fe_values.JxW(q);
                }
            }

          result_mat(mapping_order - 1, refinement) = sphere_area;
        }
    }

  result_mat.print_formatted_to_mat(std::cout, "sphere_areas", 16, true, 25);

  return 0;
}
