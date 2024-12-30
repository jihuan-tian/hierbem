/**
 * @file verify-integration-on-cylinder-hull.cu
 * @brief Verify the accuracy of manifold mapping used for computing the surface area of a cylinder hull.
 * Compare with the results using different orders of @p MappingQ.
 *
 * @ingroup testers
 * @author Jihuan Tian
 * @date 2023-09-25
 */

#include <deal.II/base/logstream.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_manifold.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>

#include "debug_tools.h"
#include "grid_out_ext.h"
#include "lapack_full_matrix_ext.h"

using namespace dealii;

int
main(int /* argc */, char * /* argv */[])
{
  deallog.pop();
  deallog.depth_console(3);

  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  /**
   * Generate a cylinder with volume mesh.
   */
  Triangulation<spacedim> volume_mesh;
  GridGenerator::cylinder(volume_mesh);

  /**
   * Extract surface mesh of the cylinder hull, which has the 0 boundary id.
   */
  Triangulation<dim, spacedim> surface_mesh;
  std::set<types::boundary_id> boundary_id_set_for_hull{0};
  GridGenerator::extract_boundary_mesh(volume_mesh,
                                       surface_mesh,
                                       boundary_id_set_for_hull);
  /**
   * Reset manifold for the surface mesh, since this information will not be
   * preserved during the extraction.
   */
  surface_mesh.set_all_manifold_ids(0);
  surface_mesh.set_manifold(0, CylindricalManifold<dim, spacedim>());

  std::ofstream mesh_out("hull.msh");
  HierBEM::write_msh_correct(surface_mesh, mesh_out);

  /**
   * Define 0-th order finite element space.
   */
  FE_DGQ<dim, spacedim>     fe(0);
  DoFHandler<dim, spacedim> dof_handler;
  // dof_handler.initialize(surface_mesh, fe);
  dof_handler.reinit(surface_mesh);
  dof_handler.distribute_dofs(fe);

  /**
   * Function to be integrated on the cylinder hull. At the moment, it is simply
   * a constant function.
   */
  Functions::ConstantFunction<spacedim> func(1.0);

  const unsigned int refinement_num    = 5;
  const unsigned int max_mapping_order = 5;
  const unsigned int max_quad_order    = 3;

  /**
   * Result matrix for the high order mapping: the first dimension is mapping
   * order, the second dimension is refinement.
   */
  HierBEM::LAPACKFullMatrixExt<double> result_mat_mappingq(max_mapping_order,
                                                           refinement_num + 1);
  /**
   * Result matrix for the mapping derived from manifold: the first dimension is
   * quadrature order, the second dimension is refinement.
   */
  HierBEM::LAPACKFullMatrixExt<double> result_mat_mapping_manifold(
    max_quad_order, refinement_num + 1);

  for (unsigned int refinement = 0; refinement <= refinement_num; refinement++)
    {
      if (refinement > 0)
        {
          surface_mesh.refine_global(1);

          std::ofstream mesh_out(std::string("hull-refine-") +
                                 std::to_string(refinement) +
                                 std::string(".msh"));
          HierBEM::write_msh_correct(surface_mesh, mesh_out);
        }

      dof_handler.distribute_dofs(fe);

      /**
       * Compute the surface integral using different orders of @p MappingQ.
       */
      for (unsigned int mapping_order = 1; mapping_order <= max_mapping_order;
           mapping_order++)
        {
          double cylinder_hull_area = 0.;

          MappingQ<dim, spacedim> mapping(mapping_order);
          QGauss<dim>             quad(std::ceil((mapping_order + 1) / 2.0));
          FEValues<dim, spacedim> fe_values(mapping,
                                            fe,
                                            quad,
                                            UpdateFlags::update_JxW_values |
                                              update_quadrature_points |
                                              update_values);

          for (const auto &cell : surface_mesh.active_cell_iterators())
            {
              fe_values.reinit(cell);
              for (unsigned int q = 0; q < quad.size(); q++)
                {
                  cylinder_hull_area +=
                    func.value(fe_values.quadrature_point(q)) *
                    fe_values.JxW(q);
                }
            }

          result_mat_mappingq(mapping_order - 1, refinement) =
            cylinder_hull_area;
        }

      /**
       * Compute the surface integral using @p MappingManifold under different
       * quadrature point number in a single direction.
       */
      for (unsigned int quad_num = 1; quad_num <= max_quad_order; quad_num++)
        {
          double cylinder_hull_area = 0.;

          MappingManifold<dim, spacedim> mapping;
          QGauss<dim>                    quad(quad_num);
          FEValues<dim, spacedim>        fe_values(mapping,
                                            fe,
                                            quad,
                                            UpdateFlags::update_JxW_values |
                                              update_quadrature_points |
                                              update_values);

          for (const auto &cell : surface_mesh.active_cell_iterators())
            {
              fe_values.reinit(cell);
              for (unsigned int q = 0; q < quad.size(); q++)
                {
                  cylinder_hull_area +=
                    func.value(fe_values.quadrature_point(q)) *
                    fe_values.JxW(q);
                }
            }

          result_mat_mapping_manifold(quad_num - 1, refinement) =
            cylinder_hull_area;
        }
    }

  result_mat_mappingq.print_formatted_to_mat(
    std::cout, "cylinder_areas_mappingq", 16, true, 25);
  result_mat_mapping_manifold.print_formatted_to_mat(
    std::cout, "cylinder_areas_mapping_manifold", 16, true, 25);

  return 0;
}
