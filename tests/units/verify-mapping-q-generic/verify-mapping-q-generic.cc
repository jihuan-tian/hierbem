/**
 * \file verify-mapping-q-generic.cc
 * \brief Verify the functions provided by the @p MappingQ class.
 *
 * \ingroup testers dealii_verify
 * \author Jihuan Tian
 * \date 2022-06-30
 */

#include <deal.II/base/quadrature.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <fstream>

#include "debug_tools.h"
#include "mapping/mapping_q_ext.h"

using namespace dealii;

int
main()
{
  const unsigned int dim        = 2;
  const unsigned int spacedim   = 3;
  const unsigned int fe_order   = 2;
  const unsigned int quad_order = fe_order + 1;

  /**
   * Check the degree of the default @p Mapping object associated with a
   * @p FEValues object.
   */
  QGauss<dim>             quad(quad_order);
  FE_Q<dim, spacedim>     fe(fe_order);
  FEValues<dim, spacedim> fe_values(fe,
                                    quad,
                                    update_values | update_JxW_values);

  const MappingQ<dim, spacedim> &mapping =
    dynamic_cast<const MappingQ<dim, spacedim> &>(fe_values.get_mapping());

  std::cout << "Degree of the default mapping object in FEValues: "
            << mapping.get_degree() << std::endl;

  {
    /**
     * Create an independent mapping object.
     */
    MappingQExt<dim, spacedim> mapping(3);

    std::cout << "Degree of the mapping object: " << mapping.get_degree()
              << std::endl;

    /**
     * Create internal data in the mapping. N.B. A dummy quadrature object is
     * passed to the @p get_data function. The @p UpdateFlags is set to
     * @p update_default, which at the moment disables any memory allocation,
     * because this will be manually performed later on.
     */
    std::unique_ptr<typename Mapping<dim, spacedim>::InternalDataBase>
      database = mapping.get_data(update_default, QGauss<dim>(1));
    /**
     * Downcast the pointer of Mapping<dim, spacedim>::InternalDataBase to
     * MappingQ<dim,spacedim>::InternalData.
     */
    std::unique_ptr<typename MappingQ<dim, spacedim>::InternalData> data(
      static_cast<typename MappingQ<dim, spacedim>::InternalData *>(
        database.release()));

    /**
     * Resize the internal data objects.
     */
    data->shape_values.resize(data->n_shape_functions * 1);
    data->shape_derivatives.resize(data->n_shape_functions * 1);

    /**
     * Evaluate mapping shape function at the specified list of unit cell
     * points.
     */
    std::vector<Point<dim>> unit_points;
    unit_points.push_back(Point<dim>(0.3, 0.7));
    data->compute_shape_function_values(unit_points);

    /**
     * Print the calculated shape function values and derivatives.
     */
    std::cout << "=== Shape function values ===\n";
    print_vector_values(std::cout, data->shape_values, ",", true);
    std::cout << "=== Shape function derivatives ===\n";
    for (const auto &d : data->shape_derivatives)
      {
        std::cout << d << "\n";
      }
    std::cout << std::endl;
  }

  {
    /**
     * Create a single cell mesh.
     */
    Triangulation<dim, spacedim> tria;
    GridGenerator::hyper_rectangle(tria,
                                   Point<dim>(1.1, 10.8),
                                   Point<dim>(3.6, 15.2));
    std::ofstream out("one-cell.msh");
    GridOut().write_msh(tria, out);

    /**
     * Create an extended mapping object and output the list of real support
     * points.
     */
    MappingQExt<dim, spacedim> mapping_ext(3);
    for (const auto &e : tria.active_cell_iterators())
      {
        mapping_ext.compute_mapping_support_points(e);
        print_support_point_info(mapping_ext.get_support_points(),
                                 "mapping_q_generic_support_points");
      }

    /**
     * Transform a point in the unit cell to the real cell.
     */
    std::cout << "mapping_ext: "
              << mapping_ext.transform_unit_to_real_cell(Point<dim>(0.3, 0.7))
              << std::endl;

    /**
     * Create a finite element object with the same degree (order) as the
     * mapping and output the list of real support points.
     */
    FE_Q<dim, spacedim>       fe(3);
    MappingQ<dim, spacedim>   mapping(3);
    DoFHandler<dim, spacedim> dof_handler(tria);
    dof_handler.distribute_dofs(fe);
    print_support_point_info(fe, mapping, dof_handler, "fe_q_support_points");

    /**
     * Transform a point in the unit cell to the real cell.
     */
    std::cout << "mapping: "
              << mapping.transform_unit_to_real_cell(tria.begin_active(),
                                                     Point<dim>(0.3, 0.7))
              << std::endl;
  }

  return 0;
}
