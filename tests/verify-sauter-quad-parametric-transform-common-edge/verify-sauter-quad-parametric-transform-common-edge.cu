/**
 * @file verify-sauter-quad-parametric-transform-common-edge.cu
 * @brief Verify the Sauter quadrature point distribution for the common edge case.
 *
 * @ingroup testers
 * @author Jihuan Tian
 * @date 2023-03-22
 */

#include <iostream>

#include "debug_tools.hcu"
#include "quadrature.templates.h"
#include "sauter_quadrature_tools.h"

using namespace std;
using namespace dealii;
using namespace HierBEM;

int
main()
{
  const unsigned int dim        = 2;
  const unsigned int quad_order = 2;

  QGauss<dim * 2>    quad(quad_order + 1);
  const unsigned int n_q_points = quad.size();

  print_point_vector_to_mat(
    cout, std::string("qgauss"), quad.get_points(), 15, 25);

  /**
   * Quadrature points in the unit cells of \f$K_x\f$ and \f$K_y\f$
   * respectively.
   */
  std::vector<Point<dim>> kx_unit_quad_points(n_q_points);
  std::vector<Point<dim>> ky_unit_quad_points(n_q_points);
  // Iterate over each $k_3$ part.
  for (unsigned k = 0; k < 6; k++)
    {
      // Iterate over each quadrature point.
      for (unsigned int q = 0; q < n_q_points; q++)
        {
          sauter_common_edge_parametric_coords_to_unit_cells(
            quad.point(q), k, kx_unit_quad_points[q], ky_unit_quad_points[q]);
        }

      print_point_vector_to_mat(cout,
                                std::string("kx_k") + std::to_string(k),
                                kx_unit_quad_points,
                                15,
                                25);
      print_point_vector_to_mat(cout,
                                std::string("ky_k") + std::to_string(k),
                                ky_unit_quad_points,
                                15,

                                25);
    }
}
