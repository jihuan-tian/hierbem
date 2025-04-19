/**
 * @file sauter_quadrature_tools.h
 * @brief Introduction of sauter_quadrature_tools.h
 *
 * @date 2022-03-05
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_SAUTER_QUADRATURE_TOOLS_H_
#define HIERBEM_INCLUDE_SAUTER_QUADRATURE_TOOLS_H_

#include <deal.II/base/point.h>

#include "config.h"
#include "quadrature.templates.h"

HBEM_NS_OPEN

using namespace dealii;

/**
 * Class for Sauter's quadrature rule, which is a collection of four
 * @p QGauss rules in 4D space.
 */
template <int dim>
struct SauterQuadratureRule
{
  SauterQuadratureRule();

  SauterQuadratureRule(const unsigned int same_panel_order,
                       const unsigned int common_edge_order,
                       const unsigned int common_vertex_order,
                       const unsigned int regular_order);

  SauterQuadratureRule(const SauterQuadratureRule &sauter_quad_rule);

  unsigned int quad_order_for_same_panel;
  unsigned int quad_order_for_common_edge;
  unsigned int quad_order_for_common_vertex;
  unsigned int quad_order_for_regular;

  QGauss<dim * 2> quad_rule_for_same_panel;
  QGauss<dim * 2> quad_rule_for_common_edge;
  QGauss<dim * 2> quad_rule_for_common_vertex;
  QGauss<dim * 2> quad_rule_for_regular;
};

template <int dim>
SauterQuadratureRule<dim>::SauterQuadratureRule()
  : quad_order_for_same_panel(5)
  , quad_order_for_common_edge(4)
  , quad_order_for_common_vertex(4)
  , quad_order_for_regular(3)
  , quad_rule_for_same_panel(quad_order_for_same_panel)
  , quad_rule_for_common_edge(quad_order_for_common_edge)
  , quad_rule_for_common_vertex(quad_order_for_common_vertex)
  , quad_rule_for_regular(quad_order_for_regular)
{}

template <int dim>
SauterQuadratureRule<dim>::SauterQuadratureRule(
  const unsigned int same_panel_order,
  const unsigned int common_edge_order,
  const unsigned int common_vertex_order,
  const unsigned int regular_order)
  : quad_order_for_same_panel(same_panel_order)
  , quad_order_for_common_edge(common_edge_order)
  , quad_order_for_common_vertex(common_vertex_order)
  , quad_order_for_regular(regular_order)
  , quad_rule_for_same_panel(quad_order_for_same_panel)
  , quad_rule_for_common_edge(quad_order_for_common_edge)
  , quad_rule_for_common_vertex(quad_order_for_common_vertex)
  , quad_rule_for_regular(quad_order_for_regular)
{}

template <int dim>
SauterQuadratureRule<dim>::SauterQuadratureRule(
  const SauterQuadratureRule<dim> &sauter_quad_rule)
  : quad_order_for_same_panel(sauter_quad_rule.quad_order_for_same_panel)
  , quad_order_for_common_edge(sauter_quad_rule.quad_order_for_common_edge)
  , quad_order_for_common_vertex(sauter_quad_rule.quad_order_for_common_vertex)
  , quad_order_for_regular(sauter_quad_rule.quad_order_for_regular)
  , quad_rule_for_same_panel(sauter_quad_rule.quad_rule_for_same_panel)
  , quad_rule_for_common_edge(sauter_quad_rule.quad_rule_for_common_edge)
  , quad_rule_for_common_vertex(sauter_quad_rule.quad_rule_for_common_vertex)
  , quad_rule_for_regular(sauter_quad_rule.quad_rule_for_regular)
{}

/**
 * Transform parametric coordinates in Sauter's quadrature rule for the same
 * panel case to unit cell coordinates for \f$K_x\f$ and \f$K_y\f$
 * respectively.
 *
 * @param parametric_coords Parameter coordinates in \f$\mathbb{R}^{dim*2}\f$
 * @param k3_index
 * @param kx_unit_cell_coords [out]
 * @param ky_unit_cell_coords [out]
 */
template <int dim>
void
sauter_same_panel_parametric_coords_to_unit_cells(
  const Point<dim * 2> &parametric_coords,
  const unsigned int    k3_index,
  Point<dim>           &kx_unit_cell_coords,
  Point<dim>           &ky_unit_cell_coords)
{
  double unit_coords[4] = {
    (1 - parametric_coords(0)) * parametric_coords(3),
    (1 - parametric_coords(0) * parametric_coords(1)) * parametric_coords(2),
    parametric_coords(0) + (1 - parametric_coords(0)) * parametric_coords(3),
    parametric_coords(0) * parametric_coords(1) +
      (1 - parametric_coords(0) * parametric_coords(1)) * parametric_coords(2)};

  switch (k3_index)
    {
      case 0:
        kx_unit_cell_coords(0) = unit_coords[0];
        kx_unit_cell_coords(1) = unit_coords[1];
        ky_unit_cell_coords(0) = unit_coords[2];
        ky_unit_cell_coords(1) = unit_coords[3];

        break;
      case 1:
        kx_unit_cell_coords(0) = unit_coords[1];
        kx_unit_cell_coords(1) = unit_coords[0];
        ky_unit_cell_coords(0) = unit_coords[3];
        ky_unit_cell_coords(1) = unit_coords[2];

        break;
      case 2:
        kx_unit_cell_coords(0) = unit_coords[0];
        kx_unit_cell_coords(1) = unit_coords[3];
        ky_unit_cell_coords(0) = unit_coords[2];
        ky_unit_cell_coords(1) = unit_coords[1];

        break;
      case 3:
        kx_unit_cell_coords(0) = unit_coords[1];
        kx_unit_cell_coords(1) = unit_coords[2];
        ky_unit_cell_coords(0) = unit_coords[3];
        ky_unit_cell_coords(1) = unit_coords[0];

        break;
      case 4:
        kx_unit_cell_coords(0) = unit_coords[2];
        kx_unit_cell_coords(1) = unit_coords[1];
        ky_unit_cell_coords(0) = unit_coords[0];
        ky_unit_cell_coords(1) = unit_coords[3];

        break;
      case 5:
        kx_unit_cell_coords(0) = unit_coords[3];
        kx_unit_cell_coords(1) = unit_coords[0];
        ky_unit_cell_coords(0) = unit_coords[1];
        ky_unit_cell_coords(1) = unit_coords[2];

        break;
      case 6:
        kx_unit_cell_coords(0) = unit_coords[2];
        kx_unit_cell_coords(1) = unit_coords[3];
        ky_unit_cell_coords(0) = unit_coords[0];
        ky_unit_cell_coords(1) = unit_coords[1];

        break;
      case 7:
        kx_unit_cell_coords(0) = unit_coords[3];
        kx_unit_cell_coords(1) = unit_coords[2];
        ky_unit_cell_coords(0) = unit_coords[1];
        ky_unit_cell_coords(1) = unit_coords[0];

        break;
      default:
        Assert(false, ExcInternalError());
        break;
    }
}


/**
 * Transform parametric coordinates in Sauter's quadrature rule for the
 * common edge case to unit cell coordinates for \f$K_x\f$ and \f$K_y\f$
 * respectively.
 *
 * @param parametric_coords Parameter coordinates in \f$\mathbb{R}^{dim*2}\f$
 * @param k3_index
 * @param kx_unit_cell_coords [out]
 * @param ky_unit_cell_coords [out]
 */
template <int dim>
void
sauter_common_edge_parametric_coords_to_unit_cells(
  const Point<dim * 2> &parametric_coords,
  const unsigned int    k3_index,
  Point<dim>           &kx_unit_cell_coords,
  Point<dim>           &ky_unit_cell_coords)
{
  double unit_coords1[4] = {(1 - parametric_coords(0)) * parametric_coords(3) +
                              parametric_coords(0),
                            parametric_coords(0) * parametric_coords(2),
                            (1 - parametric_coords(0)) * parametric_coords(3),
                            parametric_coords(0) * parametric_coords(1)};
  double unit_coords2[4] = {(1 - parametric_coords(0) * parametric_coords(1)) *
                                parametric_coords(3) +
                              parametric_coords(0) * parametric_coords(1),
                            parametric_coords(0) * parametric_coords(2),
                            (1 - parametric_coords(0) * parametric_coords(1)) *
                              parametric_coords(3),
                            parametric_coords(0)};

  switch (k3_index)
    {
      case 0:
        kx_unit_cell_coords(0) = unit_coords1[0];
        kx_unit_cell_coords(1) = unit_coords1[1];
        ky_unit_cell_coords(0) = unit_coords1[2];
        ky_unit_cell_coords(1) = unit_coords1[3];

        break;
      case 1:
        kx_unit_cell_coords(0) = unit_coords1[2];
        kx_unit_cell_coords(1) = unit_coords1[1];
        ky_unit_cell_coords(0) = unit_coords1[0];
        ky_unit_cell_coords(1) = unit_coords1[3];

        break;
      case 2:
        kx_unit_cell_coords(0) = unit_coords2[0];
        kx_unit_cell_coords(1) = unit_coords2[1];
        ky_unit_cell_coords(0) = unit_coords2[2];
        ky_unit_cell_coords(1) = unit_coords2[3];

        break;
      case 3:
        kx_unit_cell_coords(0) = unit_coords2[0];
        kx_unit_cell_coords(1) = unit_coords2[3];
        ky_unit_cell_coords(0) = unit_coords2[2];
        ky_unit_cell_coords(1) = unit_coords2[1];

        break;
      case 4:
        kx_unit_cell_coords(0) = unit_coords2[2];
        kx_unit_cell_coords(1) = unit_coords2[1];
        ky_unit_cell_coords(0) = unit_coords2[0];
        ky_unit_cell_coords(1) = unit_coords2[3];

        break;
      case 5:
        kx_unit_cell_coords(0) = unit_coords2[2];
        kx_unit_cell_coords(1) = unit_coords2[3];
        ky_unit_cell_coords(0) = unit_coords2[0];
        ky_unit_cell_coords(1) = unit_coords2[1];

        break;
      default:
        Assert(false, ExcInternalError());
        break;
    }
}


/**
 * Transform parametric coordinates in Sauter's quadrature rule for the
 * common vertex case to unit cell coordinates for \f$K_x\f$ and \f$K_y\f$
 * respectively.
 *
 * \myalert{This function directly operates on the quadrature points obtained
 * from a quadrature object, which have double type. Therefore, we do not make
 * the number type as a template parameter in this function.}
 *
 * @param parametric_coords Parameter coordinates in \f$\mathbb{R}^{dim*2}\f$
 * @param k3_index
 * @param kx_unit_cell_coords [out]
 * @param ky_unit_cell_coords [out]
 */
template <int dim>
void
sauter_common_vertex_parametric_coords_to_unit_cells(
  const Point<dim * 2> &parametric_coords,
  const unsigned int    k3_index,
  Point<dim>           &kx_unit_cell_coords,
  Point<dim>           &ky_unit_cell_coords)
{
  double unit_coords[4] = {parametric_coords(0),
                           parametric_coords(0) * parametric_coords(1),
                           parametric_coords(0) * parametric_coords(2),
                           parametric_coords(0) * parametric_coords(3)};

  switch (k3_index)
    {
      case 0:
        kx_unit_cell_coords(0) = unit_coords[0];
        kx_unit_cell_coords(1) = unit_coords[1];
        ky_unit_cell_coords(0) = unit_coords[2];
        ky_unit_cell_coords(1) = unit_coords[3];

        break;
      case 1:
        kx_unit_cell_coords(0) = unit_coords[1];
        kx_unit_cell_coords(1) = unit_coords[0];
        ky_unit_cell_coords(0) = unit_coords[2];
        ky_unit_cell_coords(1) = unit_coords[3];

        break;
      case 2:
        kx_unit_cell_coords(0) = unit_coords[1];
        kx_unit_cell_coords(1) = unit_coords[2];
        ky_unit_cell_coords(0) = unit_coords[0];
        ky_unit_cell_coords(1) = unit_coords[3];

        break;
      case 3:
        kx_unit_cell_coords(0) = unit_coords[1];
        kx_unit_cell_coords(1) = unit_coords[2];
        ky_unit_cell_coords(0) = unit_coords[3];
        ky_unit_cell_coords(1) = unit_coords[0];

        break;
      default:
        Assert(false, ExcInternalError());
        break;
    }
}


/**
 * Transform parametric coordinates in Sauter's quadrature rule for the
 * regular case to unit cell coordinates for \f$K_x\f$ and \f$K_y\f$
 * respectively.
 *
 * @param parametric_coords Parameter coordinates in \f$\mathbb{R}^{dim*2}\f$
 * @param kx_unit_cell_coords [out]
 * @param ky_unit_cell_coords [out]
 */
template <int dim>
void
sauter_regular_parametric_coords_to_unit_cells(
  const Point<dim * 2> &parametric_coords,
  Point<dim>           &kx_unit_cell_coords,
  Point<dim>           &ky_unit_cell_coords)
{
  kx_unit_cell_coords(0) = parametric_coords(0);
  kx_unit_cell_coords(1) = parametric_coords(1);
  ky_unit_cell_coords(0) = parametric_coords(2);
  ky_unit_cell_coords(1) = parametric_coords(3);
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_SAUTER_QUADRATURE_TOOLS_H_
