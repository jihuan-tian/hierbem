/**
 * @file sauter_quadrature.h
 * @brief Introduction of sauter_quadrature.h
 *
 * @date 2022-03-02
 * @author Jihuan Tian
 */
#ifndef INCLUDE_SAUTER_QUADRATURE_H_
#define INCLUDE_SAUTER_QUADRATURE_H_


#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_base.h>

#include <deal.II/grid/tria_accessor.h>

#include <vector>

using namespace dealii;

/**
 * Build the topology for "DoF support point to cell" relation.
 *
 * @param dof_to_cell_topo
 * @param dof_handler
 * @param fe_index
 */
template <int dim, int spacedim>
void
build_dof_to_cell_topology(
  std::vector<std::vector<unsigned int>> &dof_to_cell_topo,
  const DoFHandler<dim, spacedim> &       dof_handler,
  const unsigned int                      fe_index = 0)
{
  const types::global_dof_index        n_dofs = dof_handler.n_dofs();
  const FiniteElement<dim, spacedim> & fe     = dof_handler.get_fe(fe_index);
  const unsigned int                   dofs_per_cell = fe.dofs_per_cell;
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  dof_to_cell_topo.resize(n_dofs);

  /**
   * Iterate over each active cell in the triangulation and extract the DoF
   * indices.
   */
  for (const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell :
       dof_handler.active_cell_iterators())
    {
      cell->get_dof_indices(local_dof_indices);
      for (auto dof_index : local_dof_indices)
        {
          dof_to_cell_topo[dof_index].push_back(cell->active_cell_index());
        }
    }
}


/**
 * Print out the topological information about DoF support point to cell
 * relation.
 *
 * @param dof_to_cell_topo
 */
void
print_dof_to_cell_topology(
  const std::vector<std::vector<unsigned int>> &dof_to_cell_topo);


/**
 * Transform parametric coordinates in Sauter's quadrature rule for the same
 * panel case to unit cell coordinates for \f$K_x\f$ and \f$K_y\f$ respectively.
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
  Point<dim> &          kx_unit_cell_coords,
  Point<dim> &          ky_unit_cell_coords)
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
  Point<dim> &          kx_unit_cell_coords,
  Point<dim> &          ky_unit_cell_coords)
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
    }
}


/**
 * Transform parametric coordinates in Sauter's quadrature rule for the
 * common vertex case to unit cell coordinates for \f$K_x\f$ and \f$K_y\f$
 * respectively.
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
  Point<dim> &          kx_unit_cell_coords,
  Point<dim> &          ky_unit_cell_coords)
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
    }
}


/**
 * Transform parametric coordinates in Sauter's quadrature rule for the
 * regular case to unit cell coordinates for \f$K_x\f$ and \f$K_y\f$ respectively.
 *
 * @param parametric_coords Parameter coordinates in \f$\mathbb{R}^{dim*2}\f$
 * @param kx_unit_cell_coords [out]
 * @param ky_unit_cell_coords [out]
 */
template <int dim>
void
sauter_regular_parametric_coords_to_unit_cells(
  const Point<dim * 2> &parametric_coords,
  Point<dim> &          kx_unit_cell_coords,
  Point<dim> &          ky_unit_cell_coords)
{
  kx_unit_cell_coords(0) = parametric_coords(0);
  kx_unit_cell_coords(1) = parametric_coords(1);
  ky_unit_cell_coords(0) = parametric_coords(2);
  ky_unit_cell_coords(1) = parametric_coords(3);
}

#endif /* INCLUDE_SAUTER_QUADRATURE_H_ */
