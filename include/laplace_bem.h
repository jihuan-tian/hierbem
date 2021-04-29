/** \file laplace_bem.h
 * \brief Implementation of BEM involving kernel functions and singular
 * numerical quadratures.
 * \ingroup sauter_quadrature
 * \date 2020-11-02
 * \author Jihuan Tian
 */

#ifndef INCLUDE_LAPLACE_BEM_H_
#define INCLUDE_LAPLACE_BEM_H_

/**
 * \ingroup sauter_quadrature
 * @{
 */

#include <deal.II/base/point.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/base/table.h>
#include <deal.II/base/table_indices.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_base.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/lac/full_matrix.templates.h>

// Triangulation
#include <deal.II/grid/tria.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "quadrature.templates.h"

using namespace dealii;

namespace LaplaceBEM
{
  // Declaration of defined classes.
  template <int dim, int spacedim, typename RangeNumberType = double>
  class BEMValues;

  // Declaration of defined functions.
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  bem_shape_values_same_panel(const FiniteElement<dim, spacedim> &kx_fe,
                              const FiniteElement<dim, spacedim> &ky_fe,
                              const QGauss<4> &          sauter_quad_rule,
                              Table<3, RangeNumberType> &kx_shape_value_table,
                              Table<3, RangeNumberType> &ky_shape_value_table);
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  bem_shape_values_common_edge(const FiniteElement<dim, spacedim> &kx_fe,
                               const FiniteElement<dim, spacedim> &ky_fe,
                               const QGauss<4> &          sauter_quad_rule,
                               Table<3, RangeNumberType> &kx_shape_value_table,
                               Table<3, RangeNumberType> &ky_shape_value_table);
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  bem_shape_values_common_vertex(
    const FiniteElement<dim, spacedim> &kx_fe,
    const FiniteElement<dim, spacedim> &ky_fe,
    const QGauss<4> &                   sauter_quad_rule,
    Table<3, RangeNumberType> &         kx_shape_value_table,
    Table<3, RangeNumberType> &         ky_shape_value_table);
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  bem_shape_values_regular(const FiniteElement<dim, spacedim> &kx_fe,
                           const FiniteElement<dim, spacedim> &ky_fe,
                           const QGauss<4> &                   sauter_quad_rule,
                           Table<3, RangeNumberType> &kx_shape_value_table,
                           Table<3, RangeNumberType> &ky_shape_value_table);
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  bem_shape_grad_matrices_same_panel(
    const FiniteElement<dim, spacedim> &   kx_fe,
    const FiniteElement<dim, spacedim> &   ky_fe,
    const QGauss<4> &                      sauter_quad_rule,
    Table<2, FullMatrix<RangeNumberType>> &kx_shape_grad_matrix_table,
    Table<2, FullMatrix<RangeNumberType>> &ky_shape_grad_matrix_table);
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  bem_shape_grad_matrices_common_edge(
    const FiniteElement<dim, spacedim> &   kx_fe,
    const FiniteElement<dim, spacedim> &   ky_fe,
    const QGauss<4> &                      sauter_quad_rule,
    Table<2, FullMatrix<RangeNumberType>> &kx_shape_grad_matrix_table,
    Table<2, FullMatrix<RangeNumberType>> &ky_shape_grad_matrix_table);
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  bem_shape_grad_matrices_common_vertex(
    const FiniteElement<dim, spacedim> &   kx_fe,
    const FiniteElement<dim, spacedim> &   ky_fe,
    const QGauss<4> &                      sauter_quad_rule,
    Table<2, FullMatrix<RangeNumberType>> &kx_shape_grad_matrix_table,
    Table<2, FullMatrix<RangeNumberType>> &ky_shape_grad_matrix_table);
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  bem_shape_grad_matrices_regular(
    const FiniteElement<dim, spacedim> &   kx_fe,
    const FiniteElement<dim, spacedim> &   ky_fe,
    const QGauss<4> &                      sauter_quad_rule,
    Table<2, FullMatrix<RangeNumberType>> &kx_shape_grad_matrix_table,
    Table<2, FullMatrix<RangeNumberType>> &ky_shape_grad_matrix_table);


  enum CellNeighboringType
  {
    SamePanel,
    CommonEdge,
    CommonVertex,
    Regular,
    None
  };


  template <int dim, int spacedim>
  std::vector<types::global_dof_index>
  get_conflict_indices(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell)
  {
    std::vector<types::global_dof_index> conflict_indices(
      cell->get_fe().dofs_per_cell);

    cell->get_dof_indices(conflict_indices);

    return conflict_indices;
  }


  template <int dim, int spacedim>
  std::array<types::global_vertex_index, GeometryInfo<dim>::vertices_per_cell>
  get_vertex_indices(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell)
  {
    std::array<types::global_vertex_index, GeometryInfo<dim>::vertices_per_cell>
      cell_vertex_indices;

    for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; v++)
      {
        cell_vertex_indices[v] = cell->vertex_index(v);
      }

    return cell_vertex_indices;
  }


  template <int dim, int spacedim>
  std::array<types::global_dof_index, GeometryInfo<dim>::vertices_per_cell>
  get_vertex_dof_indices(
    const typename DoFHandler<dim, spacedim>::cell_iterator &cell)
  {
    // Ensure that there is only one DoF associated with each vertex.
    Assert(cell->get_fe().dofs_per_vertex == 1,
           ExcDimensionMismatch(cell->get_fe().dofs_per_vertex, 1));

    std::array<types::global_dof_index, GeometryInfo<dim>::vertices_per_cell>
      cell_vertex_dof_indices;

    for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; v++)
      {
        cell_vertex_dof_indices[v] = cell->vertex_dof_index(v, 0);
      }

    return cell_vertex_dof_indices;
  }



  // Detect cell neighboring type based on vertex indices.
  template <int dim>
  CellNeighboringType
  detect_cell_neighboring_type(
    const std::array<types::global_vertex_index,
                     GeometryInfo<dim>::vertices_per_cell>
      &first_cell_vertex_indices,
    const std::array<types::global_vertex_index,
                     GeometryInfo<dim>::vertices_per_cell>
      &                                      second_cell_vertex_indices,
    std::vector<types::global_vertex_index> &vertex_index_intersection)
  {
    std::array<types::global_vertex_index, GeometryInfo<dim>::vertices_per_cell>
      first_cell_vertex_indices_sorted(first_cell_vertex_indices);
    std::array<types::global_vertex_index, GeometryInfo<dim>::vertices_per_cell>
      second_cell_vertex_indices_sorted(second_cell_vertex_indices);

    std::sort(first_cell_vertex_indices_sorted.begin(),
              first_cell_vertex_indices_sorted.end());
    std::sort(second_cell_vertex_indices_sorted.begin(),
              second_cell_vertex_indices_sorted.end());

    // Calculate the intersection of the two cells' vertex indices.
    std::set_intersection(first_cell_vertex_indices_sorted.begin(),
                          first_cell_vertex_indices_sorted.end(),
                          second_cell_vertex_indices_sorted.begin(),
                          second_cell_vertex_indices_sorted.end(),
                          std::back_inserter(vertex_index_intersection));

    CellNeighboringType cell_neighboring_type;
    switch (vertex_index_intersection.size())
      {
        case (0):
          cell_neighboring_type = Regular;
          break;
        case (1):
          cell_neighboring_type = CommonVertex;
          break;
        case (2):
          cell_neighboring_type = CommonEdge;
          break;
        case (4):
          cell_neighboring_type = SamePanel;
          break;
        default:
          cell_neighboring_type = None;
          Assert(false, ExcInternalError());
      }

    return cell_neighboring_type;
  }


  // Detect cell neighboring type based on vertex dof indices.
  template <int dim>
  CellNeighboringType
  detect_cell_neighboring_type(
    const std::array<types::global_dof_index,
                     GeometryInfo<dim>::vertices_per_cell>
      &first_cell_vertex_dof_indices,
    const std::array<types::global_dof_index,
                     GeometryInfo<dim>::vertices_per_cell>
      &                                   second_cell_vertex_dof_indices,
    std::vector<types::global_dof_index> &vertex_dof_index_intersection)
  {
    std::array<types::global_dof_index, GeometryInfo<dim>::vertices_per_cell>
      first_cell_vertex_dof_indices_sorted(first_cell_vertex_dof_indices);
    std::array<types::global_dof_index, GeometryInfo<dim>::vertices_per_cell>
      second_cell_vertex_dof_indices_sorted(second_cell_vertex_dof_indices);

    std::sort(first_cell_vertex_dof_indices_sorted.begin(),
              first_cell_vertex_dof_indices_sorted.end());
    std::sort(second_cell_vertex_dof_indices_sorted.begin(),
              second_cell_vertex_dof_indices_sorted.end());

    // Calculate the intersection of the two cells' vertex dof indices.
    std::set_intersection(first_cell_vertex_dof_indices_sorted.begin(),
                          first_cell_vertex_dof_indices_sorted.end(),
                          second_cell_vertex_dof_indices_sorted.begin(),
                          second_cell_vertex_dof_indices_sorted.end(),
                          std::back_inserter(vertex_dof_index_intersection));

    CellNeighboringType cell_neighboring_type;
    switch (vertex_dof_index_intersection.size())
      {
        case (0):
          cell_neighboring_type = Regular;
          break;
        case (1):
          cell_neighboring_type = CommonVertex;
          break;
        case (2):
          cell_neighboring_type = CommonEdge;
          break;
        case (4):
          cell_neighboring_type = SamePanel;
          break;
        default:
          Assert(false, ExcInternalError());
          cell_neighboring_type = None;
      }

    return cell_neighboring_type;
  }


  // Calculate the distance between the centers of two cells.
  template <int dim, int spacedim, typename Number = double>
  Number
  cell_distance(
    const typename Triangulation<dim, spacedim>::cell_iterator first_cell,
    const typename Triangulation<dim, spacedim>::cell_iterator second_cell)
  {
    return first_cell->center().distance(second_cell->center());
  }


  // This function calculate the matrix storing shape function gradient with
  // respect to area coordinates.
  // Each row of the matrix is the gradient of one of the shape functions.
  // @fe FE_Q finite element.
  // @dof_permutation is a list of shape function indices
  // with respect to the default hierarchical DoF numbering of FE_Q finite
  // element.
  // @p is the area coordinates at which the shape function's gradient is to be
  // evaluated.
  template <int dim, int spacedim>
  FullMatrix<double>
  shape_grad_matrix(const FiniteElement<dim, spacedim> &fe,
                    const std::vector<unsigned int> &   dof_permutation,
                    const Point<dim> &                  p)
  {
    FullMatrix<double> shape_grad_matrix(fe.dofs_per_cell, dim);

    for (unsigned int i = 0; i < fe.dofs_per_cell; i++)
      {
        // The gradient of a shape function as a tensor.
        Tensor<1, dim> shape_grad_tensor =
          fe.shape_grad(dof_permutation.at(i), p);
        // Assign the gradient of the shape function to the matrix.
        for (unsigned int j = 0; j < dim; j++)
          {
            shape_grad_matrix(i, j) = shape_grad_tensor[j];
          }
      }

    return shape_grad_matrix;
  }


  // This function calculate the matrix storing shape function gradient with
  // respect to area coordinates.
  // Each row of the matrix is the gradient of one of the shape functions.
  // The rows of shape function gradients are arranged in the default
  // hierarchical order.
  // @fe is the FE_Q finite element.
  // @p is the area coordinates at which the shape gradient is to be evaluated.
  template <int dim, int spacedim>
  FullMatrix<double>
  shape_grad_matrix_in_hierarchical_order(
    const FiniteElement<dim, spacedim> &fe,
    const Point<dim> &                  p)
  {
    FullMatrix<double> shape_grad_matrix(fe.dofs_per_cell, dim);

    for (unsigned int i = 0; i < fe.dofs_per_cell; i++)
      {
        // The gradient of a shape function as a tensor.
        Tensor<1, dim> shape_grad_tensor = fe.shape_grad(i, p);
        // Assign the gradient of the shape function to the matrix.
        for (unsigned int j = 0; j < dim; j++)
          {
            shape_grad_matrix(i, j) = shape_grad_tensor[j];
          }
      }

    return shape_grad_matrix;
  }


  // This function calculate the matrix storing shape function gradient with
  // respect to area coordinates.
  // Each row of the matrix is the gradient of one of the shape functions.
  // The rows of shape function gradients are arranged in the tensor product
  // order.
  // @fe is the FE_Q finite element.
  // @p is the area coordinates at which the shape gradient is to be evaluated.
  template <int dim, int spacedim>
  FullMatrix<double>
  shape_grad_matrix_in_tensor_product_order(
    const FiniteElement<dim, spacedim> &fe,
    const Point<dim> &                  p)
  {
    try
      {
        using FE_Poly = FE_Poly<TensorProductPolynomials<dim>, dim, spacedim>;
        const FE_Poly &fe_poly = dynamic_cast<const FE_Poly &>(fe);

        // Use the inverse numbering of polynomial space to restore the tensor
        // product ordering of shape functions.
        return shape_grad_matrix(fe,
                                 fe_poly.get_poly_space_numbering_inverse(),
                                 p);
      }
    catch (const std::bad_cast &e)
      {
        Assert(false, ExcInternalError());
        return FullMatrix<double>();
      }
  }


  // This function calculates shape function values at a area coordinates under
  // a specified DoF permutation.
  template <int dim, int spacedim>
  Vector<double>
  shape_values(const FiniteElement<dim, spacedim> &fe,
               const std::vector<unsigned int> &   dof_permutation,
               const Point<dim> &                  p)
  {
    Vector<double> shape_values_vector(fe.dofs_per_cell);

    for (unsigned int i = 0; i < fe.dofs_per_cell; i++)
      {
        shape_values_vector(i) = fe.shape_value(dof_permutation.at(i), p);
      }

    return shape_values_vector;
  }


  // This function calculates shape function values at a area coordinates under
  // the default hierarchical order.
  template <int dim, int spacedim>
  Vector<double>
  shape_values_in_hierarchical_order(const FiniteElement<dim, spacedim> &fe,
                                     const Point<dim> &                  p)
  {
    Vector<double> shape_values_vector(fe.dofs_per_cell);

    for (unsigned int i = 0; i < fe.dofs_per_cell; i++)
      {
        shape_values_vector(i) = fe.shape_value(i, p);
      }

    return shape_values_vector;
  }


  // This function calculates shape function values at a area coordinates under
  // the tensor product order.
  // @fe is the FE_Q finite element.
  // @p is the area coordinates at which the shape gradient is to be evaluated.
  template <int dim, int spacedim>
  Vector<double>
  shape_values_in_tensor_product_order(const FiniteElement<dim, spacedim> &fe,
                                       const Point<dim> &                  p)
  {
    try
      {
        using FE_Poly = FE_Poly<TensorProductPolynomials<dim>, dim, spacedim>;
        const FE_Poly &fe_poly = dynamic_cast<const FE_Poly &>(fe);

        // Use the inverse numbering of polynomial space to restore the tensor
        // product ordering of shape functions.
        return shape_values(fe, fe_poly.get_poly_space_numbering_inverse(), p);
      }
    catch (const std::bad_cast &e)
      {
        Assert(false, ExcInternalError());
        return Vector<double>();
      }
  }


  // Collect two coordinate components from the list of points in 3D space.
  FullMatrix<double>
  collect_two_components_from_point3(const std::vector<Point<3>> &points,
                                     const unsigned int first_component,
                                     const unsigned int second_component)
  {
    Assert(first_component < 3, ExcInternalError());
    Assert(second_component < 3, ExcInternalError());

    FullMatrix<double> two_component_coords(2, points.size());

    for (unsigned int i = 0; i < points.size(); i++)
      {
        two_component_coords(0, i) = points.at(i)(first_component);
        two_component_coords(1, i) = points.at(i)(second_component);
      }

    return two_component_coords;
  }


  // Calculate support points in the real cell with an ordering specified by
  // the sequence of DoF numbering.
  template <int dim, int spacedim>
  std::vector<Point<spacedim>>
  support_points_in_real_cell(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const FiniteElement<dim, spacedim> &                        fe,
    const MappingQGeneric<dim, spacedim> &                      mapping,
    const std::vector<unsigned int> &                           dof_permutation)
  {
    try
      {
        using FE_Poly = FE_Poly<TensorProductPolynomials<dim>, dim, spacedim>;
        const FE_Poly &fe_poly = dynamic_cast<const FE_Poly &>(fe);

        // The Mapping object should has the same degree as the FE_Q object.
        Assert(fe_poly.get_degree() == mapping.get_degree(),
               ExcInternalError());
      }
    catch (const std::bad_cast &e)
      {
        Assert(false, ExcInternalError());
      }

    const unsigned int           dofs_per_cell = fe.dofs_per_cell;
    std::vector<Point<spacedim>> real_support_points(dofs_per_cell);

    if (fe.has_face_support_points())
      {
        // Get the list of support points in the unit cell in the default
        // hierarchical ordering.
        const std::vector<Point<dim>> &unit_support_points =
          fe.get_unit_support_points();

        // Transform the support points from unit cell to real cell.
        for (unsigned int i = 0; i < dofs_per_cell; i++)
          {
            real_support_points.at(i) = mapping.transform_unit_to_real_cell(
              cell, unit_support_points.at(dof_permutation.at(i)));
          }
      }
    else
      {
        Assert(false, ExcInternalError());
      }

    return real_support_points;
  }

  /**
   * Calculate support points in real cell in the default hierarchical ordering.
   */
  template <int dim, int spacedim>
  std::vector<Point<spacedim>>
  hierarchical_support_points_in_real_cell(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const FiniteElement<dim, spacedim> &                        fe,
    const MappingQGeneric<dim, spacedim> &                      mapping)
  {
    const unsigned int           dofs_per_cell = fe.dofs_per_cell;
    std::vector<Point<spacedim>> real_support_points(dofs_per_cell);

    try
      {
        using FE_Poly = FE_Poly<TensorProductPolynomials<dim>, dim, spacedim>;

        // The Mapping object should has the same degree as the FE_Q object,
        // i.e. isometric element.
        Assert(dynamic_cast<const FE_Poly &>(fe).get_degree() ==
                 mapping.get_degree(),
               ExcInternalError());

        if (fe.has_face_support_points())
          {
            // Get the list of support points in the unit cell in the default
            // hierarchical ordering.
            const std::vector<Point<dim>> &unit_support_points =
              fe.get_unit_support_points();

            // Transform the support points from unit cell to real cell.
            for (unsigned int i = 0; i < dofs_per_cell; i++)
              {
                real_support_points.at(i) = mapping.transform_unit_to_real_cell(
                  cell, unit_support_points.at(i));
              }
          }
        else
          {
            Assert(false, ExcInternalError());
          }
      }
    catch (const std::bad_cast &e)
      {
        Assert(false, ExcInternalError());
      }

    return real_support_points;
  }


  /**
   * Calculate support points in real cell in the default hierarchical ordering.
   * The memory for storing these support points is pre-allocated.
   */
  template <int dim, int spacedim>
  void
  hierarchical_support_points_in_real_cell(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const FiniteElement<dim, spacedim> &                        fe,
    const MappingQGeneric<dim, spacedim> &                      mapping,
    std::vector<Point<spacedim>> &real_support_points)
  {
    const unsigned int dofs_per_cell = fe.dofs_per_cell;

    Assert(real_support_points.size() == dofs_per_cell,
           ExcDimensionMismatch(real_support_points.size(), dofs_per_cell));

    try
      {
        using FE_Poly = FE_Poly<TensorProductPolynomials<dim>, dim, spacedim>;

        // The Mapping object should has the same degree as the FE_Q object,
        // i.e. isometric element.
        Assert(dynamic_cast<const FE_Poly &>(fe).get_degree() ==
                 mapping.get_degree(),
               ExcInternalError());

        if (fe.has_face_support_points())
          {
            // Get the list of support points in the unit cell in the default
            // hierarchical ordering.
            const std::vector<Point<dim>> &unit_support_points =
              fe.get_unit_support_points();

            // Transform the support points from unit cell to real cell.
            for (unsigned int i = 0; i < dofs_per_cell; i++)
              {
                real_support_points.at(i) = mapping.transform_unit_to_real_cell(
                  cell, unit_support_points.at(i));
              }
          }
        else
          {
            Assert(false, ExcInternalError());
          }
      }
    catch (const std::bad_cast &e)
      {
        Assert(false, ExcInternalError());
      }
  }


  // Calculate support points in real cell in tensor product ordering.
  template <int dim, int spacedim>
  std::vector<Point<spacedim>>
  tensor_product_support_points_in_real_cell(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const FiniteElement<dim, spacedim> &                        fe,
    const MappingQGeneric<dim, spacedim> &                      mapping)
  {
    const unsigned int           dofs_per_cell = fe.dofs_per_cell;
    std::vector<Point<spacedim>> real_support_points(dofs_per_cell);

    try
      {
        using FE_Poly = FE_Poly<TensorProductPolynomials<dim>, dim, spacedim>;
        const FE_Poly &fe_poly = dynamic_cast<const FE_Poly &>(fe);

        // The Mapping object should has the same degree as the FE_Q object.
        Assert(fe_poly.get_degree() == mapping.get_degree(),
               ExcInternalError());

        if (fe.has_face_support_points())
          {
            std::vector<unsigned int> poly_space_inverse_numbering(
              fe_poly.get_poly_space_numbering_inverse());
            real_support_points = support_points_in_real_cell(
              cell, fe, mapping, poly_space_inverse_numbering);
          }
        else
          {
            Assert(false, ExcInternalError());
          }
      }
    catch (const std::bad_cast &e)
      {
        Assert(false, ExcInternalError());
      }

    return real_support_points;
  }


  /**
   * Calculate the surface Jacobian determinant at specified area coordinates.
   * Shape functions and support points in the real cell have been reordered to
   * the tensor product ordering before the calculation.
   */
  template <int dim, int spacedim>
  double
  surface_jacobian_det(
    const FiniteElement<dim, spacedim> &fe,
    const std::vector<Point<spacedim>> &support_points_in_real_cell,
    const Point<dim> &                  p)
  {
    // Currently, only dim=2 and spacedim=3 are supported.
    Assert((dim == 2) && (spacedim == 3), ExcInternalError());

    if (fe.has_support_points())
      {
        // Calculate the shape function's gradient matrix.
        FullMatrix<double> shape_grad_matrix_at_p(
          shape_grad_matrix_in_tensor_product_order(fe, p));

        double             jacobian_det_squared = 0.0;
        FullMatrix<double> jacobian_matrix_2x2(2, 2);
        for (unsigned int i = 0; i < 3; i++)
          {
            FullMatrix<double> support_point_components =
              collect_two_components_from_point3(support_points_in_real_cell,
                                                 i,
                                                 (i + 1) % 3);
            support_point_components.mmult(jacobian_matrix_2x2,
                                           shape_grad_matrix_at_p);
            jacobian_det_squared +=
              std::pow(jacobian_matrix_2x2.determinant(), 2);
          }

        return std::sqrt(jacobian_det_squared);
      }
    else
      {
        Assert(false, ExcInternalError());

        return -1;
      }
  }


  /**
   * Calculate the surface Jacobian determinant at specified area coordinates.
   * Shape functions and support points in the real cell have been reordered to
   * the tensor product ordering before the calculation.
   */
  template <int spacedim>
  double
  surface_jacobian_det(
    const unsigned int                  k3_index,
    const unsigned int                  quad_no,
    const Table<2, FullMatrix<double>> &shape_grad_matrix_table,
    const std::vector<Point<spacedim>> &support_points_in_real_cell)
  {
    // Currently, only spacedim=3 is supported.
    Assert(spacedim == 3, ExcInternalError());

    // Extract the shape function's gradient matrix at the specified quadrature
    // point.
    const FullMatrix<double> &shape_grad_matrix_at_p =
      shape_grad_matrix_table(k3_index, quad_no);

    double             jacobian_det_squared = 0.0;
    FullMatrix<double> jacobian_matrix_2x2(2, 2);
    for (unsigned int i = 0; i < 3; i++)
      {
        FullMatrix<double> support_point_components =
          collect_two_components_from_point3(support_points_in_real_cell,
                                             i,
                                             (i + 1) % 3);
        support_point_components.mmult(jacobian_matrix_2x2,
                                       shape_grad_matrix_at_p);
        jacobian_det_squared += std::pow(jacobian_matrix_2x2.determinant(), 2);
      }

    return std::sqrt(jacobian_det_squared);
  }


  // Calculate the surface Jacobian determinant and normal vector at specified
  // area coordinates. Shape functions and support points in the real cell have
  // been reordered to the tensor product ordering before the calculation.
  template <int dim, int spacedim>
  double
  surface_jacobian_det_and_normal_vector(
    const FiniteElement<dim, spacedim> &fe,
    const std::vector<Point<spacedim>> &support_points_in_real_cell,
    const Point<dim> &                  p,
    Tensor<1, spacedim> &               normal_vector)
  {
    // Currently, only dim=2 and spacedim=3 are supported.
    Assert((dim == 2) && (spacedim == 3), ExcInternalError());

    if (fe.has_support_points())
      {
        // Calculate the shape function's gradient matrix.
        FullMatrix<double> shape_grad_matrix_at_p(
          shape_grad_matrix_in_tensor_product_order(fe, p));

        double             surface_jacobian_det = 0.0;
        FullMatrix<double> jacobian_matrix_2x2(2, 2);
        double             surface_jacobian_det_components[3];
        for (unsigned int i = 0; i < 3; i++)
          {
            FullMatrix<double> support_point_components =
              collect_two_components_from_point3(support_points_in_real_cell,
                                                 i,
                                                 (i + 1) % 3);
            support_point_components.mmult(jacobian_matrix_2x2,
                                           shape_grad_matrix_at_p);
            surface_jacobian_det_components[i] =
              jacobian_matrix_2x2.determinant();
            surface_jacobian_det +=
              std::pow(surface_jacobian_det_components[i], 2);
          }

        surface_jacobian_det = std::sqrt(surface_jacobian_det);

        for (unsigned int i = 0; i < 3; i++)
          {
            normal_vector[i] = surface_jacobian_det_components[(i + 1) % 3] /
                               surface_jacobian_det;
          }

        return surface_jacobian_det;
      }
    else
      {
        Assert(false, ExcInternalError());

        return -1;
      }
  }


  // Calculate the surface Jacobian determinant and normal vector at specified
  // area coordinates. Shape functions and support points in the real cell have
  // been reordered to the tensor product ordering before the calculation.
  template <int spacedim>
  double
  surface_jacobian_det_and_normal_vector(
    const unsigned int                  k3_index,
    const unsigned int                  quad_no,
    const Table<2, FullMatrix<double>> &shape_grad_matrix_table,
    const std::vector<Point<spacedim>> &support_points_in_real_cell,
    Tensor<1, spacedim> &               normal_vector)
  {
    // Currently, only spacedim=3 is supported.
    Assert(spacedim == 3, ExcInternalError());

    // Extract the shape function's gradient matrix at the specified quadrature
    // point.
    const FullMatrix<double> &shape_grad_matrix_at_p =
      shape_grad_matrix_table(k3_index, quad_no);

    double             surface_jacobian_det = 0.0;
    FullMatrix<double> jacobian_matrix_2x2(2, 2);
    double             surface_jacobian_det_components[3];
    for (unsigned int i = 0; i < 3; i++)
      {
        FullMatrix<double> support_point_components =
          collect_two_components_from_point3(support_points_in_real_cell,
                                             i,
                                             (i + 1) % 3);
        support_point_components.mmult(jacobian_matrix_2x2,
                                       shape_grad_matrix_at_p);
        surface_jacobian_det_components[i] = jacobian_matrix_2x2.determinant();
        surface_jacobian_det += std::pow(surface_jacobian_det_components[i], 2);
      }

    surface_jacobian_det = std::sqrt(surface_jacobian_det);

    for (unsigned int i = 0; i < 3; i++)
      {
        normal_vector[i] =
          surface_jacobian_det_components[(i + 1) % 3] / surface_jacobian_det;
      }

    return surface_jacobian_det;
  }


  /**
   * Calculate the coordinate transformation from a unit cell to real cell for
   * Qp element.
   * @param fe
   * @param support_points_in_real_cell coordinates of the support points in the
   * real cell, which are in the tensor product order.
   * @param area_coords
   * @return
   */
  template <int dim, int spacedim>
  Point<spacedim>
  transform_unit_to_permuted_real_cell(
    const FiniteElement<dim, spacedim> &fe,
    const std::vector<Point<spacedim>> &support_points_in_real_cell,
    const Point<dim> &                  area_coords)
  {
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    Assert(dofs_per_cell == support_points_in_real_cell.size(),
           ExcDimensionMismatch(dofs_per_cell,
                                support_points_in_real_cell.size()));

    Point<spacedim> real_coords;
    Vector<double>  shape_values_vector(
      shape_values_in_tensor_product_order(fe, area_coords));

    // Linear combination of support point coordinates and evaluation
    // of shape functions at specified area coordinates.
    for (unsigned int i = 0; i < dofs_per_cell; i++)
      {
        real_coords = real_coords + shape_values_vector(i) *
                                      support_points_in_real_cell.at(i);
      }

    return real_coords;
  }


  /**
   * Calculate the coordinate transformation from a unit cell to real cell for
   * Qp element.
   * @param k3_index
   * @param quad_no
   * @param shape_value_table
   * @param support_points_in_real_cell coordinates of the support points in the
   * real cell, which are in the tensor product order.
   * @return
   */
  template <int spacedim>
  Point<spacedim>
  transform_unit_to_permuted_real_cell(
    const unsigned int                  k3_index,
    const unsigned int                  quad_no,
    const Table<3, double> &            shape_value_table,
    const std::vector<Point<spacedim>> &support_points_in_real_cell)
  {
    const unsigned int dofs_per_cell = shape_value_table.size(0);
    Assert(dofs_per_cell == support_points_in_real_cell.size(),
           ExcDimensionMismatch(dofs_per_cell,
                                support_points_in_real_cell.size()));

    Point<spacedim> real_coords;

    // Linear combination of support point coordinates and evaluation
    // of shape functions at specified area coordinates.
    for (unsigned int i = 0; i < dofs_per_cell; i++)
      {
        real_coords = real_coords + shape_value_table(i, k3_index, quad_no) *
                                      support_points_in_real_cell.at(i);
      }

    return real_coords;
  }


  /**
   * Generate the permutation of polynomial space inverse numbering by starting
   * from the specified corner in the forward direction.
   */
  template <int dim, int spacedim>
  std::vector<unsigned int>
  generate_forward_dof_permutation(const FiniteElement<dim, spacedim> &fe,
                                   unsigned int starting_corner)
  {
    // Currently, only dim=2 and spacedim=3 are supported.
    Assert((dim == 2) && (spacedim == 3), ExcInternalError());

    try
      {
        using FE_Poly = FE_Poly<TensorProductPolynomials<dim>, dim, spacedim>;
        const FE_Poly &fe_poly = dynamic_cast<const FE_Poly &>(fe);

        std::vector<unsigned int> poly_space_inverse_numbering(
          fe_poly.get_poly_space_numbering_inverse());
        std::vector<unsigned int> dof_permutation(
          poly_space_inverse_numbering.size());

        const int poly_degree = fe.degree;
        Assert((poly_degree + 1) * (poly_degree + 1) == fe.dofs_per_cell,
               ExcDimensionMismatch((poly_degree + 1) * (poly_degree + 1),
                                    fe.dofs_per_cell));

        // Store the inverse numbering into a matrix for further traversing.
        unsigned int             c = 0;
        FullMatrix<unsigned int> dof_numbering_matrix(poly_degree + 1,
                                                      poly_degree + 1);
        for (int i = poly_degree; i >= 0; i--)
          {
            for (int j = 0; j <= poly_degree; j++)
              {
                dof_numbering_matrix(i, j) = poly_space_inverse_numbering.at(c);
                c++;
              }
          }

        switch (starting_corner)
          {
            case 0:
              return poly_space_inverse_numbering;

              break;
            case 1:
              c = 0;
              for (int j = poly_degree; j >= 0; j--)
                {
                  for (int i = poly_degree; i >= 0; i--)
                    {
                      dof_permutation.at(c) = dof_numbering_matrix(i, j);
                      c++;
                    }
                }

              break;
            case 2:
              c = 0;
              for (int j = 0; j <= poly_degree; j++)
                {
                  for (int i = 0; i <= poly_degree; i++)
                    {
                      dof_permutation.at(c) = dof_numbering_matrix(i, j);
                      c++;
                    }
                }

              break;
            case 3:
              c = 0;
              for (int i = 0; i <= poly_degree; i++)
                {
                  for (int j = poly_degree; j >= 0; j--)
                    {
                      dof_permutation.at(c) = dof_numbering_matrix(i, j);
                      c++;
                    }
                }

              break;
            default:
              Assert(false, ExcInternalError());
          }

        return dof_permutation;
      }
    catch (const std::bad_cast &e)
      {
        Assert(false, ExcInternalError());

        return std::vector<unsigned int>();
      }
  }


  /**
   * Generate the permutation of polynomial space inverse numbering by starting
   * from the specified corner in the forward direction. This overloaded version
   * has the returned vector as its argument.
   */
  template <int dim, int spacedim>
  void
  generate_forward_dof_permutation(const FiniteElement<dim, spacedim> &fe,
                                   unsigned int               starting_corner,
                                   std::vector<unsigned int> &dof_permutation)
  {
    // Currently, only dim=2 and spacedim=3 are supported.
    Assert((dim == 2) && (spacedim == 3), ExcInternalError());

    try
      {
        using FE_Poly = FE_Poly<TensorProductPolynomials<dim>, dim, spacedim>;
        const FE_Poly &fe_poly = dynamic_cast<const FE_Poly &>(fe);

        std::vector<unsigned int> poly_space_inverse_numbering(
          fe_poly.get_poly_space_numbering_inverse());

        const int poly_degree = fe.degree;
        Assert((poly_degree + 1) * (poly_degree + 1) == fe.dofs_per_cell,
               ExcDimensionMismatch((poly_degree + 1) * (poly_degree + 1),
                                    fe.dofs_per_cell));
        Assert(dof_permutation.size() == fe.dofs_per_cell,
               ExcDimensionMismatch(dof_permutation.size(), fe.dofs_per_cell));

        // Store the inverse numbering into a matrix for further traversing.
        unsigned int             c = 0;
        FullMatrix<unsigned int> dof_numbering_matrix(poly_degree + 1,
                                                      poly_degree + 1);
        for (int i = poly_degree; i >= 0; i--)
          {
            for (int j = 0; j <= poly_degree; j++)
              {
                dof_numbering_matrix(i, j) = poly_space_inverse_numbering.at(c);
                c++;
              }
          }

        switch (starting_corner)
          {
            case 0:
              dof_permutation = poly_space_inverse_numbering;

              break;
            case 1:
              c = 0;
              for (int j = poly_degree; j >= 0; j--)
                {
                  for (int i = poly_degree; i >= 0; i--)
                    {
                      dof_permutation.at(c) = dof_numbering_matrix(i, j);
                      c++;
                    }
                }

              break;
            case 2:
              c = 0;
              for (int j = 0; j <= poly_degree; j++)
                {
                  for (int i = 0; i <= poly_degree; i++)
                    {
                      dof_permutation.at(c) = dof_numbering_matrix(i, j);
                      c++;
                    }
                }

              break;
            case 3:
              c = 0;
              for (int i = 0; i <= poly_degree; i++)
                {
                  for (int j = poly_degree; j >= 0; j--)
                    {
                      dof_permutation.at(c) = dof_numbering_matrix(i, j);
                      c++;
                    }
                }

              break;
            default:
              Assert(false, ExcInternalError());
          }
      }
    catch (const std::bad_cast &e)
      {
        Assert(false, ExcInternalError());
      }
  }


  /**
   * Generate the permutation of polynomial space inverse numbering by starting
   * from the specified corner in the backward direction. The index for
   * starting_corner starts from zero.
   */
  template <int dim, int spacedim>
  std::vector<unsigned int>
  generate_backward_dof_permutation(const FiniteElement<dim, spacedim> &fe,
                                    unsigned int starting_corner)
  {
    // Currently, only dim=2 and spacedim=3 are supported.
    Assert(dim == 2, ExcInternalError());

    try
      {
        using FE_Poly = FE_Poly<TensorProductPolynomials<dim>, dim, spacedim>;
        const FE_Poly &fe_poly = dynamic_cast<const FE_Poly &>(fe);

        std::vector<unsigned int> poly_space_inverse_numbering(
          fe_poly.get_poly_space_numbering_inverse());
        std::vector<unsigned int> dof_permutation(
          poly_space_inverse_numbering.size());

        const int poly_degree = fe.degree;
        Assert((poly_degree + 1) * (poly_degree + 1) == fe.dofs_per_cell,
               ExcDimensionMismatch((poly_degree + 1) * (poly_degree + 1),
                                    fe.dofs_per_cell));

        // Store the inverse numbering into a matrix for further traversing.
        unsigned int             c = 0;
        FullMatrix<unsigned int> dof_numbering_matrix(poly_degree + 1,
                                                      poly_degree + 1);
        for (int i = poly_degree; i >= 0; i--)
          {
            for (int j = 0; j <= poly_degree; j++)
              {
                dof_numbering_matrix(i, j) = poly_space_inverse_numbering.at(c);
                c++;
              }
          }

        switch (starting_corner)
          {
            case 0:
              c = 0;
              for (int j = 0; j <= poly_degree; j++)
                {
                  for (int i = poly_degree; i >= 0; i--)
                    {
                      dof_permutation.at(c) = dof_numbering_matrix(i, j);
                      c++;
                    }
                }

              break;
            case 1:
              c = 0;
              for (int i = poly_degree; i >= 0; i--)
                {
                  for (int j = poly_degree; j >= 0; j--)
                    {
                      dof_permutation.at(c) = dof_numbering_matrix(i, j);
                      c++;
                    }
                }

              break;
            case 2:
              c = 0;
              for (int i = 0; i <= poly_degree; i++)
                {
                  for (int j = 0; j <= poly_degree; j++)
                    {
                      dof_permutation.at(c) = dof_numbering_matrix(i, j);
                      c++;
                    }
                }

              break;
            case 3:
              c = 0;
              for (int j = poly_degree; j >= 0; j--)
                {
                  for (int i = 0; i <= poly_degree; i++)
                    {
                      dof_permutation.at(c) = dof_numbering_matrix(i, j);
                      c++;
                    }
                }

              break;
            default:
              Assert(false, ExcInternalError());
          }

        return dof_permutation;
      }
    catch (const std::bad_cast &e)
      {
        Assert(false, ExcInternalError());

        return std::vector<unsigned int>();
      }
  }


  /**
   * Generate the permutation of polynomial space inverse numbering by starting
   * from the specified corner in the backward direction. The index for
   * starting_corner starts from zero. This overloaded version has the returned
   * vector as its argument.
   */
  template <int dim, int spacedim>
  void
  generate_backward_dof_permutation(const FiniteElement<dim, spacedim> &fe,
                                    unsigned int               starting_corner,
                                    std::vector<unsigned int> &dof_permutation)
  {
    // Currently, only dim=2 and spacedim=3 are supported.
    Assert(dim == 2, ExcInternalError());

    try
      {
        using FE_Poly = FE_Poly<TensorProductPolynomials<dim>, dim, spacedim>;
        const FE_Poly &fe_poly = dynamic_cast<const FE_Poly &>(fe);

        std::vector<unsigned int> poly_space_inverse_numbering(
          fe_poly.get_poly_space_numbering_inverse());

        const int poly_degree = fe.degree;
        Assert((poly_degree + 1) * (poly_degree + 1) == fe.dofs_per_cell,
               ExcDimensionMismatch((poly_degree + 1) * (poly_degree + 1),
                                    fe.dofs_per_cell));
        Assert(dof_permutation.size() == fe.dofs_per_cell,
               ExcDimensionMismatch(dof_permutation.size(), fe.dofs_per_cell));

        // Store the inverse numbering into a matrix for further traversing.
        unsigned int             c = 0;
        FullMatrix<unsigned int> dof_numbering_matrix(poly_degree + 1,
                                                      poly_degree + 1);
        for (int i = poly_degree; i >= 0; i--)
          {
            for (int j = 0; j <= poly_degree; j++)
              {
                dof_numbering_matrix(i, j) = poly_space_inverse_numbering.at(c);
                c++;
              }
          }

        switch (starting_corner)
          {
            case 0:
              c = 0;
              for (int j = 0; j <= poly_degree; j++)
                {
                  for (int i = poly_degree; i >= 0; i--)
                    {
                      dof_permutation.at(c) = dof_numbering_matrix(i, j);
                      c++;
                    }
                }

              break;
            case 1:
              c = 0;
              for (int i = poly_degree; i >= 0; i--)
                {
                  for (int j = poly_degree; j >= 0; j--)
                    {
                      dof_permutation.at(c) = dof_numbering_matrix(i, j);
                      c++;
                    }
                }

              break;
            case 2:
              c = 0;
              for (int i = 0; i <= poly_degree; i++)
                {
                  for (int j = 0; j <= poly_degree; j++)
                    {
                      dof_permutation.at(c) = dof_numbering_matrix(i, j);
                      c++;
                    }
                }

              break;
            case 3:
              c = 0;
              for (int j = poly_degree; j >= 0; j--)
                {
                  for (int i = 0; i <= poly_degree; i++)
                    {
                      dof_permutation.at(c) = dof_numbering_matrix(i, j);
                      c++;
                    }
                }

              break;
            default:
              Assert(false, ExcInternalError());
          }
      }
    catch (const std::bad_cast &e)
      {
        Assert(false, ExcInternalError());
      }
  }


  template <typename T>
  std::vector<T>
  permute_vector(const std::vector<T> &           input_vector,
                 const std::vector<unsigned int> &permutation_indices)
  {
    const unsigned int N = input_vector.size();
    Assert(N == permutation_indices.size(),
           ExcDimensionMismatch(N, permutation_indices.size()));

    std::vector<T> permuted_vector(N);

    for (unsigned int i = 0; i < N; i++)
      {
        permuted_vector.at(i) = input_vector.at(permutation_indices.at(i));
      }

    return permuted_vector;
  }


  /**
   * Permute a vector according to the specified permutation indices. The memory
   * for the output vector should be pre-allocated.
   * @param input_vector
   * @param permutation_indices
   * @param permuted_vector
   */
  template <typename T>
  void
  permute_vector(const std::vector<T> &           input_vector,
                 const std::vector<unsigned int> &permutation_indices,
                 std::vector<T> &                 permuted_vector)
  {
    const unsigned int N = input_vector.size();
    Assert(N == permutation_indices.size(),
           ExcDimensionMismatch(N, permutation_indices.size()));
    Assert(N == permuted_vector.size(),
           ExcDimensionMismatch(N, permuted_vector.size()));

    for (unsigned int i = 0; i < N; i++)
      {
        permuted_vector.at(i) = input_vector.at(permutation_indices.at(i));
      }
  }


  template <int dim, int spacedim, typename RangeNumberType>
  class BEMValues
  {
  public:
    // N.B. There is no default constructor for the class
    // <code>BEMValues</code>, because all the internal references to finite
    // element objects and quadrature objects should be initialized once the
    // <code>BEMValues</code> object is declared.
    BEMValues(const FiniteElement<dim, spacedim> &kx_fe,
              const FiniteElement<dim, spacedim> &ky_fe,
              const QGauss<4> &                   quad_rule_for_same_panel,
              const QGauss<4> &                   quad_rule_for_common_edge,
              const QGauss<4> &                   quad_rule_for_common_vertex,
              const QGauss<4> &                   quad_rule_for_regular);


    /**
     * Copy constructor for class <code>BEMValues</code>.
     * @param bem_values
     */
    BEMValues(const BEMValues<dim, spacedim, RangeNumberType> &bem_values);

    const FiniteElement<dim, spacedim> &kx_fe;
    const FiniteElement<dim, spacedim> &ky_fe;
    const QGauss<4> &                   quad_rule_for_same_panel;
    const QGauss<4> &                   quad_rule_for_common_edge;
    const QGauss<4> &                   quad_rule_for_common_vertex;
    const QGauss<4> &                   quad_rule_for_regular;

    Table<3, RangeNumberType> kx_shape_value_table_for_same_panel;
    Table<3, RangeNumberType> ky_shape_value_table_for_same_panel;
    Table<3, RangeNumberType> kx_shape_value_table_for_common_edge;
    Table<3, RangeNumberType> ky_shape_value_table_for_common_edge;
    Table<3, RangeNumberType> kx_shape_value_table_for_common_vertex;
    Table<3, RangeNumberType> ky_shape_value_table_for_common_vertex;
    Table<3, RangeNumberType> kx_shape_value_table_for_regular;
    Table<3, RangeNumberType> ky_shape_value_table_for_regular;

    Table<2, FullMatrix<RangeNumberType>>
      kx_shape_grad_matrix_table_for_same_panel;
    Table<2, FullMatrix<RangeNumberType>>
      ky_shape_grad_matrix_table_for_same_panel;
    Table<2, FullMatrix<RangeNumberType>>
      kx_shape_grad_matrix_table_for_common_edge;
    Table<2, FullMatrix<RangeNumberType>>
      ky_shape_grad_matrix_table_for_common_edge;
    Table<2, FullMatrix<RangeNumberType>>
      kx_shape_grad_matrix_table_for_common_vertex;
    Table<2, FullMatrix<RangeNumberType>>
      ky_shape_grad_matrix_table_for_common_vertex;
    Table<2, FullMatrix<RangeNumberType>>
      kx_shape_grad_matrix_table_for_regular;
    Table<2, FullMatrix<RangeNumberType>>
      ky_shape_grad_matrix_table_for_regular;

    void
    fill_shape_value_tables();

    void
    fill_shape_grad_matrix_tables();

  protected:
    void
    init_shape_value_tables();

    void
    init_shape_grad_matrix_tables();
  };


  template <int dim, int spacedim, typename RangeNumberType>
  BEMValues<dim, spacedim, RangeNumberType>::BEMValues(
    const FiniteElement<dim, spacedim> &kx_fe,
    const FiniteElement<dim, spacedim> &ky_fe,
    const QGauss<4> &                   quad_rule_for_same_panel,
    const QGauss<4> &                   quad_rule_for_common_edge,
    const QGauss<4> &                   quad_rule_for_common_vertex,
    const QGauss<4> &                   quad_rule_for_regular)
    : kx_fe(kx_fe)
    , ky_fe(ky_fe)
    , quad_rule_for_same_panel(quad_rule_for_same_panel)
    , quad_rule_for_common_edge(quad_rule_for_common_edge)
    , quad_rule_for_common_vertex(quad_rule_for_common_vertex)
    , quad_rule_for_regular(quad_rule_for_regular)
  {
    init_shape_value_tables();
    init_shape_grad_matrix_tables();
  }


  template <int dim, int spacedim, typename RangeNumberType>
  BEMValues<dim, spacedim, RangeNumberType>::BEMValues(
    const BEMValues<dim, spacedim, RangeNumberType> &bem_values)
    : kx_fe(bem_values.kx_fe)
    , ky_fe(bem_values.ky_fe)
    , quad_rule_for_same_panel(bem_values.quad_rule_for_same_panel)
    , quad_rule_for_common_edge(bem_values.quad_rule_for_common_edge)
    , quad_rule_for_common_vertex(bem_values.quad_rule_for_common_vertex)
    , quad_rule_for_regular(bem_values.quad_rule_for_regular)
  {
    kx_shape_value_table_for_same_panel =
      bem_values.kx_shape_value_table_for_same_panel;
    ky_shape_value_table_for_same_panel =
      bem_values.ky_shape_value_table_for_same_panel;

    kx_shape_value_table_for_common_edge =
      bem_values.kx_shape_value_table_for_common_edge;
    ky_shape_value_table_for_common_edge =
      bem_values.ky_shape_value_table_for_common_edge;

    kx_shape_value_table_for_common_vertex =
      bem_values.kx_shape_value_table_for_common_vertex;
    ky_shape_value_table_for_common_vertex =
      bem_values.ky_shape_value_table_for_common_vertex;

    kx_shape_value_table_for_regular =
      bem_values.kx_shape_value_table_for_regular;
    ky_shape_value_table_for_regular =
      bem_values.ky_shape_value_table_for_regular;

    kx_shape_grad_matrix_table_for_same_panel =
      bem_values.kx_shape_grad_matrix_table_for_same_panel;
    ky_shape_grad_matrix_table_for_same_panel =
      bem_values.ky_shape_grad_matrix_table_for_same_panel;

    kx_shape_grad_matrix_table_for_common_edge =
      bem_values.kx_shape_grad_matrix_table_for_common_edge;
    ky_shape_grad_matrix_table_for_common_edge =
      bem_values.ky_shape_grad_matrix_table_for_common_edge;

    kx_shape_grad_matrix_table_for_common_vertex =
      bem_values.kx_shape_grad_matrix_table_for_common_vertex;
    ky_shape_grad_matrix_table_for_common_vertex =
      bem_values.ky_shape_grad_matrix_table_for_common_vertex;

    kx_shape_grad_matrix_table_for_regular =
      bem_values.kx_shape_grad_matrix_table_for_regular;
    ky_shape_grad_matrix_table_for_regular =
      bem_values.ky_shape_grad_matrix_table_for_regular;
  }


  template <int dim, int spacedim, typename RangeNumberType>
  void
  BEMValues<dim, spacedim, RangeNumberType>::init_shape_value_tables()
  {
    kx_shape_value_table_for_same_panel.reinit(
      TableIndices<3>(kx_fe.dofs_per_cell, 8, quad_rule_for_same_panel.size()));
    ky_shape_value_table_for_same_panel.reinit(
      TableIndices<3>(ky_fe.dofs_per_cell, 8, quad_rule_for_same_panel.size()));

    kx_shape_value_table_for_common_edge.reinit(TableIndices<3>(
      kx_fe.dofs_per_cell, 6, quad_rule_for_common_edge.size()));
    ky_shape_value_table_for_common_edge.reinit(TableIndices<3>(
      ky_fe.dofs_per_cell, 6, quad_rule_for_common_edge.size()));

    kx_shape_value_table_for_common_vertex.reinit(TableIndices<3>(
      kx_fe.dofs_per_cell, 4, quad_rule_for_common_vertex.size()));
    ky_shape_value_table_for_common_vertex.reinit(TableIndices<3>(
      ky_fe.dofs_per_cell, 4, quad_rule_for_common_vertex.size()));

    kx_shape_value_table_for_regular.reinit(
      TableIndices<3>(kx_fe.dofs_per_cell, 1, quad_rule_for_regular.size()));
    ky_shape_value_table_for_regular.reinit(
      TableIndices<3>(ky_fe.dofs_per_cell, 1, quad_rule_for_regular.size()));
  }


  template <int dim, int spacedim, typename RangeNumberType>
  void
  BEMValues<dim, spacedim, RangeNumberType>::init_shape_grad_matrix_tables()
  {
    kx_shape_grad_matrix_table_for_same_panel.reinit(
      TableIndices<2>(8, quad_rule_for_same_panel.size()));
    ky_shape_grad_matrix_table_for_same_panel.reinit(
      TableIndices<2>(8, quad_rule_for_same_panel.size()));

    kx_shape_grad_matrix_table_for_common_edge.reinit(
      TableIndices<2>(6, quad_rule_for_common_edge.size()));
    ky_shape_grad_matrix_table_for_common_edge.reinit(
      TableIndices<2>(6, quad_rule_for_common_edge.size()));

    kx_shape_grad_matrix_table_for_common_vertex.reinit(
      TableIndices<2>(4, quad_rule_for_common_vertex.size()));
    ky_shape_grad_matrix_table_for_common_vertex.reinit(
      TableIndices<2>(4, quad_rule_for_common_vertex.size()));

    kx_shape_grad_matrix_table_for_regular.reinit(
      TableIndices<2>(1, quad_rule_for_regular.size()));
    ky_shape_grad_matrix_table_for_regular.reinit(
      TableIndices<2>(1, quad_rule_for_regular.size()));
  }


  template <int dim, int spacedim, typename RangeNumberType>
  void
  BEMValues<dim, spacedim, RangeNumberType>::fill_shape_value_tables()
  {
    bem_shape_values_same_panel(kx_fe,
                                ky_fe,
                                quad_rule_for_same_panel,
                                kx_shape_value_table_for_same_panel,
                                ky_shape_value_table_for_same_panel);

    bem_shape_values_common_edge(kx_fe,
                                 ky_fe,
                                 quad_rule_for_common_edge,
                                 kx_shape_value_table_for_common_edge,
                                 ky_shape_value_table_for_common_edge);

    bem_shape_values_common_vertex(kx_fe,
                                   ky_fe,
                                   quad_rule_for_common_vertex,
                                   kx_shape_value_table_for_common_vertex,
                                   ky_shape_value_table_for_common_vertex);

    bem_shape_values_regular(kx_fe,
                             ky_fe,
                             quad_rule_for_regular,
                             kx_shape_value_table_for_regular,
                             ky_shape_value_table_for_regular);
  }


  template <int dim, int spacedim, typename RangeNumberType>
  void
  BEMValues<dim, spacedim, RangeNumberType>::fill_shape_grad_matrix_tables()
  {
    bem_shape_grad_matrices_same_panel(
      kx_fe,
      ky_fe,
      quad_rule_for_same_panel,
      kx_shape_grad_matrix_table_for_same_panel,
      ky_shape_grad_matrix_table_for_same_panel);

    bem_shape_grad_matrices_common_edge(
      kx_fe,
      ky_fe,
      quad_rule_for_common_edge,
      kx_shape_grad_matrix_table_for_common_edge,
      ky_shape_grad_matrix_table_for_common_edge);

    bem_shape_grad_matrices_common_vertex(
      kx_fe,
      ky_fe,
      quad_rule_for_common_vertex,
      kx_shape_grad_matrix_table_for_common_vertex,
      ky_shape_grad_matrix_table_for_common_vertex);

    bem_shape_grad_matrices_regular(kx_fe,
                                    ky_fe,
                                    quad_rule_for_regular,
                                    kx_shape_grad_matrix_table_for_regular,
                                    ky_shape_grad_matrix_table_for_regular);
  }


  /**
   * Structure holding cell-wise local matrix data and DoF indices,
   * which is used for SMP parallel computation of the term $(v,
   * \frac{1}{2}u)$.
   */
  struct CellWisePerTaskData
  {
    FullMatrix<double> local_matrix;
    // N.B. Memory should be preallocated for this vector before calling
    // <code>get_dof_indices</code>.
    std::vector<types::global_dof_index> local_dof_indices;

    CellWisePerTaskData(const FiniteElement<2, 3> &fe)
      : local_matrix(fe.dofs_per_cell, fe.dofs_per_cell)
      , local_dof_indices(fe.dofs_per_cell)
    {}
  };


  /**
   * Structure holding temporary data which are needed for cell-wise
   * integration for the term $(v, \frac{1}{2}u)$.
   */
  struct CellWiseScratchData
  {
    FEValues<2, 3> fe_values;

    /**
     * Constructor for the structure.
     * @param fe
     * @param quadrature
     * @param update_flags
     */
    CellWiseScratchData(const FiniteElement<2, 3> &fe,
                        const Quadrature<2> &      quadrature,
                        const UpdateFlags          update_flags)
      : fe_values(fe, quadrature, update_flags)
    {}


    /**
     * Copy constructor for the structure. Because <code>FEValues</code> is
     * neither copyable nor has it copy constructor, this copy constructor
     * is mandatory for replication into each task.
     * @param scratch_data
     */
    CellWiseScratchData(const CellWiseScratchData &scratch_data)
      : fe_values(scratch_data.fe_values.get_fe(),
                  scratch_data.fe_values.get_quadrature(),
                  scratch_data.fe_values.get_update_flags())
    {}
  };


  struct PairCellWiseScratchData
  {
    std::vector<types::global_dof_index> vertex_dof_index_intersection;
    std::vector<Point<3>>                kx_support_points_hierarchical;
    std::vector<Point<3>>                ky_support_points_hierarchical;
    std::vector<Point<3>>                kx_support_points_permuted;
    std::vector<Point<3>>                ky_support_points_permuted;
    std::vector<types::global_dof_index> kx_local_dof_indices_hierarchical;
    std::vector<types::global_dof_index> ky_local_dof_indices_hierarchical;
    std::vector<unsigned int>            kx_fe_poly_space_numbering_inverse;
    std::vector<unsigned int>            ky_fe_poly_space_numbering_inverse;
    std::vector<unsigned int> ky_fe_reversed_poly_space_numbering_inverse;
    std::vector<unsigned int> kx_local_dof_permutation;
    std::vector<unsigned int> ky_local_dof_permutation;

    Table<2, double>       kx_jacobians_same_panel;
    Table<2, double>       kx_jacobians_common_edge;
    Table<2, double>       kx_jacobians_common_vertex;
    Table<2, double>       kx_jacobians_regular;
    Table<2, Tensor<1, 3>> kx_normals_same_panel;
    Table<2, Tensor<1, 3>> kx_normals_common_edge;
    Table<2, Tensor<1, 3>> kx_normals_common_vertex;
    Table<2, Tensor<1, 3>> kx_normals_regular;
    Table<2, Point<3>>     kx_quad_points_same_panel;
    Table<2, Point<3>>     kx_quad_points_common_edge;
    Table<2, Point<3>>     kx_quad_points_common_vertex;
    Table<2, Point<3>>     kx_quad_points_regular;


    Table<2, double>       ky_jacobians_same_panel;
    Table<2, double>       ky_jacobians_common_edge;
    Table<2, double>       ky_jacobians_common_vertex;
    Table<2, double>       ky_jacobians_regular;
    Table<2, Tensor<1, 3>> ky_normals_same_panel;
    Table<2, Tensor<1, 3>> ky_normals_common_edge;
    Table<2, Tensor<1, 3>> ky_normals_common_vertex;
    Table<2, Tensor<1, 3>> ky_normals_regular;
    Table<2, Point<3>>     ky_quad_points_same_panel;
    Table<2, Point<3>>     ky_quad_points_common_edge;
    Table<2, Point<3>>     ky_quad_points_common_vertex;
    Table<2, Point<3>>     ky_quad_points_regular;

    PairCellWiseScratchData(const FiniteElement<2, 3> &kx_fe,
                            const FiniteElement<2, 3> &ky_fe,
                            const BEMValues<2, 3> &    bem_values)
      : vertex_dof_index_intersection(0)
      , kx_support_points_hierarchical(kx_fe.dofs_per_cell)
      , ky_support_points_hierarchical(ky_fe.dofs_per_cell)
      , kx_support_points_permuted(kx_fe.dofs_per_cell)
      , ky_support_points_permuted(ky_fe.dofs_per_cell)
      , kx_local_dof_indices_hierarchical(kx_fe.dofs_per_cell)
      , ky_local_dof_indices_hierarchical(ky_fe.dofs_per_cell)
      , kx_fe_poly_space_numbering_inverse(kx_fe.dofs_per_cell)
      , ky_fe_poly_space_numbering_inverse(ky_fe.dofs_per_cell)
      , ky_fe_reversed_poly_space_numbering_inverse(ky_fe.dofs_per_cell)
      , kx_local_dof_permutation(kx_fe.dofs_per_cell)
      , ky_local_dof_permutation(ky_fe.dofs_per_cell)
      , kx_jacobians_same_panel(8, bem_values.quad_rule_for_same_panel.size())
      , kx_jacobians_common_edge(6, bem_values.quad_rule_for_common_edge.size())
      , kx_jacobians_common_vertex(
          4,
          bem_values.quad_rule_for_common_vertex.size())
      , kx_jacobians_regular(1, bem_values.quad_rule_for_regular.size())
      , kx_normals_same_panel(8, bem_values.quad_rule_for_same_panel.size())
      , kx_normals_common_edge(6, bem_values.quad_rule_for_common_edge.size())
      , kx_normals_common_vertex(4,
                                 bem_values.quad_rule_for_common_vertex.size())
      , kx_normals_regular(1, bem_values.quad_rule_for_regular.size())
      , kx_quad_points_same_panel(8, bem_values.quad_rule_for_same_panel.size())
      , kx_quad_points_common_edge(6,
                                   bem_values.quad_rule_for_common_edge.size())
      , kx_quad_points_common_vertex(
          4,
          bem_values.quad_rule_for_common_vertex.size())
      , kx_quad_points_regular(1, bem_values.quad_rule_for_regular.size())
      , ky_jacobians_same_panel(8, bem_values.quad_rule_for_same_panel.size())
      , ky_jacobians_common_edge(6, bem_values.quad_rule_for_common_edge.size())
      , ky_jacobians_common_vertex(
          4,
          bem_values.quad_rule_for_common_vertex.size())
      , ky_jacobians_regular(1, bem_values.quad_rule_for_regular.size())
      , ky_normals_same_panel(8, bem_values.quad_rule_for_same_panel.size())
      , ky_normals_common_edge(6, bem_values.quad_rule_for_common_edge.size())
      , ky_normals_common_vertex(4,
                                 bem_values.quad_rule_for_common_vertex.size())
      , ky_normals_regular(1, bem_values.quad_rule_for_regular.size())
      , ky_quad_points_same_panel(8, bem_values.quad_rule_for_same_panel.size())
      , ky_quad_points_common_edge(6,
                                   bem_values.quad_rule_for_common_edge.size())
      , ky_quad_points_common_vertex(
          4,
          bem_values.quad_rule_for_common_vertex.size())
      , ky_quad_points_regular(1, bem_values.quad_rule_for_regular.size())
    {
      vertex_dof_index_intersection.reserve(GeometryInfo<2>::vertices_per_cell);

      // Downcast references of FiniteElement objects to FE_Poly references
      // for obtaining the inverse polynomial space numbering.
      using FE_Poly             = FE_Poly<TensorProductPolynomials<2>, 2, 3>;
      const FE_Poly &kx_fe_poly = dynamic_cast<const FE_Poly &>(kx_fe);
      const FE_Poly &ky_fe_poly = dynamic_cast<const FE_Poly &>(ky_fe);
      kx_fe_poly_space_numbering_inverse =
        kx_fe_poly.get_poly_space_numbering_inverse();
      ky_fe_poly_space_numbering_inverse =
        ky_fe_poly.get_poly_space_numbering_inverse();
      generate_backward_dof_permutation(
        ky_fe, 0, ky_fe_reversed_poly_space_numbering_inverse);
    }
  };


  struct PairCellWisePerTaskData
  {
    FullMatrix<double>                   dlp_matrix;
    FullMatrix<double>                   slp_matrix;
    std::vector<types::global_dof_index> kx_local_dof_indices_permuted;
    std::vector<types::global_dof_index> ky_local_dof_indices_permuted;

    PairCellWisePerTaskData(const FiniteElement<2, 3> &kx_fe,
                            const FiniteElement<2, 3> &ky_fe)
      : dlp_matrix(kx_fe.dofs_per_cell, ky_fe.dofs_per_cell)
      , slp_matrix(kx_fe.dofs_per_cell, ky_fe.dofs_per_cell)
      , kx_local_dof_indices_permuted(kx_fe.dofs_per_cell)
      , ky_local_dof_indices_permuted(ky_fe.dofs_per_cell)
    {}
  };


  namespace LaplaceKernel
  {
    enum KernelType
    {
      SingleLayer,
      DoubleLayer,
      AdjointDoubleLayer,
      HyperSingular,
      NoneType
    };


    // Base class for a BEM Laplace kernel function.
    template <int dim, typename RangeNumberType = double>
    class KernelFunction : public Subscriptor
    {
    public:
      static const unsigned int dimension = dim;
      const KernelType          kernel_type;
      const unsigned int        n_components;

      KernelFunction(const KernelType   kernel_type  = NoneType,
                     const unsigned int n_components = 1);
      virtual ~KernelFunction() override = 0;

      KernelFunction &
      operator=(const KernelFunction &f);

      virtual RangeNumberType
      value(const Point<dim> &    x,
            const Point<dim> &    y,
            const Tensor<1, dim> &nx,
            const Tensor<1, dim> &ny,
            const unsigned int    component = 0) const;

      virtual void
      vector_value(const Point<dim> &       x,
                   const Point<dim> &       y,
                   const Tensor<1, dim> &   nx,
                   const Tensor<1, dim> &   ny,
                   Vector<RangeNumberType> &values) const;

      virtual void
      value_list(const std::vector<Point<dim>> &    x_points,
                 const std::vector<Point<dim>> &    y_points,
                 const std::vector<Tensor<1, dim>> &nx_list,
                 const std::vector<Tensor<1, dim>> &ny_list,
                 std::vector<RangeNumberType> &     values,
                 const unsigned int                 component = 0) const;

      virtual void
      vector_value_list(const std::vector<Point<dim>> &       x_points,
                        const std::vector<Point<dim>> &       y_points,
                        const std::vector<Tensor<1, dim>> &   nx_list,
                        const std::vector<Tensor<1, dim>> &   ny_list,
                        std::vector<Vector<RangeNumberType>> &values) const;
    };


    // Expose the static member variable outside the class body.
    template <int dim, typename RangeNumberType>
    const unsigned int KernelFunction<dim, RangeNumberType>::dimension;


    template <int dim, typename RangeNumberType>
    KernelFunction<dim, RangeNumberType>::KernelFunction(
      const KernelType   kernel_type,
      const unsigned int n_components)
      : kernel_type(kernel_type)
      , n_components(n_components)
    {
      Assert(kernel_type != NoneType, ExcInternalError());
      Assert(n_components > 0, ExcZero());
    }


    // Explicitly provide a default definition of the destructor of the class.
    template <int dim, typename RangeNumberType>
    inline KernelFunction<dim, RangeNumberType>::~KernelFunction() = default;


    template <int dim, typename RangeNumberType>
    KernelFunction<dim, RangeNumberType> &
    KernelFunction<dim, RangeNumberType>::operator=(const KernelFunction &f)
    {
      // As a pure base class, it does nothing here but only assert the number
      // of components. The following sentence suppresses compiler warnings
      // about the unused input argument f.
      (void)f;
      AssertDimension(n_components, f.n_components);
      return *this;
    }


    template <int dim, typename RangeNumberType>
    RangeNumberType
    KernelFunction<dim, RangeNumberType>::value(const Point<dim> &,
                                                const Point<dim> &,
                                                const Tensor<1, dim> &,
                                                const Tensor<1, dim> &,
                                                const unsigned int) const
    {
      // Member function of the pure abstract base class should not be called.
      Assert(false, ExcPureFunctionCalled());
      return 0;
    }


    // This is the default behavior of value_list, which usually needs not be
    // re-implemented in child classes.
    template <int dim, typename RangeNumberType>
    void
    KernelFunction<dim, RangeNumberType>::value_list(
      const std::vector<Point<dim>> &    x_points,
      const std::vector<Point<dim>> &    y_points,
      const std::vector<Tensor<1, dim>> &nx_list,
      const std::vector<Tensor<1, dim>> &ny_list,
      std::vector<RangeNumberType> &     values,
      const unsigned int                 component) const
    {
      Assert(values.size() == x_points.size(),
             ExcDimensionMismatch(values.size(), x_points.size()));
      Assert(values.size() == y_points.size(),
             ExcDimensionMismatch(values.size(), y_points.size()));
      Assert(values.size() == nx_list.size(),
             ExcDimensionMismatch(values.size(), nx_list.size()));
      Assert(values.size() == ny_list.size(),
             ExcDimensionMismatch(values.size(), ny_list.size()));

      for (unsigned int i = 0; i < x_points.size(); i++)
        {
          values[i] = this->value(
            x_points[i], y_points[i], nx_list[i], ny_list[i], component);
        }
    }


    // This is the default behavior of vector_value, which usually needs not be
    // re-implemented in child classes.
    template <int dim, typename RangeNumberType>
    void
    KernelFunction<dim, RangeNumberType>::vector_value(
      const Point<dim> &       x,
      const Point<dim> &       y,
      const Tensor<1, dim> &   nx,
      const Tensor<1, dim> &   ny,
      Vector<RangeNumberType> &values) const
    {
      AssertDimension(values.size(), this->n_components);

      for (unsigned int i = 0; i < this->n_components; i++)
        {
          values(i) = this->value(x, y, nx, ny, i);
        }
    }


    // This is the default behavior of vector_value_list, which usually needs
    // not be re-implemented in child classes.
    template <int dim, typename RangeNumberType>
    void
    KernelFunction<dim, RangeNumberType>::vector_value_list(
      const std::vector<Point<dim>> &       x_points,
      const std::vector<Point<dim>> &       y_points,
      const std::vector<Tensor<1, dim>> &   nx_list,
      const std::vector<Tensor<1, dim>> &   ny_list,
      std::vector<Vector<RangeNumberType>> &values) const
    {
      Assert(values.size() == x_points.size(),
             ExcDimensionMismatch(values.size(), x_points.size()));
      Assert(values.size() == y_points.size(),
             ExcDimensionMismatch(values.size(), y_points.size()));
      Assert(values.size() == nx_list.size(),
             ExcDimensionMismatch(values.size(), nx_list.size()));
      Assert(values.size() == ny_list.size(),
             ExcDimensionMismatch(values.size(), ny_list.size()));

      for (unsigned int i = 0; i < x_points.size(); i++)
        {
          this->vector_value(
            x_points[i], y_points[i], nx_list[i], ny_list[i], values[i]);
        }
    }


    /**
     * Single layer kernel.
     */
    template <int dim, typename RangeNumberType = double>
    class SingleLayerKernel : public KernelFunction<dim, RangeNumberType>
    {
    public:
      SingleLayerKernel()
        : KernelFunction<dim, RangeNumberType>(SingleLayer)
      {}

      virtual RangeNumberType
      value(const Point<dim> &    x,
            const Point<dim> &    y,
            const Tensor<1, dim> &nx,
            const Tensor<1, dim> &ny,
            const unsigned int    component = 0) const override;
    };


    template <int dim, typename RangeNumberType>
    RangeNumberType
    SingleLayerKernel<dim, RangeNumberType>::value(
      const Point<dim> &x,
      const Point<dim> &y,
      const Tensor<1, dim> & /* nx */,
      const Tensor<1, dim> & /* ny */,
      const unsigned int /* component */) const
    {
      switch (dim)
        {
          case 2:
            return (-0.5 / numbers::PI * std::log(1.0 / (x - y).norm()));

          case 3:
            return (0.25 / numbers::PI / (x - y).norm());

          default:
            Assert(false, ExcInternalError());
            return 0.;
        }
    }


    /**
     * Double layer kernel.
     */
    template <int dim, typename RangeNumberType = double>
    class DoubleLayerKernel : public KernelFunction<dim, RangeNumberType>
    {
    public:
      DoubleLayerKernel()
        : KernelFunction<dim, RangeNumberType>(DoubleLayer)
      {}

      virtual RangeNumberType
      value(const Point<dim> &    x,
            const Point<dim> &    y,
            const Tensor<1, dim> &nx,
            const Tensor<1, dim> &ny,
            const unsigned int    component = 0) const override;
    };


    template <int dim, typename RangeNumberType>
    RangeNumberType
    DoubleLayerKernel<dim, RangeNumberType>::value(
      const Point<dim> &x,
      const Point<dim> &y,
      const Tensor<1, dim> & /* nx */,
      const Tensor<1, dim> &ny,
      const unsigned int /* component */) const
    {
      switch (dim)
        {
          case 2:
            return ((y - x) * ny) / 2.0 / numbers::PI / (y - x).norm_square();

          case 3:
            return ((x - y) * ny) / 4.0 / numbers::PI /
                   std::pow((x - y).norm(), 3.0);

          default:
            Assert(false, ExcInternalError());
            return 0.;
        }
    }


    // Class for the adjoint double layer kernel.
    template <int dim, typename RangeNumberType = double>
    class AdjointDoubleLayerKernel : public KernelFunction<dim, RangeNumberType>
    {
    public:
      AdjointDoubleLayerKernel()
        : KernelFunction<dim, RangeNumberType>(AdjointDoubleLayer)
      {}

      virtual RangeNumberType
      value(const Point<dim> &    x,
            const Point<dim> &    y,
            const Tensor<1, dim> &nx,
            const Tensor<1, dim> &ny,
            const unsigned int    component = 0) const override;
    };


    template <int dim, typename RangeNumberType>
    RangeNumberType
    AdjointDoubleLayerKernel<dim, RangeNumberType>::value(
      const Point<dim> &    x,
      const Point<dim> &    y,
      const Tensor<1, dim> &nx,
      const Tensor<1, dim> & /* ny */,
      const unsigned int /* component */) const
    {
      switch (dim)
        {
          case 2:
            return ((x - y) * nx) / 2.0 / numbers::PI / (x - y).norm_square();

          case 3:
            return ((y - x) * nx) / 4.0 / numbers::PI /
                   std::pow((x - y).norm(), 3.0);

          default:
            Assert(false, ExcInternalError());
            return 0.;
        }
    }


    // Class for the hyper singular kernel.
    template <int dim, typename RangeNumberType = double>
    class HyperSingularKernel : public KernelFunction<dim, RangeNumberType>
    {
    public:
      HyperSingularKernel()
        : KernelFunction<dim, RangeNumberType>(HyperSingular)
      {}

      virtual RangeNumberType
      value(const Point<dim> &    x,
            const Point<dim> &    y,
            const Tensor<1, dim> &nx,
            const Tensor<1, dim> &ny,
            const unsigned int    component = 0) const override;
    };


    template <int dim, typename RangeNumberType>
    RangeNumberType
    HyperSingularKernel<dim, RangeNumberType>::value(
      const Point<dim> &    x,
      const Point<dim> &    y,
      const Tensor<1, dim> &nx,
      const Tensor<1, dim> &ny,
      const unsigned int /* component */) const
    {
      double r2 = (x - y).norm_square();

      switch (dim)
        {
          case 2:
            {
              double r4 = r2 * r2;

              return 0.5 / numbers::PI *
                     (nx * ny / r2 -
                      2.0 * (nx * (y - x)) * (ny * (y - x)) / r4);
            }

          case 3:
            {
              double r3 = (x - y).norm() * r2;
              double r5 = r2 * r3;

              return 0.25 / numbers::PI *
                     (nx * ny / r3 -
                      3.0 * (nx * (x - y)) * (ny * (x - y)) / r5);
            }

          default:
            {
              Assert(false, ExcInternalError());
              return 0.;
            }
        }
    }
  } // namespace LaplaceKernel


  /**
   * Kernel function pulled back to unit cell.
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  class KernelPulledbackToUnitCell : public Subscriptor
  {
  public:
    static const unsigned int dimension = dim;
    const unsigned int        n_components;


    /**
     * Constructor for KernelPulledbackToUnitCell.
     *
     * @param kx_support_points Permuted list of support points in $K_x$.
     * @param ky_support_points Permuted list of support points in $K_y$.
     * @param kx_dof_index Index for accessing the list of DoFs in
     * tensor product order in $K_x$.
     * @param ky_dof_index Index for accessing the list of DoFs in
     * tensor product order in $K_y$.
     */
    KernelPulledbackToUnitCell(
      const LaplaceKernel::KernelFunction<spacedim, RangeNumberType>
        &                                 kernel_function,
      const CellNeighboringType &         cell_neighboring_type,
      const std::vector<Point<spacedim>> &kx_support_points,
      const std::vector<Point<spacedim>> &ky_support_points,
      const FiniteElement<dim, spacedim> &kx_fe,
      const FiniteElement<dim, spacedim> &ky_fe,
      const unsigned int                  kx_dof_index = 0,
      const unsigned int                  ky_dof_index = 0);


    /**
     *
     * @param kernel_function
     * @param cell_neighboring_type
     * @param kx_support_points
     * @param ky_support_points
     * @param kx_fe
     * @param ky_fe
     * @param bem_values
     * @param kx_dof_index
     * @param ky_dof_index
     */
    KernelPulledbackToUnitCell(
      const LaplaceKernel::KernelFunction<spacedim, RangeNumberType>
        &                                              kernel_function,
      const CellNeighboringType &                      cell_neighboring_type,
      const std::vector<Point<spacedim>> &             kx_support_points,
      const std::vector<Point<spacedim>> &             ky_support_points,
      const FiniteElement<dim, spacedim> &             kx_fe,
      const FiniteElement<dim, spacedim> &             ky_fe,
      const BEMValues<dim, spacedim, RangeNumberType> *bem_values,
      const unsigned int                               kx_dof_index = 0,
      const unsigned int                               ky_dof_index = 0);


    /**
     *
     * @param kernel_function
     * @param cell_neighboring_type
     * @param kx_support_points
     * @param ky_support_points
     * @param kx_fe
     * @param ky_fe
     * @param bem_values
     * @param kx_dof_index
     * @param ky_dof_index
     */
    KernelPulledbackToUnitCell(
      const LaplaceKernel::KernelFunction<spacedim, RangeNumberType>
        &                                              kernel_function,
      const CellNeighboringType &                      cell_neighboring_type,
      const std::vector<Point<spacedim>> &             kx_support_points,
      const std::vector<Point<spacedim>> &             ky_support_points,
      const FiniteElement<dim, spacedim> &             kx_fe,
      const FiniteElement<dim, spacedim> &             ky_fe,
      const BEMValues<dim, spacedim, RangeNumberType> *bem_values,
      const PairCellWiseScratchData *                  scratch,
      const unsigned int                               kx_dof_index = 0,
      const unsigned int                               ky_dof_index = 0);


    virtual ~KernelPulledbackToUnitCell();


    KernelPulledbackToUnitCell &
    operator=(const KernelPulledbackToUnitCell &f);


    /**
     * Associate the KernelPulledbackToUnitCell with new support point and
     * finite element data.
     */
    void
    reinit(const CellNeighboringType &         cell_neighboring_type,
           const std::vector<Point<spacedim>> &kx_support_points,
           const std::vector<Point<spacedim>> &ky_support_points,
           const FiniteElement<dim, spacedim> &kx_fe,
           const FiniteElement<dim, spacedim> &ky_fe);


    void
    set_kx_dof_index(const unsigned int kx_dof_index);
    void
    set_ky_dof_index(const unsigned int ky_dof_index);


    /**
     * Evaluation of the function which depends on the kernel type. Different
     * types of kernel function require different normal vector data.
     */
    virtual RangeNumberType
    value(const Point<dim> & x_hat,
          const Point<dim> & y_hat,
          const unsigned int component = 0) const;


    virtual RangeNumberType
    value(const unsigned int k3_index,
          const unsigned int quad_no,
          const unsigned int component = 0) const;


    virtual void
    vector_value(const Point<dim> &       x_hat,
                 const Point<dim> &       y_hat,
                 Vector<RangeNumberType> &values) const;


    virtual void
    value_list(const std::vector<Point<dim>> &x_hat_list,
               const std::vector<Point<dim>> &y_hat_list,
               std::vector<RangeNumberType> & values,
               const unsigned int             component = 0) const;


    virtual void
    vector_value_list(const std::vector<Point<dim>> &       x_hat_list,
                      const std::vector<Point<dim>> &       y_hat_list,
                      std::vector<Vector<RangeNumberType>> &values) const;


  private:
    const LaplaceKernel::KernelFunction<spacedim, RangeNumberType>
      &kernel_function;

    CellNeighboringType cell_neighboring_type;

    const std::vector<Point<spacedim>> &kx_support_points;
    const std::vector<Point<spacedim>> &ky_support_points;

    const FiniteElement<dim, spacedim> &kx_fe;
    const FiniteElement<dim, spacedim> &ky_fe;

    const BEMValues<dim, spacedim, RangeNumberType> *bem_values;
    const PairCellWiseScratchData *                  scratch;

    /**
     * Index for accessing the list of DoFs in tensor product order in $K_x$.
     */
    unsigned int kx_dof_index;
    /**
     * Index for accessing the list of DoFs in tensor product order in $K_y$.
     */
    unsigned int ky_dof_index;
  };


  template <int dim, int spacedim, typename RangeNumberType>
  const unsigned int
    KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType>::dimension;


  template <int dim, int spacedim, typename RangeNumberType>
  KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType>::
    KernelPulledbackToUnitCell(
      const LaplaceKernel::KernelFunction<spacedim, RangeNumberType>
        &                                 kernel_function,
      const CellNeighboringType &         cell_neighboring_type,
      const std::vector<Point<spacedim>> &kx_support_points,
      const std::vector<Point<spacedim>> &ky_support_points,
      const FiniteElement<dim, spacedim> &kx_fe,
      const FiniteElement<dim, spacedim> &ky_fe,
      const unsigned int                  kx_dof_index,
      const unsigned int                  ky_dof_index)
    : n_components(kernel_function.n_components)
    , kernel_function(kernel_function)
    , cell_neighboring_type(cell_neighboring_type)
    , kx_support_points(kx_support_points)
    , ky_support_points(ky_support_points)
    , kx_fe(kx_fe)
    , ky_fe(ky_fe)
    , bem_values(nullptr)
    , scratch(nullptr)
    , kx_dof_index(kx_dof_index)
    , ky_dof_index(ky_dof_index)
  {}


  template <int dim, int spacedim, typename RangeNumberType>
  KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType>::
    KernelPulledbackToUnitCell(
      const LaplaceKernel::KernelFunction<spacedim, RangeNumberType>
        &                                              kernel_function,
      const CellNeighboringType &                      cell_neighboring_type,
      const std::vector<Point<spacedim>> &             kx_support_points,
      const std::vector<Point<spacedim>> &             ky_support_points,
      const FiniteElement<dim, spacedim> &             kx_fe,
      const FiniteElement<dim, spacedim> &             ky_fe,
      const BEMValues<dim, spacedim, RangeNumberType> *bem_values,
      const unsigned int                               kx_dof_index,
      const unsigned int                               ky_dof_index)
    : n_components(kernel_function.n_components)
    , kernel_function(kernel_function)
    , cell_neighboring_type(cell_neighboring_type)
    , kx_support_points(kx_support_points)
    , ky_support_points(ky_support_points)
    , kx_fe(kx_fe)
    , ky_fe(ky_fe)
    , bem_values(bem_values)
    , scratch(nullptr)
    , kx_dof_index(kx_dof_index)
    , ky_dof_index(ky_dof_index)
  {}


  template <int dim, int spacedim, typename RangeNumberType>
  KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType>::
    KernelPulledbackToUnitCell(
      const LaplaceKernel::KernelFunction<spacedim, RangeNumberType>
        &                                              kernel_function,
      const CellNeighboringType &                      cell_neighboring_type,
      const std::vector<Point<spacedim>> &             kx_support_points,
      const std::vector<Point<spacedim>> &             ky_support_points,
      const FiniteElement<dim, spacedim> &             kx_fe,
      const FiniteElement<dim, spacedim> &             ky_fe,
      const BEMValues<dim, spacedim, RangeNumberType> *bem_values,
      const PairCellWiseScratchData *                  scratch,
      const unsigned int                               kx_dof_index,
      const unsigned int                               ky_dof_index)
    : n_components(kernel_function.n_components)
    , kernel_function(kernel_function)
    , cell_neighboring_type(cell_neighboring_type)
    , kx_support_points(kx_support_points)
    , ky_support_points(ky_support_points)
    , kx_fe(kx_fe)
    , ky_fe(ky_fe)
    , bem_values(bem_values)
    , scratch(scratch)
    , kx_dof_index(kx_dof_index)
    , ky_dof_index(ky_dof_index)
  {}


  template <int dim, int spacedim, typename RangeNumberType>
  KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType>::
    ~KernelPulledbackToUnitCell() = default;


  template <int dim, int spacedim, typename RangeNumberType>
  KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType> &
  KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType>::
  operator=(const KernelPulledbackToUnitCell &f)
  {
    AssertDimension(n_components, f.n_components);

    kernel_function       = f.kernel_function;
    cell_neighboring_type = f.cell_neighboring_type;
    kx_support_points     = f.kx_support_points;
    ky_support_points     = f.ky_support_points;
    kx_fe                 = f.kx_fe;
    ky_fe                 = f.ky_fe;

    return *this;
  }


  template <int dim, int spacedim, typename RangeNumberType>
  void
  KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType>::reinit(
    const CellNeighboringType &         cell_neighboring_type,
    const std::vector<Point<spacedim>> &kx_support_points,
    const std::vector<Point<spacedim>> &ky_support_points,
    const FiniteElement<dim, spacedim> &kx_fe,
    const FiniteElement<dim, spacedim> &ky_fe)
  {
    this->cell_neighboring_type = cell_neighboring_type;
    this->kx_support_points     = kx_support_points;
    this->ky_support_points     = ky_support_points;
    this->kx_fe                 = kx_fe;
    this->ky_fe                 = ky_fe;
  }


  template <int dim, int spacedim, typename RangeNumberType>
  void
  KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType>::set_kx_dof_index(
    const unsigned int kx_dof_index)
  {
    this->kx_dof_index = kx_dof_index;
  }


  template <int dim, int spacedim, typename RangeNumberType>
  void
  KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType>::set_ky_dof_index(
    const unsigned int ky_dof_index)
  {
    this->ky_dof_index = ky_dof_index;
  }


  template <int dim, int spacedim, typename RangeNumberType>
  RangeNumberType
  KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType>::value(
    const Point<dim> & x_hat,
    const Point<dim> & y_hat,
    const unsigned int component) const
  {
    Assert(dim == 2 && spacedim == 3, ExcNotImplemented());

    Point<spacedim> x =
      transform_unit_to_permuted_real_cell(kx_fe, kx_support_points, x_hat);
    Point<spacedim> y =
      transform_unit_to_permuted_real_cell(ky_fe, ky_support_points, y_hat);
    RangeNumberType     Jx = 0;
    RangeNumberType     Jy = 0;
    Tensor<1, spacedim> nx, ny;

    switch (kernel_function.kernel_type)
      {
        case LaplaceKernel::SingleLayer:
          Jx = surface_jacobian_det(kx_fe, kx_support_points, x_hat);
          Jy = surface_jacobian_det(ky_fe, ky_support_points, y_hat);

          break;
        case LaplaceKernel::DoubleLayer:
          Jx = surface_jacobian_det(kx_fe, kx_support_points, x_hat);
          Jy = surface_jacobian_det_and_normal_vector(ky_fe,
                                                      ky_support_points,
                                                      y_hat,
                                                      ny);

          // Negate the normal vector in $K_y$.
          if (cell_neighboring_type == CommonEdge)
            {
              ny = -ny;
            }

          break;
        case LaplaceKernel::AdjointDoubleLayer:
          Jx = surface_jacobian_det_and_normal_vector(kx_fe,
                                                      kx_support_points,
                                                      x_hat,
                                                      nx);
          Jy = surface_jacobian_det(ky_fe, ky_support_points, y_hat);

          break;
        case LaplaceKernel::HyperSingular:
          Jx = surface_jacobian_det_and_normal_vector(kx_fe,
                                                      kx_support_points,
                                                      x_hat,
                                                      nx);
          Jy = surface_jacobian_det_and_normal_vector(ky_fe,
                                                      ky_support_points,
                                                      y_hat,
                                                      ny);

          // Negate the normal vector in $K_y$.
          if (cell_neighboring_type == CommonEdge)
            {
              ny = -ny;
            }

          break;
        case LaplaceKernel::NoneType:
          Assert(false, ExcInternalError());
          break;
        default:
          Assert(false, ExcInternalError());
          break;
      }

    try
      {
        using FE_Poly = FE_Poly<TensorProductPolynomials<dim>, dim, spacedim>;
        const FE_Poly &kx_fe_poly = dynamic_cast<const FE_Poly &>(kx_fe);
        const FE_Poly &ky_fe_poly = dynamic_cast<const FE_Poly &>(ky_fe);

        std::vector<unsigned int> kx_poly_space_inverse_numbering =
          kx_fe_poly.get_poly_space_numbering_inverse();
        std::vector<unsigned int> ky_poly_space_inverse_numbering =
          ky_fe_poly.get_poly_space_numbering_inverse();

        //        // DEBUG
        //        deallog
        //          << kernel_function.value(x, y, nx, ny, component) << "," <<
        //          Jx << ","
        //          << Jy << ","
        //          <<
        //          kx_fe.shape_value(kx_poly_space_inverse_numbering[kx_dof_index],
        //                               x_hat)
        //          << ","
        //          <<
        //          ky_fe.shape_value(ky_poly_space_inverse_numbering[ky_dof_index],
        //                               y_hat)
        //          << std::endl;

        return kernel_function.value(x, y, nx, ny, component) * Jx * Jy *
               kx_fe.shape_value(kx_poly_space_inverse_numbering[kx_dof_index],
                                 x_hat) *
               ky_fe.shape_value(ky_poly_space_inverse_numbering[ky_dof_index],
                                 y_hat);
      }
    catch (const std::bad_cast &e)
      {
        Assert(false, ExcInternalError());

        return 0.;
      }
  }


  //


  template <int dim, int spacedim, typename RangeNumberType>
  RangeNumberType
  KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType>::value(
    const unsigned int k3_index,
    const unsigned int quad_no,
    const unsigned int component) const
  {
    Assert(dim == 2 && spacedim == 3, ExcNotImplemented());

    const Table<3, RangeNumberType> *kx_shape_value_table;
    const Table<3, RangeNumberType> *ky_shape_value_table;

    Point<spacedim>     x, y;
    RangeNumberType     Jx = 0;
    RangeNumberType     Jy = 0;
    Tensor<1, spacedim> nx, ny;

    // Select shape function value table according to the cell neighboring type.
    switch (cell_neighboring_type)
      {
        case SamePanel:
          kx_shape_value_table =
            &(bem_values->kx_shape_value_table_for_same_panel);
          ky_shape_value_table =
            &(bem_values->ky_shape_value_table_for_same_panel);

          x  = scratch->kx_quad_points_same_panel(k3_index, quad_no);
          y  = scratch->ky_quad_points_same_panel(k3_index, quad_no);
          Jx = scratch->kx_jacobians_same_panel(k3_index, quad_no);
          Jy = scratch->ky_jacobians_same_panel(k3_index, quad_no);
          nx = scratch->kx_normals_same_panel(k3_index, quad_no);
          ny = scratch->ky_normals_same_panel(k3_index, quad_no);

          break;
        case CommonEdge:
          kx_shape_value_table =
            &(bem_values->kx_shape_value_table_for_common_edge);
          ky_shape_value_table =
            &(bem_values->ky_shape_value_table_for_common_edge);

          x  = scratch->kx_quad_points_common_edge(k3_index, quad_no);
          y  = scratch->ky_quad_points_common_edge(k3_index, quad_no);
          Jx = scratch->kx_jacobians_common_edge(k3_index, quad_no);
          Jy = scratch->ky_jacobians_common_edge(k3_index, quad_no);
          nx = scratch->kx_normals_common_edge(k3_index, quad_no);
          ny = scratch->ky_normals_common_edge(k3_index, quad_no);

          break;
        case CommonVertex:
          kx_shape_value_table =
            &(bem_values->kx_shape_value_table_for_common_vertex);
          ky_shape_value_table =
            &(bem_values->ky_shape_value_table_for_common_vertex);

          x  = scratch->kx_quad_points_common_vertex(k3_index, quad_no);
          y  = scratch->ky_quad_points_common_vertex(k3_index, quad_no);
          Jx = scratch->kx_jacobians_common_vertex(k3_index, quad_no);
          Jy = scratch->ky_jacobians_common_vertex(k3_index, quad_no);
          nx = scratch->kx_normals_common_vertex(k3_index, quad_no);
          ny = scratch->ky_normals_common_vertex(k3_index, quad_no);

          break;
        case Regular:
          kx_shape_value_table =
            &(bem_values->kx_shape_value_table_for_regular);
          ky_shape_value_table =
            &(bem_values->ky_shape_value_table_for_regular);

          x  = scratch->kx_quad_points_regular(k3_index, quad_no);
          y  = scratch->ky_quad_points_regular(k3_index, quad_no);
          Jx = scratch->kx_jacobians_regular(k3_index, quad_no);
          Jy = scratch->ky_jacobians_regular(k3_index, quad_no);
          nx = scratch->kx_normals_regular(k3_index, quad_no);
          ny = scratch->ky_normals_regular(k3_index, quad_no);

          break;
        default:
          kx_shape_value_table = nullptr;
          ky_shape_value_table = nullptr;

          Assert(false, ExcInternalError());
      }

    // Negate the normal vector in $K_y$.
    if (cell_neighboring_type == CommonEdge)
      {
        ny = -ny;
      }

    return kernel_function.value(x, y, nx, ny, component) * Jx * Jy *
           (*kx_shape_value_table)(kx_dof_index, k3_index, quad_no) *
           (*ky_shape_value_table)(ky_dof_index, k3_index, quad_no);
  }


  template <int dim, int spacedim, typename RangeNumberType>
  void
  KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType>::value_list(
    const std::vector<Point<dim>> &x_hat_list,
    const std::vector<Point<dim>> &y_hat_list,
    std::vector<RangeNumberType> & values,
    const unsigned int             component) const
  {
    Assert(values.size() == x_hat_list.size(),
           ExcDimensionMismatch(values.size(), x_hat_list.size()));
    Assert(values.size() == y_hat_list.size(),
           ExcDimensionMismatch(values.size(), y_hat_list.size()));

    for (unsigned int i = 0; i < x_hat_list.size(); i++)
      {
        values[i] = this->value(x_hat_list[i], y_hat_list[i], component);
      }
  }


  template <int dim, int spacedim, typename RangeNumberType>
  void
  KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType>::vector_value(
    const Point<dim> &       x_hat,
    const Point<dim> &       y_hat,
    Vector<RangeNumberType> &values) const
  {
    AssertDimension(values.size(), this->n_components);

    for (unsigned int i = 0; i < this->n_components; i++)
      {
        values(i) = this->value(x_hat, y_hat, i);
      }
  }


  template <int dim, int spacedim, typename RangeNumberType>
  void
  KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType>::vector_value_list(
    const std::vector<Point<dim>> &       x_hat_list,
    const std::vector<Point<dim>> &       y_hat_list,
    std::vector<Vector<RangeNumberType>> &values) const
  {
    Assert(values.size() == x_hat_list.size(),
           ExcDimensionMismatch(values.size(), x_hat_list.size()));
    Assert(values.size() == y_hat_list.size(),
           ExcDimensionMismatch(values.size(), y_hat_list.size()));

    for (unsigned int i = 0; i < x_hat_list.size(); i++)
      {
        this->vector_value(x_hat_list[i], y_hat_list[i], values[i]);
      }
  }


  /**
   * Class for pullback the kernel function on the product of two unit cells to
   * Sauter's parametric space.
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  class KernelPulledbackToSauterSpace : public Subscriptor
  {
  public:
    const unsigned int n_components;

    KernelPulledbackToSauterSpace(
      const KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType> &kernel,
      const CellNeighboringType cell_neighboring_type);


    KernelPulledbackToSauterSpace(
      const KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType> &kernel,
      const CellNeighboringType                        cell_neighboring_type,
      const BEMValues<dim, spacedim, RangeNumberType> *bem_values);


    ~KernelPulledbackToSauterSpace();


    KernelPulledbackToSauterSpace &
    operator=(const KernelPulledbackToSauterSpace &f);


    /**
     * Evaluate the pullback of kernel function on Sauter's parametric space.
     *
     * @param p The coordinates at which the kernel function is to be evaluated.
     * It should be noted that this point has a dimension of dim*2.
     */
    RangeNumberType
    value(const Point<dim * 2> p, const unsigned int component = 0) const;


    /**
     * Evaluate the pullback of kernel function on Sauter's parametric space at
     * the quad_no'th quadrature point under the given 4D quadrature rule.
     * @param quad_no quadrature point index
     * @param component
     * @return
     */
    RangeNumberType
    value(const unsigned int quad_no, const unsigned int component = 0) const;


    void
    value_list(const std::vector<Point<dim * 2>> &points,
               std::vector<RangeNumberType> &     values,
               const unsigned int                 component = 0) const;

  private:
    const KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType>
      &                                              kernel_on_unit_cell;
    CellNeighboringType                              cell_neighboring_type;
    const BEMValues<dim, spacedim, RangeNumberType> *bem_values;
  };


  template <int dim, int spacedim, typename RangeNumberType>
  KernelPulledbackToSauterSpace<dim, spacedim, RangeNumberType>::
    KernelPulledbackToSauterSpace(
      const KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType> &kernel,
      const CellNeighboringType cell_neighboring_type)
    : n_components(kernel.n_components)
    , kernel_on_unit_cell(kernel)
    , cell_neighboring_type(cell_neighboring_type)
    , bem_values(nullptr)
  {}


  template <int dim, int spacedim, typename RangeNumberType>
  KernelPulledbackToSauterSpace<dim, spacedim, RangeNumberType>::
    KernelPulledbackToSauterSpace(
      const KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType> &kernel,
      const CellNeighboringType                        cell_neighboring_type,
      const BEMValues<dim, spacedim, RangeNumberType> *bem_values)
    : n_components(kernel.n_components)
    , kernel_on_unit_cell(kernel)
    , cell_neighboring_type(cell_neighboring_type)
    , bem_values(bem_values)
  {}


  template <int dim, int spacedim, typename RangeNumberType>
  KernelPulledbackToSauterSpace<dim, spacedim, RangeNumberType>::
    ~KernelPulledbackToSauterSpace() = default;


  template <int dim, int spacedim, typename RangeNumberType>
  KernelPulledbackToSauterSpace<dim, spacedim, RangeNumberType> &
  KernelPulledbackToSauterSpace<dim, spacedim, RangeNumberType>::
  operator=(const KernelPulledbackToSauterSpace &f)
  {
    AssertDimension(n_components, f.n_components);

    kernel_on_unit_cell = f.kernel_on_unit_cell;

    return *this;
  }


  template <int dim, int spacedim, typename RangeNumberType>
  RangeNumberType
  KernelPulledbackToSauterSpace<dim, spacedim, RangeNumberType>::value(
    const Point<dim * 2> p,
    const unsigned int   component) const
  {
    switch (cell_neighboring_type)
      {
        case SamePanel:
          {
            double jacobian_det   = p(0) * (1 - p(0)) * (1 - p(0) * p(1));
            double unit_coords[4] = {(1 - p(0)) * p(3),
                                     (1 - p(0) * p(1)) * p(2),
                                     p(0) + (1 - p(0)) * p(3),
                                     p(0) * p(1) + (1 - p(0) * p(1)) * p(2)};

            return jacobian_det * (kernel_on_unit_cell.value(
                                     Point<dim>(unit_coords[0], unit_coords[1]),
                                     Point<dim>(unit_coords[2], unit_coords[3]),
                                     component) +
                                   kernel_on_unit_cell.value(
                                     Point<dim>(unit_coords[1], unit_coords[0]),
                                     Point<dim>(unit_coords[3], unit_coords[2]),
                                     component) +
                                   kernel_on_unit_cell.value(
                                     Point<dim>(unit_coords[0], unit_coords[3]),
                                     Point<dim>(unit_coords[2], unit_coords[1]),
                                     component) +
                                   kernel_on_unit_cell.value(
                                     Point<dim>(unit_coords[1], unit_coords[2]),
                                     Point<dim>(unit_coords[3], unit_coords[0]),
                                     component) +
                                   kernel_on_unit_cell.value(
                                     Point<dim>(unit_coords[2], unit_coords[1]),
                                     Point<dim>(unit_coords[0], unit_coords[3]),
                                     component) +
                                   kernel_on_unit_cell.value(
                                     Point<dim>(unit_coords[3], unit_coords[0]),
                                     Point<dim>(unit_coords[1], unit_coords[2]),
                                     component) +
                                   kernel_on_unit_cell.value(
                                     Point<dim>(unit_coords[2], unit_coords[3]),
                                     Point<dim>(unit_coords[0], unit_coords[1]),
                                     component) +
                                   kernel_on_unit_cell.value(
                                     Point<dim>(unit_coords[3], unit_coords[2]),
                                     Point<dim>(unit_coords[1], unit_coords[0]),
                                     component));

            break;
          }
        case CommonEdge:
          {
            double jacobian_det1   = p(0) * p(0) * (1 - p(0));
            double jacobian_det2   = p(0) * p(0) * (1 - p(0) * p(1));
            double unit_coords1[4] = {(1 - p(0)) * p(3) + p(0),
                                      p(0) * p(2),
                                      (1 - p(0)) * p(3),
                                      p(0) * p(1)};
            double unit_coords2[4] = {(1 - p(0) * p(1)) * p(3) + p(0) * p(1),
                                      p(0) * p(2),
                                      (1 - p(0) * p(1)) * p(3),
                                      p(0)};

            return jacobian_det1 *
                     (kernel_on_unit_cell.value(
                        Point<dim>(unit_coords1[0], unit_coords1[1]),
                        Point<dim>(unit_coords1[2], unit_coords1[3]),
                        component) +
                      kernel_on_unit_cell.value(
                        Point<dim>(unit_coords1[2], unit_coords1[1]),
                        Point<dim>(unit_coords1[0], unit_coords1[3]),
                        component)) +
                   jacobian_det2 *
                     (kernel_on_unit_cell.value(
                        Point<dim>(unit_coords2[0], unit_coords2[1]),
                        Point<dim>(unit_coords2[2], unit_coords2[3]),
                        component) +
                      kernel_on_unit_cell.value(
                        Point<dim>(unit_coords2[0], unit_coords2[3]),
                        Point<dim>(unit_coords2[2], unit_coords2[1]),
                        component) +
                      kernel_on_unit_cell.value(
                        Point<dim>(unit_coords2[2], unit_coords2[1]),
                        Point<dim>(unit_coords2[0], unit_coords2[3]),
                        component) +
                      kernel_on_unit_cell.value(
                        Point<dim>(unit_coords2[2], unit_coords2[3]),
                        Point<dim>(unit_coords2[0], unit_coords2[1]),
                        component));

            break;
          }
        case CommonVertex:
          {
            double jacobian_det   = std::pow(p(0), 3);
            double unit_coords[4] = {p(0),
                                     p(0) * p(1),
                                     p(0) * p(2),
                                     p(0) * p(3)};

            return jacobian_det * (kernel_on_unit_cell.value(
                                     Point<dim>(unit_coords[0], unit_coords[1]),
                                     Point<dim>(unit_coords[2], unit_coords[3]),
                                     component) +
                                   kernel_on_unit_cell.value(
                                     Point<dim>(unit_coords[1], unit_coords[0]),
                                     Point<dim>(unit_coords[2], unit_coords[3]),
                                     component) +
                                   kernel_on_unit_cell.value(
                                     Point<dim>(unit_coords[1], unit_coords[2]),
                                     Point<dim>(unit_coords[0], unit_coords[3]),
                                     component) +
                                   kernel_on_unit_cell.value(
                                     Point<dim>(unit_coords[1], unit_coords[2]),
                                     Point<dim>(unit_coords[3], unit_coords[0]),
                                     component));

            break;
          }
        case Regular:
          {
            // There is no coordinate transformation for the regular case, so
            // directly evaluate the kernel function on the product of unit
            // cells.

            return kernel_on_unit_cell.value(Point<dim>(p(0), p(1)),
                                             Point<dim>(p(2), p(3)),
                                             component);

            break;
          }
        default:
          Assert(false, ExcInternalError());
          return 0;
      }
  }


  template <int dim, int spacedim, typename RangeNumberType>
  RangeNumberType
  KernelPulledbackToSauterSpace<dim, spacedim, RangeNumberType>::value(
    const unsigned int quad_no,
    const unsigned int component) const
  {
    RangeNumberType kernel_value = 0.;

    switch (cell_neighboring_type)
      {
        case SamePanel:
          {
            // Current point in the Sauter's parametric space, at which the
            // pulled back kernel function is to be evaluated.
            const Point<dim * 2> &p =
              bem_values->quad_rule_for_same_panel.point(quad_no);
            double jacobian_det = p(0) * (1 - p(0)) * (1 - p(0) * p(1));

            for (unsigned int k3_index = 0; k3_index < 8; k3_index++)
              {
                kernel_value +=
                  kernel_on_unit_cell.value(k3_index, quad_no, component);
              }

            kernel_value *= jacobian_det;

            break;
          }
        case CommonEdge:
          {
            // Current point in the Sauter's parametric space, at which the
            // pulled back kernel function is to be evaluated.
            const Point<dim * 2> &p =
              bem_values->quad_rule_for_common_edge.point(quad_no);
            double jacobian_det1 = p(0) * p(0) * (1 - p(0));
            double jacobian_det2 = p(0) * p(0) * (1 - p(0) * p(1));

            kernel_value =
              jacobian_det1 *
                (kernel_on_unit_cell.value(0, quad_no, component) +
                 kernel_on_unit_cell.value(1, quad_no, component)) +
              jacobian_det2 *
                (kernel_on_unit_cell.value(2, quad_no, component) +
                 kernel_on_unit_cell.value(3, quad_no, component) +
                 kernel_on_unit_cell.value(4, quad_no, component) +
                 kernel_on_unit_cell.value(5, quad_no, component));

            break;
          }
        case CommonVertex:
          {
            // Current point in the Sauter's parametric space, at which the
            // pulled back kernel function is to be evaluated.
            const Point<dim * 2> &p =
              bem_values->quad_rule_for_common_vertex.point(quad_no);
            double jacobian_det = std::pow(p(0), 3);

            for (unsigned int k3_index = 0; k3_index < 4; k3_index++)
              {
                kernel_value +=
                  kernel_on_unit_cell.value(k3_index, quad_no, component);
              }

            kernel_value *= jacobian_det;

            break;
          }
        case Regular:
          {
            // There is no coordinate transformation for the regular case, so
            // directly evaluate the kernel function on the product of unit
            // cells.
            kernel_value = kernel_on_unit_cell.value(0, quad_no, component);

            break;
          }
        default:
          Assert(false, ExcInternalError());
      }

    return kernel_value;
  }


  template <int dim, int spacedim, typename RangeNumberType>
  void
  KernelPulledbackToSauterSpace<dim, spacedim, RangeNumberType>::value_list(
    const std::vector<Point<dim * 2>> &points,
    std::vector<RangeNumberType> &     values,
    const unsigned int                 component) const
  {
    Assert(values.size() == points.size(),
           ExcDimensionMismatch(values.size(), points.size()));

    for (unsigned int i = 0; i < points.size(); i++)
      {
        values[i] = this->value(points[i], component);
      }
  }


  template <int dim, int spacedim, typename RangeNumberType = double>
  RangeNumberType
  ApplyQuadrature(
    const Quadrature<dim * 2> &quad_rule,
    const KernelPulledbackToSauterSpace<dim, spacedim, RangeNumberType> &f,
    unsigned int component = 0)
  {
    RangeNumberType result = 0.;

    const std::vector<Point<dim * 2>> &quad_points  = quad_rule.get_points();
    const std::vector<double> &        quad_weights = quad_rule.get_weights();

    for (unsigned int q = 0; q < quad_rule.size(); q++)
      {
        result += f.value(quad_points[q], component) * quad_weights[q];
      }

    return result;
  }


  template <int dim, int spacedim, typename RangeNumberType = double>
  RangeNumberType
  ApplyQuadratureUsingBEMValues(
    const Quadrature<dim * 2> &quad_rule,
    const KernelPulledbackToSauterSpace<dim, spacedim, RangeNumberType> &f,
    unsigned int component = 0)
  {
    RangeNumberType result = 0.;

    const std::vector<double> &quad_weights = quad_rule.get_weights();

    for (unsigned int q = 0; q < quad_rule.size(); q++)
      {
        // Evaluate the integrand with precalculated shape values and shape
        // gradient matrices.
        result += f.value(q, component) * quad_weights[q];
      }

    return result;
  }


  /**
   * Get the DoF vertex indices from a list of DoF indices which have been
   * arranged in forward or backward tensor product ordering. N.B. There are
   * <code>GeometryInfo<dim>::vertices_per_cell</code> vertices in the returned
   * array, among which the last two DoF vertex indices have been swapped so
   * that the vertex DoFs are arranged in clockwise or counter clockwise order
   * instead of the zigzag order.
   */
  template <int dim, int spacedim>
  std::array<types::global_dof_index, GeometryInfo<dim>::vertices_per_cell>
  get_vertex_dof_indices_swapped(
    const FiniteElement<dim, spacedim> &        fe,
    const std::vector<types::global_dof_index> &dof_indices)
  {
    Assert(dim == 2, ExcNotImplemented());

    std::array<types::global_dof_index, GeometryInfo<dim>::vertices_per_cell>
      vertex_dof_indices;

    vertex_dof_indices[0] = dof_indices[0];
    vertex_dof_indices[1] = dof_indices[fe.dofs_per_face - 1];
    vertex_dof_indices[2] = dof_indices[dof_indices.size() - 1];
    vertex_dof_indices[3] =
      dof_indices[dof_indices.size() - 1 - (fe.dofs_per_face - 1)];

    return vertex_dof_indices;
  }


  /**
   * Get the DoF vertex indices from a list of DoF indices which have been
   * arranged in forward or backward tensor product ordering. N.B. There are
   * <code>GeometryInfo<dim>::vertices_per_cell</code> vertices in the returned
   * array, among which the last two DoF vertex indices have been swapped so
   * that the vertex DoFs are arranged in clockwise or counter clockwise order
   * instead of the zigzag order.
   */
  template <int dim, int spacedim>
  void
  get_vertex_dof_indices_swapped(
    const FiniteElement<dim, spacedim> &        fe,
    const std::vector<types::global_dof_index> &dof_indices,
    std::array<types::global_dof_index, GeometryInfo<dim>::vertices_per_cell>
      &vertex_dof_indices)
  {
    Assert(dim == 2, ExcNotImplemented());

    vertex_dof_indices[0] = dof_indices[0];
    vertex_dof_indices[1] = dof_indices[fe.dofs_per_face - 1];
    vertex_dof_indices[2] = dof_indices[dof_indices.size() - 1];
    vertex_dof_indices[3] =
      dof_indices[dof_indices.size() - 1 - (fe.dofs_per_face - 1)];
  }


  /**
   * Get the starting vertex DoF index.
   */
  template <int vertices_per_cell>
  unsigned int
  get_start_vertex_dof_index(
    const std::vector<types::global_dof_index> &vertex_dof_index_intersection,
    const std::array<types::global_dof_index, vertices_per_cell>
      &local_vertex_dof_indices_swapped)
  {
    unsigned int starting_vertex_index = 9999;

    // There are two cases to be processed here, common edge and common vertex.
    // In the common edge case, there are two DoF indices in
    // <code>vertex_dof_index_intersection</code>, their array indices wrt. the
    // vector <code>local_vertex_dof_indices_swapped</code> will be searched. By
    // considering this vector as a closed loop list, the two DoF indices in
    // this vector are successively located, the first one of which is the
    // vertex to start DoF traversing.
    switch (vertex_dof_index_intersection.size())
      {
        case 2: // Common edge case
          {
            typename std::array<types::global_dof_index,
                                vertices_per_cell>::const_iterator
              first_common_vertex_iterator =
                std::find(local_vertex_dof_indices_swapped.cbegin(),
                          local_vertex_dof_indices_swapped.cend(),
                          vertex_dof_index_intersection[0]);
            typename std::array<types::global_dof_index,
                                vertices_per_cell>::const_iterator
              second_common_vertex_iterator =
                std::find(local_vertex_dof_indices_swapped.cbegin(),
                          local_vertex_dof_indices_swapped.cend(),
                          vertex_dof_index_intersection[1]);

            if ((first_common_vertex_iterator + 1) !=
                local_vertex_dof_indices_swapped.cend())
              {
                if (*(first_common_vertex_iterator + 1) ==
                    vertex_dof_index_intersection[1])
                  {
                    starting_vertex_index =
                      first_common_vertex_iterator -
                      local_vertex_dof_indices_swapped.cbegin();
                  }
                else
                  {
                    starting_vertex_index =
                      second_common_vertex_iterator -
                      local_vertex_dof_indices_swapped.cbegin();
                  }
              }
            else
              {
                if ((*local_vertex_dof_indices_swapped.cbegin()) ==
                    vertex_dof_index_intersection[1])
                  {
                    starting_vertex_index =
                      first_common_vertex_iterator -
                      local_vertex_dof_indices_swapped.cbegin();
                  }
                else
                  {
                    starting_vertex_index =
                      second_common_vertex_iterator -
                      local_vertex_dof_indices_swapped.cbegin();
                  }
              }

            break;
          }
        case 1: // Common vertex case
          {
            typename std::array<types::global_dof_index,
                                vertices_per_cell>::const_iterator
              first_common_vertex_iterator =
                std::find(local_vertex_dof_indices_swapped.cbegin(),
                          local_vertex_dof_indices_swapped.cend(),
                          vertex_dof_index_intersection[0]);
            Assert(first_common_vertex_iterator !=
                     local_vertex_dof_indices_swapped.cend(),
                   ExcInternalError());

            starting_vertex_index = first_common_vertex_iterator -
                                    local_vertex_dof_indices_swapped.cbegin();

            break;
          }
        default:
          Assert(false, ExcInternalError());
          break;
      }

    // Because the last two elements in the original list of vertex DoF indices
    // have been swapped, we need to correct the starting vertex index here.
    if (starting_vertex_index == 2)
      {
        starting_vertex_index = 3;
      }
    else if (starting_vertex_index == 3)
      {
        starting_vertex_index = 2;
      }

    return starting_vertex_index;
  }


  /**
   * This function implements Sauter's quadrature rule on quadrangular mesh.
   * It handles various cases including same panel, common edge, common vertex
   * and regular cell neighboring types.
   *
   * @param kernel_function Laplace kernel function.
   * @param kx_cell_iter Iterator pointing to $K_x$.
   * @param kx_cell_iter Iterator pointing to $K_y$.
   * @param kx_mapping Mapping used for $K_x$. Because a mesher usually generates
   * 1st order grid, if there is no additional manifold specification, the
   * mapping should be 1st order.
   * @param kx_mapping Mapping used for $K_y$. Because a mesher usually generates
   * 1st order grid, if there is no additional manifold specification, the
   * mapping should be 1st order.
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  FullMatrix<RangeNumberType>
  SauterQuadRule(
    const LaplaceKernel::KernelFunction<spacedim, RangeNumberType>
      &                                                      kernel_function,
    const typename DoFHandler<dim, spacedim>::cell_iterator &kx_cell_iter,
    const typename DoFHandler<dim, spacedim>::cell_iterator &ky_cell_iter,
    const MappingQGeneric<dim, spacedim> &                   kx_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const MappingQGeneric<dim, spacedim> &ky_mapping =
      MappingQGeneric<dim, spacedim>(1))
  {
    // Geometry information.
    const unsigned int vertices_per_cell = GeometryInfo<dim>::vertices_per_cell;

    // Determine the cell neighboring type based on the vertex dof indices. The
    // common dof indices will be stored into the vector
    // <code>vertex_dof_index_intersection</code> if there is any.
    std::array<types::global_dof_index, vertices_per_cell>
      kx_vertex_dof_indices(
        get_vertex_dof_indices<dim, spacedim>(kx_cell_iter));
    std::array<types::global_dof_index, vertices_per_cell>
      ky_vertex_dof_indices(
        get_vertex_dof_indices<dim, spacedim>(ky_cell_iter));

    std::vector<types::global_dof_index> vertex_dof_index_intersection;
    vertex_dof_index_intersection.reserve(vertices_per_cell);
    CellNeighboringType cell_neighboring_type =
      detect_cell_neighboring_type<dim>(kx_vertex_dof_indices,
                                        ky_vertex_dof_indices,
                                        vertex_dof_index_intersection);

    const FiniteElement<dim, spacedim> &kx_fe = kx_cell_iter->get_fe();
    const FiniteElement<dim, spacedim> &ky_fe = ky_cell_iter->get_fe();

    const unsigned int kx_n_dofs = kx_fe.dofs_per_cell;
    const unsigned int ky_n_dofs = ky_fe.dofs_per_cell;

    // Support points of $K_x$ and $K_y$ in the default
    // hierarchical order.
    std::vector<Point<spacedim>> kx_support_points_hierarchical =
      hierarchical_support_points_in_real_cell(kx_cell_iter, kx_fe, kx_mapping);
    std::vector<Point<spacedim>> ky_support_points_hierarchical =
      hierarchical_support_points_in_real_cell(ky_cell_iter, ky_fe, ky_mapping);

    // Permuted support points to be used in the common edge and common vertex
    // cases instead of the original support points in hierarchical order.
    std::vector<Point<spacedim>> kx_support_points_permuted;
    std::vector<Point<spacedim>> ky_support_points_permuted;
    kx_support_points_permuted.reserve(kx_n_dofs);
    ky_support_points_permuted.reserve(ky_n_dofs);

    // Global indices for the local DoFs in the default hierarchical order.
    std::vector<types::global_dof_index> kx_local_dof_indices_hierarchical(
      kx_n_dofs);
    std::vector<types::global_dof_index> ky_local_dof_indices_hierarchical(
      ky_n_dofs);
    // N.B. The vector holding local DoF indices has to have the right size
    // before being passed to the function <code>get_dof_indices</code>.
    kx_cell_iter->get_dof_indices(kx_local_dof_indices_hierarchical);
    ky_cell_iter->get_dof_indices(ky_local_dof_indices_hierarchical);

    // Permuted local DoF indices, which has the same permutation as that
    // applied to support points.
    std::vector<types::global_dof_index> kx_local_dof_indices_permuted;
    std::vector<types::global_dof_index> ky_local_dof_indices_permuted;
    kx_local_dof_indices_permuted.reserve(kx_n_dofs);
    ky_local_dof_indices_permuted.reserve(ky_n_dofs);

    // Generate 4D Gauss-Legendre quadrature rules for various cell
    // neighboring types.
    const unsigned int quad_order_for_same_panel    = 5;
    const unsigned int quad_order_for_common_edge   = 4;
    const unsigned int quad_order_for_common_vertex = 4;
    const unsigned int quad_order_for_regular       = 3;

    QGauss<4> quad_rule_for_same_panel(quad_order_for_same_panel);
    QGauss<4> quad_rule_for_common_edge(quad_order_for_common_edge);
    QGauss<4> quad_rule_for_common_vertex(quad_order_for_common_vertex);
    QGauss<4> quad_rule_for_regular(quad_order_for_regular);

    FullMatrix<RangeNumberType> cell_matrix(kx_n_dofs, ky_n_dofs);

    try
      {
        // Downcast references of FiniteElement objects to FE_Poly references
        // for obtaining the inverse polynomial space numbering.
        // TODO: Inverse polynomial space numbering may be obtained by calling
        // <code>FETools::hierarchic_to_lexicographic_numbering</code> or
        // <code>FETools::lexicographic_to_hierarchic_numbering</code>.
        using FE_Poly = FE_Poly<TensorProductPolynomials<dim>, dim, spacedim>;
        const FE_Poly &kx_fe_poly = dynamic_cast<const FE_Poly &>(kx_fe);
        const FE_Poly &ky_fe_poly = dynamic_cast<const FE_Poly &>(ky_fe);

        // Polynomial space inverse numbering for recovering the tensor product
        // ordering.
        std::vector<unsigned int> kx_fe_poly_space_numbering_inverse =
          kx_fe_poly.get_poly_space_numbering_inverse();
        std::vector<unsigned int> ky_fe_poly_space_numbering_inverse =
          ky_fe_poly.get_poly_space_numbering_inverse();

        switch (cell_neighboring_type)
          {
            case SamePanel:
              {
                Assert(vertex_dof_index_intersection.size() ==
                         vertices_per_cell,
                       ExcInternalError());

                // Get support points in tensor product oder.
                kx_support_points_permuted =
                  permute_vector(kx_support_points_hierarchical,
                                 kx_fe_poly_space_numbering_inverse);
                ky_support_points_permuted =
                  permute_vector(ky_support_points_hierarchical,
                                 ky_fe_poly_space_numbering_inverse);

                // Get permuted local DoF indices.
                kx_local_dof_indices_permuted =
                  permute_vector(kx_local_dof_indices_hierarchical,
                                 kx_fe_poly_space_numbering_inverse);
                ky_local_dof_indices_permuted =
                  permute_vector(ky_local_dof_indices_hierarchical,
                                 ky_fe_poly_space_numbering_inverse);


                //                // DEBUG: Print out permuted support points
                //                and DoF indices for
                //                // debugging.
                //                deallog << "Support points and DoF indices in
                //                Kx:\n"; deallog << "DoF_index X Y Z\n"; for
                //                (unsigned int i = 0; i < kx_n_dofs; i++)
                //                  {
                //                    deallog <<
                //                    kx_local_dof_indices_permuted[i] << " "
                //                            << kx_support_points_permuted[i]
                //                            << std::endl;
                //                  }
                //
                //                deallog << "Support points and DoF indices in
                //                Ky:\n"; deallog << "DoF_index X Y Z\n"; for
                //                (unsigned int i = 0; i < ky_n_dofs; i++)
                //                  {
                //                    deallog <<
                //                    ky_local_dof_indices_permuted[i] << " "
                //                            << ky_support_points_permuted[i]
                //                            << std::endl;
                //                  }


                // Iterate over DoFs for test function space in tensor product
                // order in $K_x$.
                for (unsigned int i = 0; i < kx_n_dofs; i++)
                  {
                    // Iterate over DoFs for ansatz function space in tensor
                    // product order in $K_y$.
                    for (unsigned int j = 0; j < ky_n_dofs; j++)
                      {
                        // Pullback the kernel function to unit cell.
                        KernelPulledbackToUnitCell<dim,
                                                   spacedim,
                                                   RangeNumberType>
                          kernel_pullback_on_unit(kernel_function,
                                                  cell_neighboring_type,
                                                  kx_support_points_permuted,
                                                  ky_support_points_permuted,
                                                  kx_fe,
                                                  ky_fe,
                                                  i,
                                                  j);

                        // Pullback the kernel function to Sauter parameter
                        // space.
                        KernelPulledbackToSauterSpace<dim,
                                                      spacedim,
                                                      RangeNumberType>
                          kernel_pullback_on_sauter(kernel_pullback_on_unit,
                                                    cell_neighboring_type);

                        // Apply 4d Sauter numerical quadrature.
                        cell_matrix(i, j) =
                          ApplyQuadrature(quad_rule_for_same_panel,
                                          kernel_pullback_on_sauter);
                      }
                  }

                break;
              }
            case CommonEdge:
              {
                // This part handles the common edge case of Sauter's
                // quadrature rule.
                // 1. Get the DoF indices in tensor product order for $K_x$.
                // 2. Get the DoF indices in reversed tensor product order for
                // $K_x$.
                // 3. Extract DoF indices only for cell vertices in $K_x$ and
                // $K_y$. N.B. The DoF indices for the last two vertices are
                // swapped, such that the four vertices are in clockwise or
                // counter clockwise order.
                // 4. Determine the starting vertex.

                Assert(vertex_dof_index_intersection.size() ==
                         GeometryInfo<dim>::vertices_per_face,
                       ExcInternalError());

                kx_local_dof_indices_permuted =
                  permute_vector(kx_local_dof_indices_hierarchical,
                                 kx_fe_poly_space_numbering_inverse);

                std::vector<unsigned int>
                  ky_fe_reversed_poly_space_numbering_inverse =
                    generate_backward_dof_permutation(ky_fe, 0);
                ky_local_dof_indices_permuted =
                  permute_vector(ky_local_dof_indices_hierarchical,
                                 ky_fe_reversed_poly_space_numbering_inverse);

                std::array<types::global_dof_index, vertices_per_cell>
                  kx_local_vertex_dof_indices_swapped =
                    get_vertex_dof_indices_swapped(
                      kx_fe, kx_local_dof_indices_permuted);
                std::array<types::global_dof_index, vertices_per_cell>
                  ky_local_vertex_dof_indices_swapped =
                    get_vertex_dof_indices_swapped(
                      ky_fe, ky_local_dof_indices_permuted);

                // Determine the starting vertex index in $K_x$ and $K_y$.
                unsigned int kx_starting_vertex_index =
                  get_start_vertex_dof_index<vertices_per_cell>(
                    vertex_dof_index_intersection,
                    kx_local_vertex_dof_indices_swapped);
                Assert(kx_starting_vertex_index < vertices_per_cell,
                       ExcInternalError());
                unsigned int ky_starting_vertex_index =
                  get_start_vertex_dof_index<vertices_per_cell>(
                    vertex_dof_index_intersection,
                    ky_local_vertex_dof_indices_swapped);
                Assert(ky_starting_vertex_index < vertices_per_cell,
                       ExcInternalError());

                // Generate the permutation of DoFs in $K_x$ and $K_y$ by
                // starting from <code>kx_starting_vertex_index</code> or
                // <code>ky_starting_vertex_index</code>.
                std::vector<unsigned int> kx_local_dof_permutation =
                  generate_forward_dof_permutation(kx_fe,
                                                   kx_starting_vertex_index);
                std::vector<unsigned int> ky_local_dof_permutation =
                  generate_backward_dof_permutation(ky_fe,
                                                    ky_starting_vertex_index);

                kx_support_points_permuted =
                  permute_vector(kx_support_points_hierarchical,
                                 kx_local_dof_permutation);
                ky_support_points_permuted =
                  permute_vector(ky_support_points_hierarchical,
                                 ky_local_dof_permutation);

                kx_local_dof_indices_permuted =
                  permute_vector(kx_local_dof_indices_hierarchical,
                                 kx_local_dof_permutation);
                ky_local_dof_indices_permuted =
                  permute_vector(ky_local_dof_indices_hierarchical,
                                 ky_local_dof_permutation);


                //                // DEBUG: Print out permuted support points
                //                and DoF indices for
                //                // debugging.
                //                deallog << "Support points and DoF indices in
                //                Kx:\n"; deallog << "DoF_index X Y Z\n"; for
                //                (unsigned int i = 0; i < kx_n_dofs; i++)
                //                  {
                //                    deallog <<
                //                    kx_local_dof_indices_permuted[i] << " "
                //                            << kx_support_points_permuted[i]
                //                            << std::endl;
                //                  }
                //
                //                deallog << "Support points and DoF indices in
                //                Ky:\n"; deallog << "DoF_index X Y Z\n"; for
                //                (unsigned int i = 0; i < ky_n_dofs; i++)
                //                  {
                //                    deallog <<
                //                    ky_local_dof_indices_permuted[i] << " "
                //                            << ky_support_points_permuted[i]
                //                            << std::endl;
                //                  }


                // Iterate over DoFs for test function space in tensor product
                // order in $K_x$.
                for (unsigned int i = 0; i < kx_n_dofs; i++)
                  {
                    // Iterate over DoFs for ansatz function space in tensor
                    // product order in $K_y$.
                    for (unsigned int j = 0; j < ky_n_dofs; j++)
                      {
                        // Pullback the kernel function to unit cell.
                        KernelPulledbackToUnitCell<dim,
                                                   spacedim,
                                                   RangeNumberType>
                          kernel_pullback_on_unit(kernel_function,
                                                  cell_neighboring_type,
                                                  kx_support_points_permuted,
                                                  ky_support_points_permuted,
                                                  kx_fe,
                                                  ky_fe,
                                                  i,
                                                  j);

                        // Pullback the kernel function to Sauter parameter
                        // space.
                        KernelPulledbackToSauterSpace<dim,
                                                      spacedim,
                                                      RangeNumberType>
                          kernel_pullback_on_sauter(kernel_pullback_on_unit,
                                                    cell_neighboring_type);

                        // Apply 4d Sauter numerical quadrature.
                        cell_matrix(i, j) =
                          ApplyQuadrature(quad_rule_for_common_edge,
                                          kernel_pullback_on_sauter);
                      }
                  }

                break;
              }
            case CommonVertex:
              {
                Assert(vertex_dof_index_intersection.size() == 1,
                       ExcInternalError());

                kx_local_dof_indices_permuted =
                  permute_vector(kx_local_dof_indices_hierarchical,
                                 kx_fe_poly_space_numbering_inverse);

                ky_local_dof_indices_permuted =
                  permute_vector(ky_local_dof_indices_hierarchical,
                                 ky_fe_poly_space_numbering_inverse);

                std::array<types::global_dof_index, vertices_per_cell>
                  kx_local_vertex_dof_indices_swapped =
                    get_vertex_dof_indices_swapped(
                      kx_fe, kx_local_dof_indices_permuted);
                std::array<types::global_dof_index, vertices_per_cell>
                  ky_local_vertex_dof_indices_swapped =
                    get_vertex_dof_indices_swapped(
                      ky_fe, ky_local_dof_indices_permuted);

                // Determine the starting vertex index in $K_x$ and $K_y$.
                unsigned int kx_starting_vertex_index =
                  get_start_vertex_dof_index<vertices_per_cell>(
                    vertex_dof_index_intersection,
                    kx_local_vertex_dof_indices_swapped);
                Assert(kx_starting_vertex_index < vertices_per_cell,
                       ExcInternalError());
                unsigned int ky_starting_vertex_index =
                  get_start_vertex_dof_index<vertices_per_cell>(
                    vertex_dof_index_intersection,
                    ky_local_vertex_dof_indices_swapped);
                Assert(ky_starting_vertex_index < vertices_per_cell,
                       ExcInternalError());

                // Generate the permutation of DoFs in $K_x$ and $K_y$ by
                // starting from <code>kx_starting_vertex_index</code> or
                // <code>ky_starting_vertex_index</code>.
                std::vector<unsigned int> kx_local_dof_permutation =
                  generate_forward_dof_permutation(kx_fe,
                                                   kx_starting_vertex_index);
                std::vector<unsigned int> ky_local_dof_permutation =
                  generate_forward_dof_permutation(ky_fe,
                                                   ky_starting_vertex_index);

                kx_support_points_permuted =
                  permute_vector(kx_support_points_hierarchical,
                                 kx_local_dof_permutation);
                ky_support_points_permuted =
                  permute_vector(ky_support_points_hierarchical,
                                 ky_local_dof_permutation);

                kx_local_dof_indices_permuted =
                  permute_vector(kx_local_dof_indices_hierarchical,
                                 kx_local_dof_permutation);
                ky_local_dof_indices_permuted =
                  permute_vector(ky_local_dof_indices_hierarchical,
                                 ky_local_dof_permutation);


                //                // DEBUG: Print out permuted support points
                //                and DoF indices for
                //                // debugging.
                //                deallog << "Support points and DoF indices in
                //                Kx:\n"; deallog << "DoF_index X Y Z\n"; for
                //                (unsigned int i = 0; i < kx_n_dofs; i++)
                //                  {
                //                    deallog <<
                //                    kx_local_dof_indices_permuted[i] << " "
                //                            << kx_support_points_permuted[i]
                //                            << std::endl;
                //                  }
                //
                //                deallog << "Support points and DoF indices in
                //                Ky:\n"; deallog << "DoF_index X Y Z\n"; for
                //                (unsigned int i = 0; i < ky_n_dofs; i++)
                //                  {
                //                    deallog <<
                //                    ky_local_dof_indices_permuted[i] << " "
                //                            << ky_support_points_permuted[i]
                //                            << std::endl;
                //                  }


                // Iterate over DoFs for test function space in tensor product
                // order in $K_x$.
                for (unsigned int i = 0; i < kx_n_dofs; i++)
                  {
                    // Iterate over DoFs for ansatz function space in tensor
                    // product order in $K_y$.
                    for (unsigned int j = 0; j < ky_n_dofs; j++)
                      {
                        // Pullback the kernel function to unit cell.
                        KernelPulledbackToUnitCell<dim,
                                                   spacedim,
                                                   RangeNumberType>
                          kernel_pullback_on_unit(kernel_function,
                                                  cell_neighboring_type,
                                                  kx_support_points_permuted,
                                                  ky_support_points_permuted,
                                                  kx_fe,
                                                  ky_fe,
                                                  i,
                                                  j);

                        // Pullback the kernel function to Sauter parameter
                        // space.
                        KernelPulledbackToSauterSpace<dim,
                                                      spacedim,
                                                      RangeNumberType>
                          kernel_pullback_on_sauter(kernel_pullback_on_unit,
                                                    cell_neighboring_type);

                        // Apply 4d Sauter numerical quadrature.
                        cell_matrix(i, j) =
                          ApplyQuadrature(quad_rule_for_common_vertex,
                                          kernel_pullback_on_sauter);
                      }
                  }

                break;
              }
            case Regular:
              {
                Assert(vertex_dof_index_intersection.size() == 0,
                       ExcInternalError());

                kx_local_dof_indices_permuted =
                  permute_vector(kx_local_dof_indices_hierarchical,
                                 kx_fe_poly_space_numbering_inverse);

                ky_local_dof_indices_permuted =
                  permute_vector(ky_local_dof_indices_hierarchical,
                                 ky_fe_poly_space_numbering_inverse);


                kx_support_points_permuted =
                  permute_vector(kx_support_points_hierarchical,
                                 kx_fe_poly_space_numbering_inverse);
                ky_support_points_permuted =
                  permute_vector(ky_support_points_hierarchical,
                                 ky_fe_poly_space_numbering_inverse);


                //                // DEBUG: Print out permuted support points
                //                and DoF indices for
                //                // debugging.
                //                deallog << "Support points and DoF indices in
                //                Kx:\n"; deallog << "DoF_index X Y Z\n"; for
                //                (unsigned int i = 0; i < kx_n_dofs; i++)
                //                  {
                //                    deallog <<
                //                    kx_local_dof_indices_permuted[i] << " "
                //                            << kx_support_points_permuted[i]
                //                            << std::endl;
                //                  }
                //
                //                deallog << "Support points and DoF indices in
                //                Ky:\n"; deallog << "DoF_index X Y Z\n"; for
                //                (unsigned int i = 0; i < ky_n_dofs; i++)
                //                  {
                //                    deallog <<
                //                    ky_local_dof_indices_permuted[i] << " "
                //                            << ky_support_points_permuted[i]
                //                            << std::endl;
                //                  }


                // Iterate over DoFs for test function space in tensor product
                // order in $K_x$.
                for (unsigned int i = 0; i < kx_n_dofs; i++)
                  {
                    // Iterate over DoFs for ansatz function space in tensor
                    // product order in $K_y$.
                    for (unsigned int j = 0; j < ky_n_dofs; j++)
                      {
                        // Pullback the kernel function to unit cell.
                        KernelPulledbackToUnitCell<dim,
                                                   spacedim,
                                                   RangeNumberType>
                          kernel_pullback_on_unit(kernel_function,
                                                  cell_neighboring_type,
                                                  kx_support_points_permuted,
                                                  ky_support_points_permuted,
                                                  kx_fe,
                                                  ky_fe,
                                                  i,
                                                  j);

                        // Pullback the kernel function to Sauter parameter
                        // space.
                        KernelPulledbackToSauterSpace<dim,
                                                      spacedim,
                                                      RangeNumberType>
                          kernel_pullback_on_sauter(kernel_pullback_on_unit,
                                                    cell_neighboring_type);

                        // Apply 4d Sauter numerical quadrature.
                        cell_matrix(i, j) =
                          ApplyQuadrature(quad_rule_for_regular,
                                          kernel_pullback_on_sauter);
                      }
                  }

                break;
              }
            default:
              {
                Assert(false, ExcNotImplemented());
              }
          }
      }
    catch (const std::bad_cast &e)
      {
        Assert(false, ExcInternalError());
      }

    return cell_matrix;
  }


  template <int dim, int spacedim, typename RangeNumberType = double>
  FullMatrix<RangeNumberType>
  SauterQuadRule(
    const LaplaceKernel::KernelFunction<spacedim, RangeNumberType>
      &                                                      kernel_function,
    const BEMValues<dim, spacedim> &                         bem_values,
    const typename DoFHandler<dim, spacedim>::cell_iterator &kx_cell_iter,
    const typename DoFHandler<dim, spacedim>::cell_iterator &ky_cell_iter,
    const MappingQGeneric<dim, spacedim> &                   kx_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const MappingQGeneric<dim, spacedim> &ky_mapping =
      MappingQGeneric<dim, spacedim>(1))
  {
    // Geometry information.
    const unsigned int vertices_per_cell = GeometryInfo<dim>::vertices_per_cell;

    // Determine the cell neighboring type based on the vertex dof indices. The
    // common dof indices will be stored into the vector
    // <code>vertex_dof_index_intersection</code> if there is any.
    std::array<types::global_dof_index, vertices_per_cell>
      kx_vertex_dof_indices(
        get_vertex_dof_indices<dim, spacedim>(kx_cell_iter));
    std::array<types::global_dof_index, vertices_per_cell>
      ky_vertex_dof_indices(
        get_vertex_dof_indices<dim, spacedim>(ky_cell_iter));

    std::vector<types::global_dof_index> vertex_dof_index_intersection;
    vertex_dof_index_intersection.reserve(vertices_per_cell);
    CellNeighboringType cell_neighboring_type =
      detect_cell_neighboring_type<dim>(kx_vertex_dof_indices,
                                        ky_vertex_dof_indices,
                                        vertex_dof_index_intersection);

    const FiniteElement<dim, spacedim> &kx_fe = kx_cell_iter->get_fe();
    const FiniteElement<dim, spacedim> &ky_fe = ky_cell_iter->get_fe();

    const unsigned int kx_n_dofs = kx_fe.dofs_per_cell;
    const unsigned int ky_n_dofs = ky_fe.dofs_per_cell;

    // Support points of $K_x$ and $K_y$ in the default
    // hierarchical order.
    std::vector<Point<spacedim>> kx_support_points_hierarchical =
      hierarchical_support_points_in_real_cell(kx_cell_iter, kx_fe, kx_mapping);
    std::vector<Point<spacedim>> ky_support_points_hierarchical =
      hierarchical_support_points_in_real_cell(ky_cell_iter, ky_fe, ky_mapping);

    // Permuted support points to be used in the common edge and common vertex
    // cases instead of the original support points in hierarchical order.
    std::vector<Point<spacedim>> kx_support_points_permuted;
    std::vector<Point<spacedim>> ky_support_points_permuted;
    kx_support_points_permuted.reserve(kx_n_dofs);
    ky_support_points_permuted.reserve(ky_n_dofs);

    // Global indices for the local DoFs in the default hierarchical order.
    std::vector<types::global_dof_index> kx_local_dof_indices_hierarchical(
      kx_n_dofs);
    std::vector<types::global_dof_index> ky_local_dof_indices_hierarchical(
      ky_n_dofs);
    // N.B. The vector holding local DoF indices has to have the right size
    // before being passed to the function <code>get_dof_indices</code>.
    kx_cell_iter->get_dof_indices(kx_local_dof_indices_hierarchical);
    ky_cell_iter->get_dof_indices(ky_local_dof_indices_hierarchical);

    // Permuted local DoF indices, which has the same permutation as that
    // applied to support points.
    std::vector<types::global_dof_index> kx_local_dof_indices_permuted;
    std::vector<types::global_dof_index> ky_local_dof_indices_permuted;
    kx_local_dof_indices_permuted.reserve(kx_n_dofs);
    ky_local_dof_indices_permuted.reserve(ky_n_dofs);

    FullMatrix<RangeNumberType> cell_matrix(kx_n_dofs, ky_n_dofs);

    try
      {
        // Downcast references of FiniteElement objects to FE_Poly references
        // for obtaining the inverse polynomial space numbering.
        // TODO: Inverse polynomial space numbering may be obtained by calling
        // <code>FETools::hierarchic_to_lexicographic_numbering</code> or
        // <code>FETools::lexicographic_to_hierarchic_numbering</code>.
        using FE_Poly = FE_Poly<TensorProductPolynomials<dim>, dim, spacedim>;
        const FE_Poly &kx_fe_poly = dynamic_cast<const FE_Poly &>(kx_fe);
        const FE_Poly &ky_fe_poly = dynamic_cast<const FE_Poly &>(ky_fe);

        // Polynomial space inverse numbering for recovering the tensor product
        // ordering.
        std::vector<unsigned int> kx_fe_poly_space_numbering_inverse =
          kx_fe_poly.get_poly_space_numbering_inverse();
        std::vector<unsigned int> ky_fe_poly_space_numbering_inverse =
          ky_fe_poly.get_poly_space_numbering_inverse();

        // Quadrature rule to be adopted depending on the cell neighboring type.
        const QGauss<4> *active_quad_rule;

        switch (cell_neighboring_type)
          {
            case SamePanel:
              {
                Assert(vertex_dof_index_intersection.size() ==
                         vertices_per_cell,
                       ExcInternalError());

                // Get support points in tensor product oder.
                kx_support_points_permuted =
                  permute_vector(kx_support_points_hierarchical,
                                 kx_fe_poly_space_numbering_inverse);
                ky_support_points_permuted =
                  permute_vector(ky_support_points_hierarchical,
                                 ky_fe_poly_space_numbering_inverse);

                // Get permuted local DoF indices.
                kx_local_dof_indices_permuted =
                  permute_vector(kx_local_dof_indices_hierarchical,
                                 kx_fe_poly_space_numbering_inverse);
                ky_local_dof_indices_permuted =
                  permute_vector(ky_local_dof_indices_hierarchical,
                                 ky_fe_poly_space_numbering_inverse);

                active_quad_rule = &(bem_values.quad_rule_for_same_panel);


                //                // DEBUG: Print out permuted support points
                //                and DoF indices for
                //                // debugging.
                //                deallog << "Support points and DoF indices in
                //                Kx:\n"; deallog << "DoF_index X Y Z\n"; for
                //                (unsigned int i = 0; i < kx_n_dofs; i++)
                //                  {
                //                    deallog <<
                //                    kx_local_dof_indices_permuted[i] << " "
                //                            << kx_support_points_permuted[i]
                //                            << std::endl;
                //                  }
                //
                //                deallog << "Support points and DoF indices in
                //                Ky:\n"; deallog << "DoF_index X Y Z\n"; for
                //                (unsigned int i = 0; i < ky_n_dofs; i++)
                //                  {
                //                    deallog <<
                //                    ky_local_dof_indices_permuted[i] << " "
                //                            << ky_support_points_permuted[i]
                //                            << std::endl;
                //                  }

                break;
              }
            case CommonEdge:
              {
                // This part handles the common edge case of Sauter's
                // quadrature rule.
                // 1. Get the DoF indices in tensor product order for $K_x$.
                // 2. Get the DoF indices in reversed tensor product order for
                // $K_x$.
                // 3. Extract DoF indices only for cell vertices in $K_x$ and
                // $K_y$. N.B. The DoF indices for the last two vertices are
                // swapped, such that the four vertices are in clockwise or
                // counter clockwise order.
                // 4. Determine the starting vertex.

                Assert(vertex_dof_index_intersection.size() ==
                         GeometryInfo<dim>::vertices_per_face,
                       ExcInternalError());

                kx_local_dof_indices_permuted =
                  permute_vector(kx_local_dof_indices_hierarchical,
                                 kx_fe_poly_space_numbering_inverse);

                std::vector<unsigned int>
                  ky_fe_reversed_poly_space_numbering_inverse =
                    generate_backward_dof_permutation(ky_fe, 0);
                ky_local_dof_indices_permuted =
                  permute_vector(ky_local_dof_indices_hierarchical,
                                 ky_fe_reversed_poly_space_numbering_inverse);

                std::array<types::global_dof_index, vertices_per_cell>
                  kx_local_vertex_dof_indices_swapped =
                    get_vertex_dof_indices_swapped(
                      kx_fe, kx_local_dof_indices_permuted);
                std::array<types::global_dof_index, vertices_per_cell>
                  ky_local_vertex_dof_indices_swapped =
                    get_vertex_dof_indices_swapped(
                      ky_fe, ky_local_dof_indices_permuted);

                // Determine the starting vertex index in $K_x$ and $K_y$.
                unsigned int kx_starting_vertex_index =
                  get_start_vertex_dof_index<vertices_per_cell>(
                    vertex_dof_index_intersection,
                    kx_local_vertex_dof_indices_swapped);
                Assert(kx_starting_vertex_index < vertices_per_cell,
                       ExcInternalError());
                unsigned int ky_starting_vertex_index =
                  get_start_vertex_dof_index<vertices_per_cell>(
                    vertex_dof_index_intersection,
                    ky_local_vertex_dof_indices_swapped);
                Assert(ky_starting_vertex_index < vertices_per_cell,
                       ExcInternalError());

                // Generate the permutation of DoFs in $K_x$ and $K_y$ by
                // starting from <code>kx_starting_vertex_index</code> or
                // <code>ky_starting_vertex_index</code>.
                std::vector<unsigned int> kx_local_dof_permutation =
                  generate_forward_dof_permutation(kx_fe,
                                                   kx_starting_vertex_index);
                std::vector<unsigned int> ky_local_dof_permutation =
                  generate_backward_dof_permutation(ky_fe,
                                                    ky_starting_vertex_index);

                kx_support_points_permuted =
                  permute_vector(kx_support_points_hierarchical,
                                 kx_local_dof_permutation);
                ky_support_points_permuted =
                  permute_vector(ky_support_points_hierarchical,
                                 ky_local_dof_permutation);

                kx_local_dof_indices_permuted =
                  permute_vector(kx_local_dof_indices_hierarchical,
                                 kx_local_dof_permutation);
                ky_local_dof_indices_permuted =
                  permute_vector(ky_local_dof_indices_hierarchical,
                                 ky_local_dof_permutation);

                active_quad_rule = &(bem_values.quad_rule_for_common_edge);


                //                // DEBUG: Print out permuted support points
                //                and DoF indices for
                //                // debugging.
                //                deallog << "Support points and DoF indices in
                //                Kx:\n"; deallog << "DoF_index X Y Z\n"; for
                //                (unsigned int i = 0; i < kx_n_dofs; i++)
                //                  {
                //                    deallog <<
                //                    kx_local_dof_indices_permuted[i] << " "
                //                            << kx_support_points_permuted[i]
                //                            << std::endl;
                //                  }
                //
                //                deallog << "Support points and DoF indices in
                //                Ky:\n"; deallog << "DoF_index X Y Z\n"; for
                //                (unsigned int i = 0; i < ky_n_dofs; i++)
                //                  {
                //                    deallog <<
                //                    ky_local_dof_indices_permuted[i] << " "
                //                            << ky_support_points_permuted[i]
                //                            << std::endl;
                //                  }

                break;
              }
            case CommonVertex:
              {
                Assert(vertex_dof_index_intersection.size() == 1,
                       ExcInternalError());

                kx_local_dof_indices_permuted =
                  permute_vector(kx_local_dof_indices_hierarchical,
                                 kx_fe_poly_space_numbering_inverse);

                ky_local_dof_indices_permuted =
                  permute_vector(ky_local_dof_indices_hierarchical,
                                 ky_fe_poly_space_numbering_inverse);

                std::array<types::global_dof_index, vertices_per_cell>
                  kx_local_vertex_dof_indices_swapped =
                    get_vertex_dof_indices_swapped(
                      kx_fe, kx_local_dof_indices_permuted);
                std::array<types::global_dof_index, vertices_per_cell>
                  ky_local_vertex_dof_indices_swapped =
                    get_vertex_dof_indices_swapped(
                      ky_fe, ky_local_dof_indices_permuted);

                // Determine the starting vertex index in $K_x$ and $K_y$.
                unsigned int kx_starting_vertex_index =
                  get_start_vertex_dof_index<vertices_per_cell>(
                    vertex_dof_index_intersection,
                    kx_local_vertex_dof_indices_swapped);
                Assert(kx_starting_vertex_index < vertices_per_cell,
                       ExcInternalError());
                unsigned int ky_starting_vertex_index =
                  get_start_vertex_dof_index<vertices_per_cell>(
                    vertex_dof_index_intersection,
                    ky_local_vertex_dof_indices_swapped);
                Assert(ky_starting_vertex_index < vertices_per_cell,
                       ExcInternalError());

                // Generate the permutation of DoFs in $K_x$ and $K_y$ by
                // starting from <code>kx_starting_vertex_index</code> or
                // <code>ky_starting_vertex_index</code>.
                std::vector<unsigned int> kx_local_dof_permutation =
                  generate_forward_dof_permutation(kx_fe,
                                                   kx_starting_vertex_index);
                std::vector<unsigned int> ky_local_dof_permutation =
                  generate_forward_dof_permutation(ky_fe,
                                                   ky_starting_vertex_index);

                kx_support_points_permuted =
                  permute_vector(kx_support_points_hierarchical,
                                 kx_local_dof_permutation);
                ky_support_points_permuted =
                  permute_vector(ky_support_points_hierarchical,
                                 ky_local_dof_permutation);

                kx_local_dof_indices_permuted =
                  permute_vector(kx_local_dof_indices_hierarchical,
                                 kx_local_dof_permutation);
                ky_local_dof_indices_permuted =
                  permute_vector(ky_local_dof_indices_hierarchical,
                                 ky_local_dof_permutation);

                active_quad_rule = &(bem_values.quad_rule_for_common_vertex);


                //                // DEBUG: Print out permuted support points
                //                and DoF indices for
                //                // debugging.
                //                deallog << "Support points and DoF indices in
                //                Kx:\n"; deallog << "DoF_index X Y Z\n"; for
                //                (unsigned int i = 0; i < kx_n_dofs; i++)
                //                  {
                //                    deallog <<
                //                    kx_local_dof_indices_permuted[i] << " "
                //                            << kx_support_points_permuted[i]
                //                            << std::endl;
                //                  }
                //
                //                deallog << "Support points and DoF indices in
                //                Ky:\n"; deallog << "DoF_index X Y Z\n"; for
                //                (unsigned int i = 0; i < ky_n_dofs; i++)
                //                  {
                //                    deallog <<
                //                    ky_local_dof_indices_permuted[i] << " "
                //                            << ky_support_points_permuted[i]
                //                            << std::endl;
                //                  }

                break;
              }
            case Regular:
              {
                Assert(vertex_dof_index_intersection.size() == 0,
                       ExcInternalError());

                kx_local_dof_indices_permuted =
                  permute_vector(kx_local_dof_indices_hierarchical,
                                 kx_fe_poly_space_numbering_inverse);

                ky_local_dof_indices_permuted =
                  permute_vector(ky_local_dof_indices_hierarchical,
                                 ky_fe_poly_space_numbering_inverse);


                kx_support_points_permuted =
                  permute_vector(kx_support_points_hierarchical,
                                 kx_fe_poly_space_numbering_inverse);
                ky_support_points_permuted =
                  permute_vector(ky_support_points_hierarchical,
                                 ky_fe_poly_space_numbering_inverse);

                active_quad_rule = &(bem_values.quad_rule_for_regular);


                //                // DEBUG: Print out permuted support points
                //                and DoF indices for
                //                // debugging.
                //                deallog << "Support points and DoF indices in
                //                Kx:\n"; deallog << "DoF_index X Y Z\n"; for
                //                (unsigned int i = 0; i < kx_n_dofs; i++)
                //                  {
                //                    deallog <<
                //                    kx_local_dof_indices_permuted[i] << " "
                //                            << kx_support_points_permuted[i]
                //                            << std::endl;
                //                  }
                //
                //                deallog << "Support points and DoF indices in
                //                Ky:\n"; deallog << "DoF_index X Y Z\n"; for
                //                (unsigned int i = 0; i < ky_n_dofs; i++)
                //                  {
                //                    deallog <<
                //                    ky_local_dof_indices_permuted[i] << " "
                //                            << ky_support_points_permuted[i]
                //                            << std::endl;
                //                  }

                break;
              }
            default:
              {
                Assert(false, ExcNotImplemented());
              }
          }

        // Iterate over DoFs for test function space in tensor product
        // order in $K_x$.
        for (unsigned int i = 0; i < kx_n_dofs; i++)
          {
            // Iterate over DoFs for ansatz function space in tensor
            // product order in $K_y$.
            for (unsigned int j = 0; j < ky_n_dofs; j++)
              {
                // Pullback the kernel function to unit cell.
                KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType>
                  kernel_pullback_on_unit(kernel_function,
                                          cell_neighboring_type,
                                          kx_support_points_permuted,
                                          ky_support_points_permuted,
                                          kx_fe,
                                          ky_fe,
                                          &bem_values,
                                          i,
                                          j);

                // Pullback the kernel function to Sauter parameter
                // space.
                KernelPulledbackToSauterSpace<dim, spacedim, RangeNumberType>
                  kernel_pullback_on_sauter(kernel_pullback_on_unit,
                                            cell_neighboring_type,
                                            &bem_values);

                // Apply 4d Sauter numerical quadrature.
                cell_matrix(i, j) =
                  ApplyQuadratureUsingBEMValues(*active_quad_rule,
                                                kernel_pullback_on_sauter);
              }
          }
      }
    catch (const std::bad_cast &e)
      {
        Assert(false, ExcInternalError());
      }

    return cell_matrix;
  }


  /**
   * Transform parametric coordinates in Sauter's quadrature rule for the same
   * panel case to unit cell coordinates for $K_x$ and $K_y$ respectively.
   * @param parametric_coords
   * @param k3_index
   * @param kx_unit_cell_coords
   * @param ky_unit_cell_coords
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
        (1 - parametric_coords(0) * parametric_coords(1)) *
          parametric_coords(2)};

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
   * Transform parametric coordinates in Sauter's quadrature rule for the common
   * edge case to unit cell coordinates for $K_x$ and $K_y$ respectively.
   * @param parametric_coords
   * @param k3_index
   * @param kx_unit_cell_coords
   * @param ky_unit_cell_coords
   */
  template <int dim>
  void
  sauter_common_edge_parametric_coords_to_unit_cells(
    const Point<dim * 2> &parametric_coords,
    const unsigned int    k3_index,
    Point<dim> &          kx_unit_cell_coords,
    Point<dim> &          ky_unit_cell_coords)
  {
    double unit_coords1[4] = {(1 - parametric_coords(0)) *
                                  parametric_coords(3) +
                                parametric_coords(0),
                              parametric_coords(0) * parametric_coords(2),
                              (1 - parametric_coords(0)) * parametric_coords(3),
                              parametric_coords(0) * parametric_coords(1)};
    double unit_coords2[4] = {
      (1 - parametric_coords(0) * parametric_coords(1)) * parametric_coords(3) +
        parametric_coords(0) * parametric_coords(1),
      parametric_coords(0) * parametric_coords(2),
      (1 - parametric_coords(0) * parametric_coords(1)) * parametric_coords(3),
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
   * Transform parametric coordinates in Sauter's quadrature rule for the common
   * vertex case to unit cell coordinates for $K_x$ and $K_y$ respectively.
   * @param parametric_coords
   * @param k3_index
   * @param kx_unit_cell_coords
   * @param ky_unit_cell_coords
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
   * regular case to unit cell coordinates for $K_x$ and $K_y$ respectively.
   * @param parametric_coords
   * @param k3_index
   * @param kx_unit_cell_coords
   * @param ky_unit_cell_coords
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


  /**
   * Calculate the table storing shape function values at Sauter quadrature
   * points for the same panel case.
   * @param kx_fe finite element for $K_x$
   * @param ky_fe finite element for $K_y$
   * @param sauter_quad_rule
   * @param kx_shape_value_table the 1st dimension is the shape function hierarchical numbering;
   * the 2nd dimension is the index for $k_3$ terms; the 3rd dimension is the
   * quadrature point number.
   * @param ky_shape_value_table the 1st dimension is the shape function hierarchical numbering;
   * the 2nd dimension is the index for $k_3$ terms; the 3rd dimension is the
   * quadrature point number.
   */
  template <int dim, int spacedim, typename RangeNumberType>
  void
  bem_shape_values_same_panel(const FiniteElement<dim, spacedim> &kx_fe,
                              const FiniteElement<dim, spacedim> &ky_fe,
                              const QGauss<4> &          sauter_quad_rule,
                              Table<3, RangeNumberType> &kx_shape_value_table,
                              Table<3, RangeNumberType> &ky_shape_value_table)
  {
    const unsigned int kx_dofs_per_cell = kx_fe.dofs_per_cell;
    const unsigned int ky_dofs_per_cell = ky_fe.dofs_per_cell;
    const unsigned int n_q_points       = sauter_quad_rule.size();

    // Make assertion about the length for each dimension of the data table.
    Assert(kx_shape_value_table.size(0) == kx_dofs_per_cell,
           ExcDimensionMismatch(kx_shape_value_table.size(0),
                                kx_dofs_per_cell));
    Assert(kx_shape_value_table.size(1) == 8,
           ExcDimensionMismatch(kx_shape_value_table.size(1), 8));
    Assert(kx_shape_value_table.size(2) == n_q_points,
           ExcDimensionMismatch(kx_shape_value_table.size(2), n_q_points));

    Assert(ky_shape_value_table.size(0) == ky_dofs_per_cell,
           ExcDimensionMismatch(ky_shape_value_table.size(0),
                                ky_dofs_per_cell));
    Assert(ky_shape_value_table.size(1) == 8,
           ExcDimensionMismatch(ky_shape_value_table.size(1), 8));
    Assert(ky_shape_value_table.size(2) == n_q_points,
           ExcDimensionMismatch(ky_shape_value_table.size(2), n_q_points));

    std::vector<Point<dim * 2>> quad_points = sauter_quad_rule.get_points();

    Point<dim> kx_quad_point;
    Point<dim> ky_quad_point;

    try
      {
        using FE_Poly = FE_Poly<TensorProductPolynomials<dim>, dim, spacedim>;
        const FE_Poly &kx_fe_poly = dynamic_cast<const FE_Poly &>(kx_fe);
        const FE_Poly &ky_fe_poly = dynamic_cast<const FE_Poly &>(ky_fe);

        std::vector<unsigned int> kx_poly_space_inverse_numbering(
          kx_fe_poly.get_poly_space_numbering_inverse());
        std::vector<unsigned int> ky_poly_space_inverse_numbering(
          ky_fe_poly.get_poly_space_numbering_inverse());

        // Iterate over each $k_3$ part.
        for (unsigned k = 0; k < 8; k++)
          {
            // Iterate over each quadrature point.
            for (unsigned int q = 0; q < n_q_points; q++)
              {
                // Transform the quadrature point in the parametric space to the
                // unit cells of $K_x$ and $K_y$.
                sauter_same_panel_parametric_coords_to_unit_cells(
                  quad_points[q], k, kx_quad_point, ky_quad_point);

                // Iterate over each shape function on the unit cell of $K_x$
                // and evaluate it at <code>kx_quad_point</code>. N.B. The shape
                // functions are in the tensor product order.
                for (unsigned int s = 0; s < kx_dofs_per_cell; s++)
                  {
                    kx_shape_value_table(s, k, q) =
                      kx_fe.shape_value(kx_poly_space_inverse_numbering[s],
                                        kx_quad_point);
                  }

                // Iterate over each shape function on the unit cell of $K_y$
                // and evaluate it at <code>ky_quad_point</code>. N.B. The shape
                // functions are in the tensor product order.
                for (unsigned int s = 0; s < ky_dofs_per_cell; s++)
                  {
                    ky_shape_value_table(s, k, q) =
                      ky_fe.shape_value(ky_poly_space_inverse_numbering[s],
                                        ky_quad_point);
                  }
              }
          }
      }
    catch (const std::bad_cast &e)
      {
        Assert(false, ExcInternalError());
      }
  }


  /**
   * Calculate the table storing shape function values at Sauter quadrature
   * points for the common edge case.
   * @param kx_fe finite element for $K_x$
   * @param ky_fe finite element for $K_y$
   * @param sauter_quad_rule
   * @param kx_shape_value_table the 1st dimension is the shape function hierarchical numbering;
   * the 2nd dimension is the index for $k_3$ terms; the 3rd dimension is the
   * quadrature point number.
   * @param ky_shape_value_table the 1st dimension is the shape function hierarchical numbering;
   * the 2nd dimension is the index for $k_3$ terms; the 3rd dimension is the
   * quadrature point number.
   */
  template <int dim, int spacedim, typename RangeNumberType>
  void
  bem_shape_values_common_edge(const FiniteElement<dim, spacedim> &kx_fe,
                               const FiniteElement<dim, spacedim> &ky_fe,
                               const QGauss<4> &          sauter_quad_rule,
                               Table<3, RangeNumberType> &kx_shape_value_table,
                               Table<3, RangeNumberType> &ky_shape_value_table)
  {
    const unsigned int kx_dofs_per_cell = kx_fe.dofs_per_cell;
    const unsigned int ky_dofs_per_cell = ky_fe.dofs_per_cell;
    const unsigned int n_q_points       = sauter_quad_rule.size();

    // Make assertion about the length for each dimension of the data table.
    Assert(kx_shape_value_table.size(0) == kx_dofs_per_cell,
           ExcDimensionMismatch(kx_shape_value_table.size(0),
                                kx_dofs_per_cell));
    Assert(kx_shape_value_table.size(1) == 6,
           ExcDimensionMismatch(kx_shape_value_table.size(1), 6));
    Assert(kx_shape_value_table.size(2) == n_q_points,
           ExcDimensionMismatch(kx_shape_value_table.size(2), n_q_points));

    Assert(ky_shape_value_table.size(0) == ky_dofs_per_cell,
           ExcDimensionMismatch(ky_shape_value_table.size(0),
                                ky_dofs_per_cell));
    Assert(ky_shape_value_table.size(1) == 6,
           ExcDimensionMismatch(ky_shape_value_table.size(1), 6));
    Assert(ky_shape_value_table.size(2) == n_q_points,
           ExcDimensionMismatch(ky_shape_value_table.size(2), n_q_points));

    std::vector<Point<dim * 2>> quad_points = sauter_quad_rule.get_points();

    Point<dim> kx_quad_point;
    Point<dim> ky_quad_point;

    try
      {
        using FE_Poly = FE_Poly<TensorProductPolynomials<dim>, dim, spacedim>;
        const FE_Poly &kx_fe_poly = dynamic_cast<const FE_Poly &>(kx_fe);
        const FE_Poly &ky_fe_poly = dynamic_cast<const FE_Poly &>(ky_fe);

        std::vector<unsigned int> kx_poly_space_inverse_numbering(
          kx_fe_poly.get_poly_space_numbering_inverse());
        std::vector<unsigned int> ky_poly_space_inverse_numbering(
          ky_fe_poly.get_poly_space_numbering_inverse());

        // Iterate over each $k_3$ part.
        for (unsigned k = 0; k < 6; k++)
          {
            // Iterate over each quadrature point.
            for (unsigned int q = 0; q < n_q_points; q++)
              {
                // Transform the quadrature point in the parametric space to the
                // unit cells of $K_x$ and $K_y$.
                sauter_common_edge_parametric_coords_to_unit_cells(
                  quad_points[q], k, kx_quad_point, ky_quad_point);

                // Iterate over each shape function on the unit cell of $K_x$
                // and evaluate it at <code>kx_quad_point</code>. N.B. The shape
                // functions are in the default hierarchical order.
                for (unsigned int s = 0; s < kx_dofs_per_cell; s++)
                  {
                    kx_shape_value_table(s, k, q) =
                      kx_fe.shape_value(kx_poly_space_inverse_numbering[s],
                                        kx_quad_point);
                  }

                // Iterate over each shape function on the unit cell of $K_y$
                // and evaluate it at <code>ky_quad_point</code>. N.B. The shape
                // functions are in the default hierarchical order.
                for (unsigned int s = 0; s < ky_dofs_per_cell; s++)
                  {
                    ky_shape_value_table(s, k, q) =
                      ky_fe.shape_value(ky_poly_space_inverse_numbering[s],
                                        ky_quad_point);
                  }
              }
          }
      }
    catch (const std::bad_cast &e)
      {
        Assert(false, ExcInternalError());
      }
  }


  /**
   * Calculate the table storing shape function values at Sauter quadrature
   * points for the common vertex case.
   * @param kx_fe finite element for $K_x$
   * @param ky_fe finite element for $K_y$
   * @param sauter_quad_rule
   * @param kx_shape_value_table the 1st dimension is the shape function hierarchical numbering;
   * the 2nd dimension is the index for $k_3$ terms; the 3rd dimension is the
   * quadrature point number.
   * @param ky_shape_value_table the 1st dimension is the shape function hierarchical numbering;
   * the 2nd dimension is the index for $k_3$ terms; the 3rd dimension is the
   * quadrature point number.
   */
  template <int dim, int spacedim, typename RangeNumberType>
  void
  bem_shape_values_common_vertex(
    const FiniteElement<dim, spacedim> &kx_fe,
    const FiniteElement<dim, spacedim> &ky_fe,
    const QGauss<4> &                   sauter_quad_rule,
    Table<3, RangeNumberType> &         kx_shape_value_table,
    Table<3, RangeNumberType> &         ky_shape_value_table)
  {
    const unsigned int kx_dofs_per_cell = kx_fe.dofs_per_cell;
    const unsigned int ky_dofs_per_cell = ky_fe.dofs_per_cell;
    const unsigned int n_q_points       = sauter_quad_rule.size();

    // Make assertion about the length for each dimension of the data table.
    Assert(kx_shape_value_table.size(0) == kx_dofs_per_cell,
           ExcDimensionMismatch(kx_shape_value_table.size(0),
                                kx_dofs_per_cell));
    Assert(kx_shape_value_table.size(1) == 4,
           ExcDimensionMismatch(kx_shape_value_table.size(1), 4));
    Assert(kx_shape_value_table.size(2) == n_q_points,
           ExcDimensionMismatch(kx_shape_value_table.size(2), n_q_points));

    Assert(ky_shape_value_table.size(0) == ky_dofs_per_cell,
           ExcDimensionMismatch(ky_shape_value_table.size(0),
                                ky_dofs_per_cell));
    Assert(ky_shape_value_table.size(1) == 4,
           ExcDimensionMismatch(ky_shape_value_table.size(1), 4));
    Assert(ky_shape_value_table.size(2) == n_q_points,
           ExcDimensionMismatch(ky_shape_value_table.size(2), n_q_points));

    std::vector<Point<dim * 2>> quad_points = sauter_quad_rule.get_points();

    Point<dim> kx_quad_point;
    Point<dim> ky_quad_point;

    try
      {
        using FE_Poly = FE_Poly<TensorProductPolynomials<dim>, dim, spacedim>;
        const FE_Poly &kx_fe_poly = dynamic_cast<const FE_Poly &>(kx_fe);
        const FE_Poly &ky_fe_poly = dynamic_cast<const FE_Poly &>(ky_fe);

        std::vector<unsigned int> kx_poly_space_inverse_numbering(
          kx_fe_poly.get_poly_space_numbering_inverse());
        std::vector<unsigned int> ky_poly_space_inverse_numbering(
          ky_fe_poly.get_poly_space_numbering_inverse());

        // Iterate over each $k_3$ part.
        for (unsigned k = 0; k < 4; k++)
          {
            // Iterate over each quadrature point.
            for (unsigned int q = 0; q < n_q_points; q++)
              {
                // Transform the quadrature point in the parametric space to the
                // unit cells of $K_x$ and $K_y$.
                sauter_common_vertex_parametric_coords_to_unit_cells(
                  quad_points[q], k, kx_quad_point, ky_quad_point);

                // Iterate over each shape function on the unit cell of $K_x$
                // and evaluate it at <code>kx_quad_point</code>. N.B. The shape
                // functions are in the default hierarchical order.
                for (unsigned int s = 0; s < kx_dofs_per_cell; s++)
                  {
                    kx_shape_value_table(s, k, q) =
                      kx_fe.shape_value(kx_poly_space_inverse_numbering[s],
                                        kx_quad_point);
                  }

                // Iterate over each shape function on the unit cell of $K_y$
                // and evaluate it at <code>ky_quad_point</code>. N.B. The shape
                // functions are in the default hierarchical order.
                for (unsigned int s = 0; s < ky_dofs_per_cell; s++)
                  {
                    ky_shape_value_table(s, k, q) =
                      ky_fe.shape_value(ky_poly_space_inverse_numbering[s],
                                        ky_quad_point);
                  }
              }
          }
      }
    catch (const std::bad_cast &e)
      {
        Assert(false, ExcInternalError());
      }
  }


  /**
   * Calculate the table storing shape function values at Sauter quadrature
   * points for the regular case.
   * @param kx_fe finite element for $K_x$
   * @param ky_fe finite element for $K_y$
   * @param sauter_quad_rule
   * @param kx_shape_value_table the 1st dimension is the shape function hierarchical numbering;
   * the 2nd dimension is the index for $k_3$ terms; the 3rd dimension is the
   * quadrature point number.
   * @param ky_shape_value_table the 1st dimension is the shape function hierarchical numbering;
   * the 2nd dimension is the index for $k_3$ terms; the 3rd dimension is the
   * quadrature point number.
   */
  template <int dim, int spacedim, typename RangeNumberType>
  void
  bem_shape_values_regular(const FiniteElement<dim, spacedim> &kx_fe,
                           const FiniteElement<dim, spacedim> &ky_fe,
                           const QGauss<4> &                   sauter_quad_rule,
                           Table<3, RangeNumberType> &kx_shape_value_table,
                           Table<3, RangeNumberType> &ky_shape_value_table)
  {
    const unsigned int kx_dofs_per_cell = kx_fe.dofs_per_cell;
    const unsigned int ky_dofs_per_cell = ky_fe.dofs_per_cell;
    const unsigned int n_q_points       = sauter_quad_rule.size();

    // Make assertion about the length for each dimension of the data table.
    Assert(kx_shape_value_table.size(0) == kx_dofs_per_cell,
           ExcDimensionMismatch(kx_shape_value_table.size(0),
                                kx_dofs_per_cell));
    Assert(kx_shape_value_table.size(1) == 1,
           ExcDimensionMismatch(kx_shape_value_table.size(1), 1));
    Assert(kx_shape_value_table.size(2) == n_q_points,
           ExcDimensionMismatch(kx_shape_value_table.size(2), n_q_points));

    Assert(ky_shape_value_table.size(0) == ky_dofs_per_cell,
           ExcDimensionMismatch(ky_shape_value_table.size(0),
                                ky_dofs_per_cell));
    Assert(ky_shape_value_table.size(1) == 1,
           ExcDimensionMismatch(ky_shape_value_table.size(1), 1));
    Assert(ky_shape_value_table.size(2) == n_q_points,
           ExcDimensionMismatch(ky_shape_value_table.size(2), n_q_points));

    std::vector<Point<dim * 2>> quad_points = sauter_quad_rule.get_points();

    Point<dim> kx_quad_point;
    Point<dim> ky_quad_point;

    try
      {
        using FE_Poly = FE_Poly<TensorProductPolynomials<dim>, dim, spacedim>;
        const FE_Poly &kx_fe_poly = dynamic_cast<const FE_Poly &>(kx_fe);
        const FE_Poly &ky_fe_poly = dynamic_cast<const FE_Poly &>(ky_fe);

        std::vector<unsigned int> kx_poly_space_inverse_numbering(
          kx_fe_poly.get_poly_space_numbering_inverse());
        std::vector<unsigned int> ky_poly_space_inverse_numbering(
          ky_fe_poly.get_poly_space_numbering_inverse());

        // Iterate over each quadrature point.
        for (unsigned int q = 0; q < n_q_points; q++)
          {
            // Transform the quadrature point in the parametric space to the
            // unit cells of $K_x$ and $K_y$.
            sauter_regular_parametric_coords_to_unit_cells(quad_points[q],
                                                           kx_quad_point,
                                                           ky_quad_point);

            // Iterate over each shape function on the unit cell of $K_x$ and
            // evaluate it at <code>kx_quad_point</code>. N.B. The shape
            // functions are in the default hierarchical order.
            for (unsigned int s = 0; s < kx_dofs_per_cell; s++)
              {
                kx_shape_value_table(s, 0, q) =
                  kx_fe.shape_value(kx_poly_space_inverse_numbering[s],
                                    kx_quad_point);
              }

            // Iterate over each shape function on $K_y$ and
            // evaluate it at <code>ky_quad_point</code>. N.B. The shape
            // functions are in the default hierarchical order.
            for (unsigned int s = 0; s < ky_dofs_per_cell; s++)
              {
                ky_shape_value_table(s, 0, q) =
                  ky_fe.shape_value(ky_poly_space_inverse_numbering[s],
                                    ky_quad_point);
              }
          }
      }
    catch (const std::bad_cast &e)
      {
        Assert(false, ExcInternalError());
      }
  }


  /**
   * Calculate the table storing shape function gradient matrices at Sauter
   * quadrature points for the same panel case. N.B. The shape functions are
   * in the tensor product order and each row of the gradient matrix corresponds
   * to a shape function.
   * @param kx_fe finite element for $K_x$
   * @param ky_fe finite element for $K_y$
   * @param sauter_quad_rule
   * @param kx_shape_grad_matrix_table the 1st dimension is the index for $k_3$
   * terms; the 2nd dimension is the quadrature point number.
   * @param ky_shape_grad_matrix_table the 1st dimension is the index for $k_3$
   * terms; the 2nd dimension is the quadrature point number.
   */
  template <int dim, int spacedim, typename RangeNumberType>
  void
  bem_shape_grad_matrices_same_panel(
    const FiniteElement<dim, spacedim> &   kx_fe,
    const FiniteElement<dim, spacedim> &   ky_fe,
    const QGauss<4> &                      sauter_quad_rule,
    Table<2, FullMatrix<RangeNumberType>> &kx_shape_grad_matrix_table,
    Table<2, FullMatrix<RangeNumberType>> &ky_shape_grad_matrix_table)
  {
    const unsigned int n_q_points = sauter_quad_rule.size();

    // Make assertion about the length for each dimension of the data table.
    Assert(kx_shape_grad_matrix_table.size(0) == 8,
           ExcDimensionMismatch(kx_shape_grad_matrix_table.size(0), 8));
    Assert(kx_shape_grad_matrix_table.size(1) == n_q_points,
           ExcDimensionMismatch(kx_shape_grad_matrix_table.size(1),
                                n_q_points));

    Assert(ky_shape_grad_matrix_table.size(0) == 8,
           ExcDimensionMismatch(ky_shape_grad_matrix_table.size(0), 8));
    Assert(ky_shape_grad_matrix_table.size(1) == n_q_points,
           ExcDimensionMismatch(ky_shape_grad_matrix_table.size(1),
                                n_q_points));

    std::vector<Point<4>> quad_points = sauter_quad_rule.get_points();

    Point<dim> kx_quad_point;
    Point<dim> ky_quad_point;

    // Iterate over each $k_3$ part.
    for (unsigned k = 0; k < 8; k++)
      {
        // Iterate over each quadrature point.
        for (unsigned int q = 0; q < n_q_points; q++)
          {
            // Transform the quadrature point in the parametric space to the
            // unit cells of $K_x$ and $K_y$.
            sauter_same_panel_parametric_coords_to_unit_cells(quad_points[q],
                                                              k,
                                                              kx_quad_point,
                                                              ky_quad_point);

            // Calculate the gradient matrix evaluated at
            // <code>kx_quad_point</code> in  $K_x$. Matrix rows correspond to
            // shape functions which are in the tensor product order.
            kx_shape_grad_matrix_table(k, q) =
              shape_grad_matrix_in_tensor_product_order(kx_fe, kx_quad_point);
            // Calculate the gradient matrix evaluated at
            // <code>ky_quad_point</code> in  $K_y$. Matrix rows correspond to
            // shape functions which are in the tensor product order.
            ky_shape_grad_matrix_table(k, q) =
              shape_grad_matrix_in_tensor_product_order(ky_fe, ky_quad_point);
          }
      }
  }


  /**
   * Calculate the table storing shape function gradient matrices at Sauter
   * quadrature points for the common edge case. N.B. The shape functions are
   * in the tensor product order and each row of the gradient matrix corresponds
   * to a shape function.
   * @param kx_fe finite element for $K_x$
   * @param ky_fe finite element for $K_y$
   * @param sauter_quad_rule
   * @param kx_shape_grad_matrix_table the 1st dimension is the index for $k_3$
   * terms; the 2nd dimension is the quadrature point number.
   * @param ky_shape_grad_matrix_table the 1st dimension is the index for $k_3$
   * terms; the 2nd dimension is the quadrature point number.
   */
  template <int dim, int spacedim, typename RangeNumberType>
  void
  bem_shape_grad_matrices_common_edge(
    const FiniteElement<dim, spacedim> &   kx_fe,
    const FiniteElement<dim, spacedim> &   ky_fe,
    const QGauss<4> &                      sauter_quad_rule,
    Table<2, FullMatrix<RangeNumberType>> &kx_shape_grad_matrix_table,
    Table<2, FullMatrix<RangeNumberType>> &ky_shape_grad_matrix_table)
  {
    const unsigned int n_q_points = sauter_quad_rule.size();

    // Make assertion about the length for each dimension of the data table.
    Assert(kx_shape_grad_matrix_table.size(0) == 6,
           ExcDimensionMismatch(kx_shape_grad_matrix_table.size(0), 6));
    Assert(kx_shape_grad_matrix_table.size(1) == n_q_points,
           ExcDimensionMismatch(kx_shape_grad_matrix_table.size(1),
                                n_q_points));

    Assert(ky_shape_grad_matrix_table.size(0) == 6,
           ExcDimensionMismatch(ky_shape_grad_matrix_table.size(0), 6));
    Assert(ky_shape_grad_matrix_table.size(1) == n_q_points,
           ExcDimensionMismatch(ky_shape_grad_matrix_table.size(1),
                                n_q_points));

    std::vector<Point<4>> quad_points = sauter_quad_rule.get_points();

    Point<dim> kx_quad_point;
    Point<dim> ky_quad_point;

    // Iterate over each $k_3$ part.
    for (unsigned k = 0; k < 6; k++)
      {
        // Iterate over each quadrature point.
        for (unsigned int q = 0; q < n_q_points; q++)
          {
            // Transform the quadrature point in the parametric space to the
            // unit cells of $K_x$ and $K_y$.
            sauter_common_edge_parametric_coords_to_unit_cells(quad_points[q],
                                                               k,
                                                               kx_quad_point,
                                                               ky_quad_point);

            // Calculate the gradient matrix evaluated at
            // <code>kx_quad_point</code> in  $K_x$. Matrix rows correspond to
            // shape functions which are in the tensor product order.
            kx_shape_grad_matrix_table(k, q) =
              shape_grad_matrix_in_tensor_product_order(kx_fe, kx_quad_point);
            // Calculate the gradient matrix evaluated at
            // <code>ky_quad_point</code> in  $K_y$. Matrix rows correspond to
            // shape functions which are in the tensor product order.
            ky_shape_grad_matrix_table(k, q) =
              shape_grad_matrix_in_tensor_product_order(ky_fe, ky_quad_point);
          }
      }
  }


  /**
   * Calculate the table storing shape function gradient matrices at Sauter
   * quadrature points for the common vertex case. N.B. The shape functions are
   * in the tensor product order and each row of the gradient matrix corresponds
   * to a shape function.
   * @param kx_fe finite element for $K_x$
   * @param ky_fe finite element for $K_y$
   * @param sauter_quad_rule
   * @param kx_shape_grad_matrix_table the 1st dimension is the index for $k_3$
   * terms; the 2nd dimension is the quadrature point number.
   * @param ky_shape_grad_matrix_table the 1st dimension is the index for $k_3$
   * terms; the 2nd dimension is the quadrature point number.
   */
  template <int dim, int spacedim, typename RangeNumberType>
  void
  bem_shape_grad_matrices_common_vertex(
    const FiniteElement<dim, spacedim> &   kx_fe,
    const FiniteElement<dim, spacedim> &   ky_fe,
    const QGauss<4> &                      sauter_quad_rule,
    Table<2, FullMatrix<RangeNumberType>> &kx_shape_grad_matrix_table,
    Table<2, FullMatrix<RangeNumberType>> &ky_shape_grad_matrix_table)
  {
    const unsigned int n_q_points = sauter_quad_rule.size();

    // Make assertion about the length for each dimension of the data table.
    Assert(kx_shape_grad_matrix_table.size(0) == 4,
           ExcDimensionMismatch(kx_shape_grad_matrix_table.size(0), 4));
    Assert(kx_shape_grad_matrix_table.size(1) == n_q_points,
           ExcDimensionMismatch(kx_shape_grad_matrix_table.size(1),
                                n_q_points));

    Assert(ky_shape_grad_matrix_table.size(0) == 4,
           ExcDimensionMismatch(ky_shape_grad_matrix_table.size(0), 4));
    Assert(ky_shape_grad_matrix_table.size(1) == n_q_points,
           ExcDimensionMismatch(ky_shape_grad_matrix_table.size(1),
                                n_q_points));

    std::vector<Point<4>> quad_points = sauter_quad_rule.get_points();

    Point<dim> kx_quad_point;
    Point<dim> ky_quad_point;

    // Iterate over each $k_3$ part.
    for (unsigned k = 0; k < 4; k++)
      {
        // Iterate over each quadrature point.
        for (unsigned int q = 0; q < n_q_points; q++)
          {
            // Transform the quadrature point in the parametric space to the
            // unit cells of $K_x$ and $K_y$.
            sauter_common_vertex_parametric_coords_to_unit_cells(quad_points[q],
                                                                 k,
                                                                 kx_quad_point,
                                                                 ky_quad_point);

            // Calculate the gradient matrix evaluated at
            // <code>kx_quad_point</code> in  $K_x$. Matrix rows correspond to
            // shape functions which are in the tensor product order.
            kx_shape_grad_matrix_table(k, q) =
              shape_grad_matrix_in_tensor_product_order(kx_fe, kx_quad_point);
            // Calculate the gradient matrix evaluated at
            // <code>ky_quad_point</code> in  $K_y$. Matrix rows correspond to
            // shape functions which are in the tensor product order.
            ky_shape_grad_matrix_table(k, q) =
              shape_grad_matrix_in_tensor_product_order(ky_fe, ky_quad_point);
          }
      }
  }


  /**
   * Calculate the table storing shape function gradient matrices at Sauter
   * quadrature points for the regular case. N.B. The shape functions are
   * in the tensor product order and each row of the gradient matrix corresponds
   * to a shape function.
   * @param kx_fe finite element for $K_x$
   * @param ky_fe finite element for $K_y$
   * @param sauter_quad_rule
   * @param kx_shape_grad_matrix_table the 1st dimension is the quadrature point
   * number.
   * @param ky_shape_grad_matrix_table the 1st dimension is the quadrature point
   * number.
   */
  template <int dim, int spacedim, typename RangeNumberType>
  void
  bem_shape_grad_matrices_regular(
    const FiniteElement<dim, spacedim> &   kx_fe,
    const FiniteElement<dim, spacedim> &   ky_fe,
    const QGauss<4> &                      sauter_quad_rule,
    Table<2, FullMatrix<RangeNumberType>> &kx_shape_grad_matrix_table,
    Table<2, FullMatrix<RangeNumberType>> &ky_shape_grad_matrix_table)
  {
    const unsigned int n_q_points = sauter_quad_rule.size();

    // Make assertion about the length for each dimension of the data table.
    Assert(kx_shape_grad_matrix_table.size(0) == 1,
           ExcDimensionMismatch(kx_shape_grad_matrix_table.size(0), 1));
    Assert(kx_shape_grad_matrix_table.size(1) == n_q_points,
           ExcDimensionMismatch(kx_shape_grad_matrix_table.size(1),
                                n_q_points));

    Assert(ky_shape_grad_matrix_table.size(0) == 1,
           ExcDimensionMismatch(ky_shape_grad_matrix_table.size(0), 1));
    Assert(ky_shape_grad_matrix_table.size(1) == n_q_points,
           ExcDimensionMismatch(ky_shape_grad_matrix_table.size(1),
                                n_q_points));

    std::vector<Point<4>> quad_points = sauter_quad_rule.get_points();

    Point<dim> kx_quad_point;
    Point<dim> ky_quad_point;

    // Iterate over each quadrature point.
    for (unsigned int q = 0; q < n_q_points; q++)
      {
        // Transform the quadrature point in the parametric space to the
        // unit cells of $K_x$ and $K_y$.
        sauter_regular_parametric_coords_to_unit_cells(quad_points[q],
                                                       kx_quad_point,
                                                       ky_quad_point);

        // Calculate the gradient matrix evaluated at
        // <code>kx_quad_point</code> in  $K_x$. Matrix rows correspond to
        // shape functions which are in the tensor product order.
        kx_shape_grad_matrix_table(0, q) =
          shape_grad_matrix_in_tensor_product_order(kx_fe, kx_quad_point);
        // Calculate the gradient matrix evaluated at
        // <code>ky_quad_point</code> in  $K_y$. Matrix rows correspond to
        // shape functions which are in the tensor product order.
        ky_shape_grad_matrix_table(0, q) =
          shape_grad_matrix_in_tensor_product_order(ky_fe, ky_quad_point);
      }
  }


  /**
   * This function implements Sauter's quadrature rule on quadrangular mesh.
   * It handles various cases including same panel, common edge, common vertex
   * and regular cell neighboring types.
   *
   * @param kernel_function Laplace kernel function.
   * @param kx_cell_iter Iterator pointing to $K_x$.
   * @param kx_cell_iter Iterator pointing to $K_y$.
   * @param kx_mapping Mapping used for $K_x$. Because a mesher usually generates
   * 1st order grid, if there is no additional manifold specification, the
   * mapping should be 1st order.
   * @param kx_mapping Mapping used for $K_y$. Because a mesher usually generates
   * 1st order grid, if there is no additional manifold specification, the
   * mapping should be 1st order.
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  SauterQuadRule(
    FullMatrix<RangeNumberType> &system_matrix,
    const LaplaceKernel::KernelFunction<spacedim, RangeNumberType>
      &                                                      kernel_function,
    const typename DoFHandler<dim, spacedim>::cell_iterator &kx_cell_iter,
    const typename DoFHandler<dim, spacedim>::cell_iterator &ky_cell_iter,
    const MappingQGeneric<dim, spacedim> &                   kx_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const MappingQGeneric<dim, spacedim> &ky_mapping =
      MappingQGeneric<dim, spacedim>(1))
  {
    // Geometry information.
    const unsigned int vertices_per_cell = GeometryInfo<dim>::vertices_per_cell;

    // Determine the cell neighboring type based on the vertex dof indices.
    // The common dof indices will be stored into the vector
    // <code>vertex_dof_index_intersection</code> if there is any.
    std::array<types::global_dof_index, vertices_per_cell>
      kx_vertex_dof_indices(
        get_vertex_dof_indices<dim, spacedim>(kx_cell_iter));
    std::array<types::global_dof_index, vertices_per_cell>
      ky_vertex_dof_indices(
        get_vertex_dof_indices<dim, spacedim>(ky_cell_iter));

    std::vector<types::global_dof_index> vertex_dof_index_intersection;
    vertex_dof_index_intersection.reserve(vertices_per_cell);
    CellNeighboringType cell_neighboring_type =
      detect_cell_neighboring_type<dim>(kx_vertex_dof_indices,
                                        ky_vertex_dof_indices,
                                        vertex_dof_index_intersection);


    const FiniteElement<dim, spacedim> &kx_fe = kx_cell_iter->get_fe();
    const FiniteElement<dim, spacedim> &ky_fe = ky_cell_iter->get_fe();

    const unsigned int kx_n_dofs = kx_fe.dofs_per_cell;
    const unsigned int ky_n_dofs = ky_fe.dofs_per_cell;

    // Support points of $K_x$ and $K_y$ in the default
    // hierarchical order.
    std::vector<Point<spacedim>> kx_support_points_hierarchical =
      hierarchical_support_points_in_real_cell(kx_cell_iter, kx_fe, kx_mapping);
    std::vector<Point<spacedim>> ky_support_points_hierarchical =
      hierarchical_support_points_in_real_cell(ky_cell_iter, ky_fe, ky_mapping);

    // Permuted support points to be used in the common edge and common vertex
    // cases instead of the original support points in hierarchical order.
    std::vector<Point<spacedim>> kx_support_points_permuted;
    std::vector<Point<spacedim>> ky_support_points_permuted;
    kx_support_points_permuted.reserve(kx_n_dofs);
    ky_support_points_permuted.reserve(ky_n_dofs);

    // Global indices for the local DoFs in the default hierarchical order.
    // N.B. These vectors should have the right size before being passed to
    // the function <code>get_dof_indices</code>.
    std::vector<types::global_dof_index> kx_local_dof_indices_hierarchical(
      kx_n_dofs);
    std::vector<types::global_dof_index> ky_local_dof_indices_hierarchical(
      ky_n_dofs);
    kx_cell_iter->get_dof_indices(kx_local_dof_indices_hierarchical);
    ky_cell_iter->get_dof_indices(ky_local_dof_indices_hierarchical);

    // Permuted local DoF indices, which has the same permutation as that
    // applied to support points.
    std::vector<types::global_dof_index> kx_local_dof_indices_permuted;
    std::vector<types::global_dof_index> ky_local_dof_indices_permuted;
    kx_local_dof_indices_permuted.reserve(kx_n_dofs);
    ky_local_dof_indices_permuted.reserve(ky_n_dofs);

    // Generate 4D Gauss-Legendre quadrature rules for various cell
    // neighboring types.
    const unsigned int quad_order_for_same_panel    = 5;
    const unsigned int quad_order_for_common_edge   = 4;
    const unsigned int quad_order_for_common_vertex = 4;
    const unsigned int quad_order_for_regular       = 3;

    QGauss<4> quad_rule_for_same_panel(quad_order_for_same_panel);
    QGauss<4> quad_rule_for_common_edge(quad_order_for_common_edge);
    QGauss<4> quad_rule_for_common_vertex(quad_order_for_common_vertex);
    QGauss<4> quad_rule_for_regular(quad_order_for_regular);

    FullMatrix<RangeNumberType> cell_matrix(kx_n_dofs, ky_n_dofs);

    try
      {
        // Downcast references of FiniteElement objects to FE_Poly references
        // for obtaining the inverse polynomial space numbering.
        // TODO: Inverse polynomial space numbering may be obtained by calling
        // <code>FETools::hierarchic_to_lexicographic_numbering</code> or
        // <code>FETools::lexicographic_to_hierarchic_numbering</code>.
        using FE_Poly = FE_Poly<TensorProductPolynomials<dim>, dim, spacedim>;
        const FE_Poly &kx_fe_poly = dynamic_cast<const FE_Poly &>(kx_fe);
        const FE_Poly &ky_fe_poly = dynamic_cast<const FE_Poly &>(ky_fe);

        // Polynomial space inverse numbering for recovering the tensor
        // product ordering.
        std::vector<unsigned int> kx_fe_poly_space_numbering_inverse =
          kx_fe_poly.get_poly_space_numbering_inverse();
        std::vector<unsigned int> ky_fe_poly_space_numbering_inverse =
          ky_fe_poly.get_poly_space_numbering_inverse();

        switch (cell_neighboring_type)
          {
            case SamePanel:
              {
                Assert(vertex_dof_index_intersection.size() ==
                         vertices_per_cell,
                       ExcInternalError());

                // Get support points in tensor product oder.
                kx_support_points_permuted =
                  permute_vector(kx_support_points_hierarchical,
                                 kx_fe_poly_space_numbering_inverse);
                ky_support_points_permuted =
                  permute_vector(ky_support_points_hierarchical,
                                 ky_fe_poly_space_numbering_inverse);

                // Get permuted local DoF indices.
                kx_local_dof_indices_permuted =
                  permute_vector(kx_local_dof_indices_hierarchical,
                                 kx_fe_poly_space_numbering_inverse);
                ky_local_dof_indices_permuted =
                  permute_vector(ky_local_dof_indices_hierarchical,
                                 ky_fe_poly_space_numbering_inverse);


                //                // DEBUG: Print out permuted support points
                //                and DoF indices for
                //                // debugging.
                //                deallog << "Support points and DoF indices
                //                in Kx:\n"; deallog << "DoF_index X Y Z\n";
                //                for (unsigned int i = 0; i < kx_n_dofs; i++)
                //                  {
                //                    deallog <<
                //                    kx_local_dof_indices_permuted[i] << " "
                //                            << kx_support_points_permuted[i]
                //                            << std::endl;
                //                  }
                //
                //                deallog << "Support points and DoF indices
                //                in Ky:\n"; deallog << "DoF_index X Y Z\n";
                //                for (unsigned int i = 0; i < ky_n_dofs; i++)
                //                  {
                //                    deallog <<
                //                    ky_local_dof_indices_permuted[i] << " "
                //                            << ky_support_points_permuted[i]
                //                            << std::endl;
                //                  }


                // Iterate over DoFs for test function space in tensor product
                // order in $K_x$.
                for (unsigned int i = 0; i < kx_n_dofs; i++)
                  {
                    // Iterate over DoFs for ansatz function space in tensor
                    // product order in $K_y$.
                    for (unsigned int j = 0; j < ky_n_dofs; j++)
                      {
                        // Pullback the kernel function to unit cell.
                        KernelPulledbackToUnitCell<dim,
                                                   spacedim,
                                                   RangeNumberType>
                          kernel_pullback_on_unit(kernel_function,
                                                  cell_neighboring_type,
                                                  kx_support_points_permuted,
                                                  ky_support_points_permuted,
                                                  kx_fe,
                                                  ky_fe,
                                                  i,
                                                  j);

                        // Pullback the kernel function to Sauter parameter
                        // space.
                        KernelPulledbackToSauterSpace<dim,
                                                      spacedim,
                                                      RangeNumberType>
                          kernel_pullback_on_sauter(kernel_pullback_on_unit,
                                                    cell_neighboring_type);

                        // Apply 4d Sauter numerical quadrature.
                        cell_matrix(i, j) =
                          ApplyQuadrature(quad_rule_for_same_panel,
                                          kernel_pullback_on_sauter);
                      }
                  }

                break;
              }
            case CommonEdge:
              {
                // This part handles the common edge case of Sauter's
                // quadrature rule.
                // 1. Get the DoF indices in tensor product order for $K_x$.
                // 2. Get the DoF indices in reversed tensor product order for
                // $K_x$.
                // 3. Extract DoF indices only for cell vertices in $K_x$ and
                // $K_y$. N.B. The DoF indices for the last two vertices are
                // swapped, such that the four vertices are in clockwise or
                // counter clockwise order.
                // 4. Determine the starting vertex.

                Assert(vertex_dof_index_intersection.size() ==
                         GeometryInfo<dim>::vertices_per_face,
                       ExcInternalError());

                kx_local_dof_indices_permuted =
                  permute_vector(kx_local_dof_indices_hierarchical,
                                 kx_fe_poly_space_numbering_inverse);

                std::vector<unsigned int>
                  ky_fe_reversed_poly_space_numbering_inverse =
                    generate_backward_dof_permutation(ky_fe, 0);
                ky_local_dof_indices_permuted =
                  permute_vector(ky_local_dof_indices_hierarchical,
                                 ky_fe_reversed_poly_space_numbering_inverse);

                std::array<types::global_dof_index, vertices_per_cell>
                  kx_local_vertex_dof_indices_swapped =
                    get_vertex_dof_indices_swapped(
                      kx_fe, kx_local_dof_indices_permuted);
                std::array<types::global_dof_index, vertices_per_cell>
                  ky_local_vertex_dof_indices_swapped =
                    get_vertex_dof_indices_swapped(
                      ky_fe, ky_local_dof_indices_permuted);

                // Determine the starting vertex index in $K_x$ and $K_y$.
                unsigned int kx_starting_vertex_index =
                  get_start_vertex_dof_index<vertices_per_cell>(
                    vertex_dof_index_intersection,
                    kx_local_vertex_dof_indices_swapped);
                Assert(kx_starting_vertex_index < vertices_per_cell,
                       ExcInternalError());
                unsigned int ky_starting_vertex_index =
                  get_start_vertex_dof_index<vertices_per_cell>(
                    vertex_dof_index_intersection,
                    ky_local_vertex_dof_indices_swapped);
                Assert(ky_starting_vertex_index < vertices_per_cell,
                       ExcInternalError());

                // Generate the permutation of DoFs in $K_x$ and $K_y$ by
                // starting from <code>kx_starting_vertex_index</code> or
                // <code>ky_starting_vertex_index</code>.
                std::vector<unsigned int> kx_local_dof_permutation =
                  generate_forward_dof_permutation(kx_fe,
                                                   kx_starting_vertex_index);
                std::vector<unsigned int> ky_local_dof_permutation =
                  generate_backward_dof_permutation(ky_fe,
                                                    ky_starting_vertex_index);

                kx_support_points_permuted =
                  permute_vector(kx_support_points_hierarchical,
                                 kx_local_dof_permutation);
                ky_support_points_permuted =
                  permute_vector(ky_support_points_hierarchical,
                                 ky_local_dof_permutation);

                kx_local_dof_indices_permuted =
                  permute_vector(kx_local_dof_indices_hierarchical,
                                 kx_local_dof_permutation);
                ky_local_dof_indices_permuted =
                  permute_vector(ky_local_dof_indices_hierarchical,
                                 ky_local_dof_permutation);


                //                // DEBUG: Print out permuted support points
                //                and DoF indices for
                //                // debugging.
                //                deallog << "Support points and DoF indices
                //                in Kx:\n"; deallog << "DoF_index X Y Z\n";
                //                for (unsigned int i = 0; i < kx_n_dofs; i++)
                //                  {
                //                    deallog <<
                //                    kx_local_dof_indices_permuted[i] << " "
                //                            << kx_support_points_permuted[i]
                //                            << std::endl;
                //                  }
                //
                //                deallog << "Support points and DoF indices
                //                in Ky:\n"; deallog << "DoF_index X Y Z\n";
                //                for (unsigned int i = 0; i < ky_n_dofs; i++)
                //                  {
                //                    deallog <<
                //                    ky_local_dof_indices_permuted[i] << " "
                //                            << ky_support_points_permuted[i]
                //                            << std::endl;
                //                  }


                // Iterate over DoFs for test function space in tensor product
                // order in $K_x$.
                for (unsigned int i = 0; i < kx_n_dofs; i++)
                  {
                    // Iterate over DoFs for ansatz function space in tensor
                    // product order in $K_y$.
                    for (unsigned int j = 0; j < ky_n_dofs; j++)
                      {
                        // Pullback the kernel function to unit cell.
                        KernelPulledbackToUnitCell<dim,
                                                   spacedim,
                                                   RangeNumberType>
                          kernel_pullback_on_unit(kernel_function,
                                                  cell_neighboring_type,
                                                  kx_support_points_permuted,
                                                  ky_support_points_permuted,
                                                  kx_fe,
                                                  ky_fe,
                                                  i,
                                                  j);

                        // Pullback the kernel function to Sauter parameter
                        // space.
                        KernelPulledbackToSauterSpace<dim,
                                                      spacedim,
                                                      RangeNumberType>
                          kernel_pullback_on_sauter(kernel_pullback_on_unit,
                                                    cell_neighboring_type);

                        // Apply 4d Sauter numerical quadrature.
                        cell_matrix(i, j) =
                          ApplyQuadrature(quad_rule_for_common_edge,
                                          kernel_pullback_on_sauter);
                      }
                  }

                break;
              }
            case CommonVertex:
              {
                Assert(vertex_dof_index_intersection.size() == 1,
                       ExcInternalError());

                kx_local_dof_indices_permuted =
                  permute_vector(kx_local_dof_indices_hierarchical,
                                 kx_fe_poly_space_numbering_inverse);

                ky_local_dof_indices_permuted =
                  permute_vector(ky_local_dof_indices_hierarchical,
                                 ky_fe_poly_space_numbering_inverse);

                std::array<types::global_dof_index, vertices_per_cell>
                  kx_local_vertex_dof_indices_swapped =
                    get_vertex_dof_indices_swapped(
                      kx_fe, kx_local_dof_indices_permuted);
                std::array<types::global_dof_index, vertices_per_cell>
                  ky_local_vertex_dof_indices_swapped =
                    get_vertex_dof_indices_swapped(
                      ky_fe, ky_local_dof_indices_permuted);

                // Determine the starting vertex index in $K_x$ and $K_y$.
                unsigned int kx_starting_vertex_index =
                  get_start_vertex_dof_index<vertices_per_cell>(
                    vertex_dof_index_intersection,
                    kx_local_vertex_dof_indices_swapped);
                Assert(kx_starting_vertex_index < vertices_per_cell,
                       ExcInternalError());
                unsigned int ky_starting_vertex_index =
                  get_start_vertex_dof_index<vertices_per_cell>(
                    vertex_dof_index_intersection,
                    ky_local_vertex_dof_indices_swapped);
                Assert(ky_starting_vertex_index < vertices_per_cell,
                       ExcInternalError());

                // Generate the permutation of DoFs in $K_x$ and $K_y$ by
                // starting from <code>kx_starting_vertex_index</code> or
                // <code>ky_starting_vertex_index</code>.
                std::vector<unsigned int> kx_local_dof_permutation =
                  generate_forward_dof_permutation(kx_fe,
                                                   kx_starting_vertex_index);
                std::vector<unsigned int> ky_local_dof_permutation =
                  generate_forward_dof_permutation(ky_fe,
                                                   ky_starting_vertex_index);

                kx_support_points_permuted =
                  permute_vector(kx_support_points_hierarchical,
                                 kx_local_dof_permutation);
                ky_support_points_permuted =
                  permute_vector(ky_support_points_hierarchical,
                                 ky_local_dof_permutation);

                kx_local_dof_indices_permuted =
                  permute_vector(kx_local_dof_indices_hierarchical,
                                 kx_local_dof_permutation);
                ky_local_dof_indices_permuted =
                  permute_vector(ky_local_dof_indices_hierarchical,
                                 ky_local_dof_permutation);


                //                // DEBUG: Print out permuted support points
                //                and DoF indices for
                //                // debugging.
                //                deallog << "Support points and DoF indices
                //                in Kx:\n"; deallog << "DoF_index X Y Z\n";
                //                for (unsigned int i = 0; i < kx_n_dofs; i++)
                //                  {
                //                    deallog <<
                //                    kx_local_dof_indices_permuted[i] << " "
                //                            << kx_support_points_permuted[i]
                //                            << std::endl;
                //                  }
                //
                //                deallog << "Support points and DoF indices
                //                in Ky:\n"; deallog << "DoF_index X Y Z\n";
                //                for (unsigned int i = 0; i < ky_n_dofs; i++)
                //                  {
                //                    deallog <<
                //                    ky_local_dof_indices_permuted[i] << " "
                //                            << ky_support_points_permuted[i]
                //                            << std::endl;
                //                  }


                // Iterate over DoFs for test function space in tensor product
                // order in $K_x$.
                for (unsigned int i = 0; i < kx_n_dofs; i++)
                  {
                    // Iterate over DoFs for ansatz function space in tensor
                    // product order in $K_y$.
                    for (unsigned int j = 0; j < ky_n_dofs; j++)
                      {
                        // Pullback the kernel function to unit cell.
                        KernelPulledbackToUnitCell<dim,
                                                   spacedim,
                                                   RangeNumberType>
                          kernel_pullback_on_unit(kernel_function,
                                                  cell_neighboring_type,
                                                  kx_support_points_permuted,
                                                  ky_support_points_permuted,
                                                  kx_fe,
                                                  ky_fe,
                                                  i,
                                                  j);

                        // Pullback the kernel function to Sauter parameter
                        // space.
                        KernelPulledbackToSauterSpace<dim,
                                                      spacedim,
                                                      RangeNumberType>
                          kernel_pullback_on_sauter(kernel_pullback_on_unit,
                                                    cell_neighboring_type);

                        // Apply 4d Sauter numerical quadrature.
                        cell_matrix(i, j) =
                          ApplyQuadrature(quad_rule_for_common_vertex,
                                          kernel_pullback_on_sauter);
                      }
                  }

                break;
              }
            case Regular:
              {
                Assert(vertex_dof_index_intersection.size() == 0,
                       ExcInternalError());

                kx_local_dof_indices_permuted =
                  permute_vector(kx_local_dof_indices_hierarchical,
                                 kx_fe_poly_space_numbering_inverse);

                ky_local_dof_indices_permuted =
                  permute_vector(ky_local_dof_indices_hierarchical,
                                 ky_fe_poly_space_numbering_inverse);


                kx_support_points_permuted =
                  permute_vector(kx_support_points_hierarchical,
                                 kx_fe_poly_space_numbering_inverse);
                ky_support_points_permuted =
                  permute_vector(ky_support_points_hierarchical,
                                 ky_fe_poly_space_numbering_inverse);


                //                // DEBUG: Print out permuted support points
                //                and DoF indices for
                //                // debugging.
                //                deallog << "Support points and DoF indices
                //                in Kx:\n"; deallog << "DoF_index X Y Z\n";
                //                for (unsigned int i = 0; i < kx_n_dofs; i++)
                //                  {
                //                    deallog <<
                //                    kx_local_dof_indices_permuted[i] << " "
                //                            << kx_support_points_permuted[i]
                //                            << std::endl;
                //                  }
                //
                //                deallog << "Support points and DoF indices
                //                in Ky:\n"; deallog << "DoF_index X Y Z\n";
                //                for (unsigned int i = 0; i < ky_n_dofs; i++)
                //                  {
                //                    deallog <<
                //                    ky_local_dof_indices_permuted[i] << " "
                //                            << ky_support_points_permuted[i]
                //                            << std::endl;
                //                  }


                // Iterate over DoFs for test function space in tensor product
                // order in $K_x$.
                for (unsigned int i = 0; i < kx_n_dofs; i++)
                  {
                    // Iterate over DoFs for ansatz function space in tensor
                    // product order in $K_y$.
                    for (unsigned int j = 0; j < ky_n_dofs; j++)
                      {
                        // Pullback the kernel function to unit cell.
                        KernelPulledbackToUnitCell<dim,
                                                   spacedim,
                                                   RangeNumberType>
                          kernel_pullback_on_unit(kernel_function,
                                                  cell_neighboring_type,
                                                  kx_support_points_permuted,
                                                  ky_support_points_permuted,
                                                  kx_fe,
                                                  ky_fe,
                                                  i,
                                                  j);

                        // Pullback the kernel function to Sauter parameter
                        // space.
                        KernelPulledbackToSauterSpace<dim,
                                                      spacedim,
                                                      RangeNumberType>
                          kernel_pullback_on_sauter(kernel_pullback_on_unit,
                                                    cell_neighboring_type);

                        // Apply 4d Sauter numerical quadrature.
                        cell_matrix(i, j) =
                          ApplyQuadrature(quad_rule_for_regular,
                                          kernel_pullback_on_sauter);
                      }
                  }

                break;
              }
            default:
              {
                Assert(false, ExcNotImplemented());
              }
          }
      }
    catch (const std::bad_cast &e)
      {
        Assert(false, ExcInternalError());
      }

    // Assemble the cell matrix to system matrix.
    for (unsigned int i = 0; i < kx_n_dofs; i++)
      {
        for (unsigned int j = 0; j < ky_n_dofs; j++)
          {
            system_matrix.add(kx_local_dof_indices_permuted[i],
                              ky_local_dof_indices_permuted[j],
                              cell_matrix(i, j));
          }
      }
  }


  /**
   * This function implements Sauter's quadrature rule on quadrangular mesh.
   * It handles various cases including same panel, common edge, common vertex
   * and regular cell neighboring types.
   *
   * @param kernel_function Laplace kernel function.
   * @param kx_cell_iter Iterator pointing to $K_x$.
   * @param kx_cell_iter Iterator pointing to $K_y$.
   * @param kx_mapping Mapping used for $K_x$. Because a mesher usually generates
   * 1st order grid, if there is no additional manifold specification, the
   * mapping should be 1st order.
   * @param kx_mapping Mapping used for $K_y$. Because a mesher usually generates
   * 1st order grid, if there is no additional manifold specification, the
   * mapping should be 1st order.
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  SauterQuadRule(
    FullMatrix<RangeNumberType> &system_matrix,
    const LaplaceKernel::KernelFunction<spacedim, RangeNumberType>
      &                                                      kernel_function,
    const BEMValues<dim, spacedim> &                         bem_values,
    const typename DoFHandler<dim, spacedim>::cell_iterator &kx_cell_iter,
    const typename DoFHandler<dim, spacedim>::cell_iterator &ky_cell_iter,
    const MappingQGeneric<dim, spacedim> &                   kx_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const MappingQGeneric<dim, spacedim> &ky_mapping =
      MappingQGeneric<dim, spacedim>(1))
  {
    // Geometry information.
    const unsigned int vertices_per_cell = GeometryInfo<dim>::vertices_per_cell;

    // Determine the cell neighboring type based on the vertex dof indices.
    // The common dof indices will be stored into the vector
    // <code>vertex_dof_index_intersection</code> if there is any.
    std::array<types::global_dof_index, vertices_per_cell>
      kx_vertex_dof_indices(
        get_vertex_dof_indices<dim, spacedim>(kx_cell_iter));
    std::array<types::global_dof_index, vertices_per_cell>
      ky_vertex_dof_indices(
        get_vertex_dof_indices<dim, spacedim>(ky_cell_iter));

    std::vector<types::global_dof_index> vertex_dof_index_intersection;
    vertex_dof_index_intersection.reserve(vertices_per_cell);
    CellNeighboringType cell_neighboring_type =
      detect_cell_neighboring_type<dim>(kx_vertex_dof_indices,
                                        ky_vertex_dof_indices,
                                        vertex_dof_index_intersection);


    const FiniteElement<dim, spacedim> &kx_fe = kx_cell_iter->get_fe();
    const FiniteElement<dim, spacedim> &ky_fe = ky_cell_iter->get_fe();

    const unsigned int kx_n_dofs = kx_fe.dofs_per_cell;
    const unsigned int ky_n_dofs = ky_fe.dofs_per_cell;

    // Support points of $K_x$ and $K_y$ in the default
    // hierarchical order.
    std::vector<Point<spacedim>> kx_support_points_hierarchical =
      hierarchical_support_points_in_real_cell(kx_cell_iter, kx_fe, kx_mapping);
    std::vector<Point<spacedim>> ky_support_points_hierarchical =
      hierarchical_support_points_in_real_cell(ky_cell_iter, ky_fe, ky_mapping);

    // Permuted support points to be used in the common edge and common vertex
    // cases instead of the original support points in hierarchical order.
    std::vector<Point<spacedim>> kx_support_points_permuted;
    std::vector<Point<spacedim>> ky_support_points_permuted;
    kx_support_points_permuted.reserve(kx_n_dofs);
    ky_support_points_permuted.reserve(ky_n_dofs);

    // Global indices for the local DoFs in the default hierarchical order.
    // N.B. These vectors should have the right size before being passed to
    // the function <code>get_dof_indices</code>.
    std::vector<types::global_dof_index> kx_local_dof_indices_hierarchical(
      kx_n_dofs);
    std::vector<types::global_dof_index> ky_local_dof_indices_hierarchical(
      ky_n_dofs);
    kx_cell_iter->get_dof_indices(kx_local_dof_indices_hierarchical);
    ky_cell_iter->get_dof_indices(ky_local_dof_indices_hierarchical);

    // Permuted local DoF indices, which has the same permutation as that
    // applied to support points.
    std::vector<types::global_dof_index> kx_local_dof_indices_permuted;
    std::vector<types::global_dof_index> ky_local_dof_indices_permuted;
    kx_local_dof_indices_permuted.reserve(kx_n_dofs);
    ky_local_dof_indices_permuted.reserve(ky_n_dofs);

    FullMatrix<RangeNumberType> cell_matrix(kx_n_dofs, ky_n_dofs);

    try
      {
        // Downcast references of FiniteElement objects to FE_Poly references
        // for obtaining the inverse polynomial space numbering.
        // TODO: Inverse polynomial space numbering may be obtained by calling
        // <code>FETools::hierarchic_to_lexicographic_numbering</code> or
        // <code>FETools::lexicographic_to_hierarchic_numbering</code>.
        using FE_Poly = FE_Poly<TensorProductPolynomials<dim>, dim, spacedim>;
        const FE_Poly &kx_fe_poly = dynamic_cast<const FE_Poly &>(kx_fe);
        const FE_Poly &ky_fe_poly = dynamic_cast<const FE_Poly &>(ky_fe);

        // Polynomial space inverse numbering for recovering the tensor
        // product ordering.
        std::vector<unsigned int> kx_fe_poly_space_numbering_inverse =
          kx_fe_poly.get_poly_space_numbering_inverse();
        std::vector<unsigned int> ky_fe_poly_space_numbering_inverse =
          ky_fe_poly.get_poly_space_numbering_inverse();

        // Quadrature rule to be adopted depending on the cell neighboring type.
        const QGauss<4> *active_quad_rule = nullptr;

        switch (cell_neighboring_type)
          {
            case SamePanel:
              {
                Assert(vertex_dof_index_intersection.size() ==
                         vertices_per_cell,
                       ExcInternalError());

                // Get support points in tensor product oder.
                kx_support_points_permuted =
                  permute_vector(kx_support_points_hierarchical,
                                 kx_fe_poly_space_numbering_inverse);
                ky_support_points_permuted =
                  permute_vector(ky_support_points_hierarchical,
                                 ky_fe_poly_space_numbering_inverse);

                // Get permuted local DoF indices.
                kx_local_dof_indices_permuted =
                  permute_vector(kx_local_dof_indices_hierarchical,
                                 kx_fe_poly_space_numbering_inverse);
                ky_local_dof_indices_permuted =
                  permute_vector(ky_local_dof_indices_hierarchical,
                                 ky_fe_poly_space_numbering_inverse);

                active_quad_rule = &(bem_values.quad_rule_for_same_panel);


                //                // DEBUG: Print out permuted support points
                //                and DoF indices for
                //                // debugging.
                //                deallog << "Support points and DoF indices
                //                in Kx:\n"; deallog << "DoF_index X Y Z\n";
                //                for (unsigned int i = 0; i < kx_n_dofs; i++)
                //                  {
                //                    deallog <<
                //                    kx_local_dof_indices_permuted[i] << " "
                //                            << kx_support_points_permuted[i]
                //                            << std::endl;
                //                  }
                //
                //                deallog << "Support points and DoF indices
                //                in Ky:\n"; deallog << "DoF_index X Y Z\n";
                //                for (unsigned int i = 0; i < ky_n_dofs; i++)
                //                  {
                //                    deallog <<
                //                    ky_local_dof_indices_permuted[i] << " "
                //                            << ky_support_points_permuted[i]
                //                            << std::endl;
                //                  }

                break;
              }
            case CommonEdge:
              {
                // This part handles the common edge case of Sauter's
                // quadrature rule.
                // 1. Get the DoF indices in tensor product order for $K_x$.
                // 2. Get the DoF indices in reversed tensor product order for
                // $K_x$.
                // 3. Extract DoF indices only for cell vertices in $K_x$ and
                // $K_y$. N.B. The DoF indices for the last two vertices are
                // swapped, such that the four vertices are in clockwise or
                // counter clockwise order.
                // 4. Determine the starting vertex.

                Assert(vertex_dof_index_intersection.size() ==
                         GeometryInfo<dim>::vertices_per_face,
                       ExcInternalError());

                kx_local_dof_indices_permuted =
                  permute_vector(kx_local_dof_indices_hierarchical,
                                 kx_fe_poly_space_numbering_inverse);

                std::vector<unsigned int>
                  ky_fe_reversed_poly_space_numbering_inverse =
                    generate_backward_dof_permutation(ky_fe, 0);
                ky_local_dof_indices_permuted =
                  permute_vector(ky_local_dof_indices_hierarchical,
                                 ky_fe_reversed_poly_space_numbering_inverse);

                std::array<types::global_dof_index, vertices_per_cell>
                  kx_local_vertex_dof_indices_swapped =
                    get_vertex_dof_indices_swapped(
                      kx_fe, kx_local_dof_indices_permuted);
                std::array<types::global_dof_index, vertices_per_cell>
                  ky_local_vertex_dof_indices_swapped =
                    get_vertex_dof_indices_swapped(
                      ky_fe, ky_local_dof_indices_permuted);

                // Determine the starting vertex index in $K_x$ and $K_y$.
                unsigned int kx_starting_vertex_index =
                  get_start_vertex_dof_index<vertices_per_cell>(
                    vertex_dof_index_intersection,
                    kx_local_vertex_dof_indices_swapped);
                Assert(kx_starting_vertex_index < vertices_per_cell,
                       ExcInternalError());
                unsigned int ky_starting_vertex_index =
                  get_start_vertex_dof_index<vertices_per_cell>(
                    vertex_dof_index_intersection,
                    ky_local_vertex_dof_indices_swapped);
                Assert(ky_starting_vertex_index < vertices_per_cell,
                       ExcInternalError());

                // Generate the permutation of DoFs in $K_x$ and $K_y$ by
                // starting from <code>kx_starting_vertex_index</code> or
                // <code>ky_starting_vertex_index</code>.
                std::vector<unsigned int> kx_local_dof_permutation =
                  generate_forward_dof_permutation(kx_fe,
                                                   kx_starting_vertex_index);
                std::vector<unsigned int> ky_local_dof_permutation =
                  generate_backward_dof_permutation(ky_fe,
                                                    ky_starting_vertex_index);

                kx_support_points_permuted =
                  permute_vector(kx_support_points_hierarchical,
                                 kx_local_dof_permutation);
                ky_support_points_permuted =
                  permute_vector(ky_support_points_hierarchical,
                                 ky_local_dof_permutation);

                kx_local_dof_indices_permuted =
                  permute_vector(kx_local_dof_indices_hierarchical,
                                 kx_local_dof_permutation);
                ky_local_dof_indices_permuted =
                  permute_vector(ky_local_dof_indices_hierarchical,
                                 ky_local_dof_permutation);

                active_quad_rule = &(bem_values.quad_rule_for_common_edge);

                //                // DEBUG: Print out permuted support points
                //                and DoF indices for
                //                // debugging.
                //                deallog << "Support points and DoF indices
                //                in Kx:\n"; deallog << "DoF_index X Y Z\n";
                //                for (unsigned int i = 0; i < kx_n_dofs; i++)
                //                  {
                //                    deallog <<
                //                    kx_local_dof_indices_permuted[i] << " "
                //                            << kx_support_points_permuted[i]
                //                            << std::endl;
                //                  }
                //
                //                deallog << "Support points and DoF indices
                //                in Ky:\n"; deallog << "DoF_index X Y Z\n";
                //                for (unsigned int i = 0; i < ky_n_dofs; i++)
                //                  {
                //                    deallog <<
                //                    ky_local_dof_indices_permuted[i] << " "
                //                            << ky_support_points_permuted[i]
                //                            << std::endl;
                //                  }

                break;
              }
            case CommonVertex:
              {
                Assert(vertex_dof_index_intersection.size() == 1,
                       ExcInternalError());

                kx_local_dof_indices_permuted =
                  permute_vector(kx_local_dof_indices_hierarchical,
                                 kx_fe_poly_space_numbering_inverse);

                ky_local_dof_indices_permuted =
                  permute_vector(ky_local_dof_indices_hierarchical,
                                 ky_fe_poly_space_numbering_inverse);

                std::array<types::global_dof_index, vertices_per_cell>
                  kx_local_vertex_dof_indices_swapped =
                    get_vertex_dof_indices_swapped(
                      kx_fe, kx_local_dof_indices_permuted);
                std::array<types::global_dof_index, vertices_per_cell>
                  ky_local_vertex_dof_indices_swapped =
                    get_vertex_dof_indices_swapped(
                      ky_fe, ky_local_dof_indices_permuted);

                // Determine the starting vertex index in $K_x$ and $K_y$.
                unsigned int kx_starting_vertex_index =
                  get_start_vertex_dof_index<vertices_per_cell>(
                    vertex_dof_index_intersection,
                    kx_local_vertex_dof_indices_swapped);
                Assert(kx_starting_vertex_index < vertices_per_cell,
                       ExcInternalError());
                unsigned int ky_starting_vertex_index =
                  get_start_vertex_dof_index<vertices_per_cell>(
                    vertex_dof_index_intersection,
                    ky_local_vertex_dof_indices_swapped);
                Assert(ky_starting_vertex_index < vertices_per_cell,
                       ExcInternalError());

                // Generate the permutation of DoFs in $K_x$ and $K_y$ by
                // starting from <code>kx_starting_vertex_index</code> or
                // <code>ky_starting_vertex_index</code>.
                std::vector<unsigned int> kx_local_dof_permutation =
                  generate_forward_dof_permutation(kx_fe,
                                                   kx_starting_vertex_index);
                std::vector<unsigned int> ky_local_dof_permutation =
                  generate_forward_dof_permutation(ky_fe,
                                                   ky_starting_vertex_index);

                kx_support_points_permuted =
                  permute_vector(kx_support_points_hierarchical,
                                 kx_local_dof_permutation);
                ky_support_points_permuted =
                  permute_vector(ky_support_points_hierarchical,
                                 ky_local_dof_permutation);

                kx_local_dof_indices_permuted =
                  permute_vector(kx_local_dof_indices_hierarchical,
                                 kx_local_dof_permutation);
                ky_local_dof_indices_permuted =
                  permute_vector(ky_local_dof_indices_hierarchical,
                                 ky_local_dof_permutation);

                active_quad_rule = &(bem_values.quad_rule_for_common_vertex);

                //                // DEBUG: Print out permuted support points
                //                and DoF indices for
                //                // debugging.
                //                deallog << "Support points and DoF indices
                //                in Kx:\n"; deallog << "DoF_index X Y Z\n";
                //                for (unsigned int i = 0; i < kx_n_dofs; i++)
                //                  {
                //                    deallog <<
                //                    kx_local_dof_indices_permuted[i] << " "
                //                            << kx_support_points_permuted[i]
                //                            << std::endl;
                //                  }
                //
                //                deallog << "Support points and DoF indices
                //                in Ky:\n"; deallog << "DoF_index X Y Z\n";
                //                for (unsigned int i = 0; i < ky_n_dofs; i++)
                //                  {
                //                    deallog <<
                //                    ky_local_dof_indices_permuted[i] << " "
                //                            << ky_support_points_permuted[i]
                //                            << std::endl;
                //                  }

                break;
              }
            case Regular:
              {
                Assert(vertex_dof_index_intersection.size() == 0,
                       ExcInternalError());

                kx_local_dof_indices_permuted =
                  permute_vector(kx_local_dof_indices_hierarchical,
                                 kx_fe_poly_space_numbering_inverse);

                ky_local_dof_indices_permuted =
                  permute_vector(ky_local_dof_indices_hierarchical,
                                 ky_fe_poly_space_numbering_inverse);


                kx_support_points_permuted =
                  permute_vector(kx_support_points_hierarchical,
                                 kx_fe_poly_space_numbering_inverse);
                ky_support_points_permuted =
                  permute_vector(ky_support_points_hierarchical,
                                 ky_fe_poly_space_numbering_inverse);

                active_quad_rule = &(bem_values.quad_rule_for_regular);

                //                // DEBUG: Print out permuted support points
                //                and DoF indices for
                //                // debugging.
                //                deallog << "Support points and DoF indices
                //                in Kx:\n"; deallog << "DoF_index X Y Z\n";
                //                for (unsigned int i = 0; i < kx_n_dofs; i++)
                //                  {
                //                    deallog <<
                //                    kx_local_dof_indices_permuted[i] << " "
                //                            << kx_support_points_permuted[i]
                //                            << std::endl;
                //                  }
                //
                //                deallog << "Support points and DoF indices
                //                in Ky:\n"; deallog << "DoF_index X Y Z\n";
                //                for (unsigned int i = 0; i < ky_n_dofs; i++)
                //                  {
                //                    deallog <<
                //                    ky_local_dof_indices_permuted[i] << " "
                //                            << ky_support_points_permuted[i]
                //                            << std::endl;
                //                  }

                break;
              }
            default:
              {
                Assert(false, ExcNotImplemented());
                active_quad_rule = nullptr;
              }
          }

        // Iterate over DoFs for test function space in tensor product
        // order in $K_x$.
        for (unsigned int i = 0; i < kx_n_dofs; i++)
          {
            // Iterate over DoFs for ansatz function space in tensor
            // product order in $K_y$.
            for (unsigned int j = 0; j < ky_n_dofs; j++)
              {
                // Pullback the kernel function to unit cell.
                KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType>
                  kernel_pullback_on_unit(kernel_function,
                                          cell_neighboring_type,
                                          kx_support_points_permuted,
                                          ky_support_points_permuted,
                                          kx_fe,
                                          ky_fe,
                                          &bem_values,
                                          i,
                                          j);

                // Pullback the kernel function to Sauter parameter
                // space.
                KernelPulledbackToSauterSpace<dim, spacedim, RangeNumberType>
                  kernel_pullback_on_sauter(kernel_pullback_on_unit,
                                            cell_neighboring_type,
                                            &bem_values);

                // Apply 4d Sauter numerical quadrature.
                cell_matrix(i, j) =
                  ApplyQuadratureUsingBEMValues(*active_quad_rule,
                                                kernel_pullback_on_sauter);
              }
          }
      }
    catch (const std::bad_cast &e)
      {
        Assert(false, ExcInternalError());
      }

    // Assemble the cell matrix to system matrix.
    for (unsigned int i = 0; i < kx_n_dofs; i++)
      {
        for (unsigned int j = 0; j < ky_n_dofs; j++)
          {
            system_matrix.add(kx_local_dof_indices_permuted[i],
                              ky_local_dof_indices_permuted[j],
                              cell_matrix(i, j));
          }
      }
  }
} // namespace LaplaceBEM

/**
 * @}
 */

#endif /* INCLUDE_LAPLACE_BEM_H_ */
