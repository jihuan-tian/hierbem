/**
 * @file bem_tools.h
 * @brief Introduction of bem_tools.h
 *
 * @date 2022-03-03
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_BEM_BEM_TOOLS_H_
#define HIERBEM_INCLUDE_BEM_BEM_TOOLS_H_

#include <deal.II/base/derivative_form.h>
#include <deal.II/base/point.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/base/table.h>
#include <deal.II/base/table_indices.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_data.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/full_matrix.templates.h>

#include <assert.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <utility>
#include <vector>

#include "config.h"
#include "linear_algebra/lapack_full_matrix_ext.h"
#include "mapping/mapping_q_ext.h"
#include "utilities/generic_functors.h"

HBEM_NS_OPEN

using namespace dealii;

namespace BEMTools
{
  /**
   * Different cell neighboring types
   */
  enum CellNeighboringType
  {
    SamePanel,    //!< SamePanel
    CommonEdge,   //!< CommonEdge
    CommonVertex, //!< CommonVertex
    Regular,      //!< Regular
    None          //!< None
  };


  /**
   * Different scenarios for detecting cell neighboring types
   */
  enum DetectCellNeighboringTypeMethod
  {
    SameTriangulations,      //!< SameTriangulations
    DifferentTriangulations, //!< DifferentTriangulations
  };


  /**
   * Get the string representation of the cell neighboring type.
   *
   * @param s
   * @return
   */
  inline const char *
  cell_neighboring_type_name(CellNeighboringType n)
  {
    switch (n)
      {
        case SamePanel:
          return "same panel";
        case CommonEdge:
          return "common edge";
        case CommonVertex:
          return "common vertex";
        case Regular:
          return "disjoint";
        default:
          return "unknown";
      }
  }


  /**
   * @brief Permute a vector by using the given permutation indices to access
   * its elements.
   *
   * @tparam VectorType1
   * @tparam VectorType2
   * @tparam IndexType
   * @param input_vector
   * @param permutation_indices
   * @param permuted_vector Result vector, whose memory should be allocated
   * before calling this function. Its size should be >= the size of the input
   * vector.
   */
  template <typename VectorType1, typename VectorType2, typename IndexType>
  void
  permute_vector(const VectorType1            &input_vector,
                 const std::vector<IndexType> &permutation_indices,
                 VectorType2                  &permuted_vector)
  {
    const IndexType N = input_vector.size();
    AssertDimension(N, permutation_indices.size());
    Assert(N <= permuted_vector.size(), ExcInternalError());

    for (IndexType i = 0; i < N; i++)
      {
        permuted_vector[i] = input_vector[permutation_indices[i]];
      }
  }


  /**
   * This function returns a list of DoF indices in the given cell iterator,
   * which is used for checking if the two cells have interaction. This
   * function is called by @p GraphColoring::make_graph_coloring.
   *
   * Reference:
   * http://localhost/dealii-9.1.1-doc/namespaceGraphColoring.html#a670720d11f544a762592112ae5213876
   *
   * @param cell
   * @return
   */
  template <int dim, int spacedim = dim>
  std::vector<types::global_dof_index>
  get_conflict_indices(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell)
  {
    std::vector<types::global_dof_index> conflict_indices(
      cell->get_fe().dofs_per_cell);

    cell->get_dof_indices(conflict_indices);

    return conflict_indices;
  }


  /**
   * Get the local index of the given vertex in the list of vertices of the
   * cell by raw comparison of vertex coordinates.
   *
   * \mynote{The template parameter @p dim cannot be deduced from the
   * arguments.}
   *
   * @param v
   * @param cell
   * @return
   */
  template <int dim, int spacedim = dim>
  unsigned int
  get_vertex_local_index_in_cell(
    const Point<spacedim>                                      &v,
    const typename Triangulation<dim, spacedim>::cell_iterator &cell)
  {
    const unsigned int vertices_per_cell = GeometryInfo<dim>::vertices_per_cell;

    unsigned int i = 0;
    for (; i < vertices_per_cell; i++)
      {
        if (is_equal(v, cell->vertex(i)))
          {
            break;
          }
      }

    return i;
  }


  /**
   * Get the local index of the given vertex in the list of vertices of the
   * cell by numerical comparison of vertex coordinates.
   *
   * \mynote{The template parameter @p dim cannot be deduced from the
   * arguments.}
   *
   * @param v
   * @param cell
   * @return
   */
  template <int dim, int spacedim = dim>
  unsigned int
  get_vertex_local_index_in_cell(
    const Point<spacedim>                                      &v,
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const double                                                threshold)
  {
    unsigned int i = 0;
    for (; i < GeometryInfo<dim>::vertices_per_cell; i++)
      {
        if (is_equal(v, cell->vertex(i), threshold))
          {
            break;
          }
      }

    return i;
  }


  /**
   * Return a list of global vertex indices for all the vertices in the given
   * cell pointed by the cell iterator obtained from a triangulation. The
   * result is obtained via the return value.
   *
   * @param cell
   * @return
   */
  template <int dim, int spacedim = dim>
  std::array<types::global_vertex_index, GeometryInfo<dim>::vertices_per_cell>
  get_vertex_indices_in_cell(
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


  /**
   * Return a list of global vertex indices for all the vertices in the given
   * cell pointed by the cell iterator obtained from a triangulation. The
   * result is obtained via argument by reference.
   *
   * @param cell
   * @param cell_vertex_indices
   */
  template <int dim, int spacedim = dim>
  void
  get_vertex_indices_in_cell(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    std::array<types::global_vertex_index, GeometryInfo<dim>::vertices_per_cell>
      &cell_vertex_indices)
  {
    for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; v++)
      {
        cell_vertex_indices[v] = cell->vertex_index(v);
      }
  }


  /**
   * Return a list of global vertex indices for all the vertices in the given
   * face pointed by the face iterator obtained from a triangulation. The
   * result is obtained via the return value.
   *
   * \mynote{The dimension of the face is @p dim-1.}
   *
   * @param face
   * @return
   */
  template <int dim, int spacedim = dim>
  std::array<types::global_vertex_index, GeometryInfo<dim>::vertices_per_face>
  get_vertex_indices_in_face(
    const typename Triangulation<dim, spacedim>::face_iterator &face)
  {
    std::array<types::global_vertex_index, GeometryInfo<dim>::vertices_per_face>
      face_vertex_indices;

    for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; v++)
      {
        face_vertex_indices[v] = face->vertex_index(v);
      }

    return face_vertex_indices;
  }


  /**
   * Return a list of global vertex indices for all the vertices in the given
   * face pointed by the face iterator obtained from a triangulation. The
   * result is obtained via argument by reference.
   *
   * \mynote{The dimension of the face is @p dim-1.}
   *
   * @param face
   * @param face_vertex_indices
   */
  template <int dim, int spacedim = dim>
  void
  get_vertex_indices_in_face(
    const typename Triangulation<dim, spacedim>::face_iterator &face,
    std::array<types::global_vertex_index, GeometryInfo<dim>::vertices_per_face>
      &face_vertex_indices)
  {
    for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; v++)
      {
        face_vertex_indices[v] = face->vertex_index(v);
      }
  }


  /**
   * Calculate the distance between the centers of two cells.
   *
   * @param first_cell
   * @param second_cell
   * @return
   */
  template <int dim, int spacedim, typename Number = double>
  Number
  cell_distance(
    const typename Triangulation<dim, spacedim>::cell_iterator first_cell,
    const typename Triangulation<dim, spacedim>::cell_iterator second_cell)
  {
    return first_cell->center().distance(second_cell->center());
  }


  /**
   * Calculates the matrix which stores shape function gradient values with
   * respect to area coordinates. Each row of the matrix is the gradient of
   * one of the shape functions. The order of the matrix rows corresponding to
   * the shape function gradients is determined by the given numbering
   * @p dof_permuation.
   *
   * \mynote{The support points, shape functions and DoFs in the finite
   * element are enumerated in the hierarchic order for the continuous
   * element @p FE_Q, while the discontinuous element @p FE_DGQ adopts the
   * lexicographic order.}
   *
   * @param fe
   * @param dof_permutation The numbering for accessing the shape functions in
   * the specified order.
   * @param p The area coordinates at which the shape function's gradient is
   * to be evaluated.
   * @return The matrix storing the gradient of each shape function. Its
   * dimension is @p dofs_per_cell*dim.
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  LAPACKFullMatrixExt<RangeNumberType>
  shape_grad_matrix(const FiniteElement<dim, spacedim> &fe,
                    const std::vector<unsigned int>    &dof_permutation,
                    const Point<dim>                   &p)
  {
    LAPACKFullMatrixExt<RangeNumberType> grad_matrix(fe.dofs_per_cell, dim);

    for (unsigned int i = 0; i < fe.dofs_per_cell; i++)
      {
        // The gradient of a shape function as a tensor.
        Tensor<1, dim> shape_grad_tensor = fe.shape_grad(dof_permutation[i], p);
        // Assign the gradient of the shape function to the matrix. The value
        // returned from @p fe.shape_grad has double type, which will be cast
        // to the result type @p RangeNumberType . It is possible that
        // @p RangeNumberType is float, when we need a low precision simulation.
        for (unsigned int j = 0; j < dim; j++)
          grad_matrix(i, j) =
            static_cast<RangeNumberType>(shape_grad_tensor[j]);
      }

    return grad_matrix;
  }


  /**
   * Collect the gradient of @p MappingQ shape functions into a matrix.
   *
   * The shape functions and their derivatives of the mapping object have been
   * evaluated at a list of points in the unit cell, therefore we need to
   * specify at which point we will collect the data.
   *
   * @param mapping_data The @p InternalData within @p MappingQ .
   * @param quad_no  The index of the point in the unit cell
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  mappingq_shape_grad_matrix(
    const typename MappingQ<dim, spacedim>::InternalData &mapping_data,
    const unsigned int                                    point_no,
    LAPACKFullMatrixExt<RangeNumberType>                 &grad_matrix)
  {
    AssertDimension(dim, grad_matrix.n());
    Assert(mapping_data.n_shape_functions >= grad_matrix.m(),
           ExcInternalError());

    // Iterate over each effective shape function of the mapping.
    for (unsigned int s = 0; s < grad_matrix.m(); s++)
      {
        // Iterate over each manifold dimension.
        for (unsigned int d = 0; d < dim; d++)
          {
            grad_matrix(s, d) = static_cast<RangeNumberType>(
              mapping_data.derivative(point_no, s)[d]);
          }
      }
  }


  /**
   * Calculate the matrix which stores shape function gradient values with
   * respect to area coordinates. Each row of the matrix is the gradient of
   * one of the shape functions. The matrix rows corresponding to the shape
   * function gradients are arranged in the default DoF order.
   *
   * \mynote{The support points, shape functions and DoFs in the finite
   * element are enumerated in the hierarchic order for the continuous
   * element @p FE_Q, while the discontinuous element @p FE_DGQ adopts the
   * lexicographic order.}
   *
   * @param fe
   * @param p The area coordinates at which the shape function's gradient is
   * to be evaluated.
   * @return The matrix storing the gradient of each shape function. Its
   * dimension is @p dofs_per_cell*dim.
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  LAPACKFullMatrixExt<RangeNumberType>
  shape_grad_matrix_in_default_dof_order(const FiniteElement<dim, spacedim> &fe,
                                         const Point<dim>                   &p)
  {
    LAPACKFullMatrixExt<RangeNumberType> shape_grad_matrix(fe.dofs_per_cell,
                                                           dim);

    for (unsigned int i = 0; i < fe.dofs_per_cell; i++)
      {
        // The gradient of a shape function as a tensor.
        Tensor<1, dim> shape_grad_tensor = fe.shape_grad(i, p);
        // Assign the gradient of the shape function to the matrix.
        for (unsigned int j = 0; j < dim; j++)
          shape_grad_matrix(i, j) =
            static_cast<RangeNumberType>(shape_grad_tensor[j]);
      }

    return shape_grad_matrix;
  }


  /**
   * Calculate the matrix which stores shape function gradient values with
   * respect to area coordinates. Each row of the matrix is the gradient of
   * one of the shape functions. The matrix rows corresponding to the shape
   * function gradients are arranged in the lexicographic order.
   *
   * \mynote{The support points, shape functions and DoFs in the finite
   * element are enumerated in the hierarchic order for the continuous
   * element @p FE_Q, while the discontinuous element @p FE_DGQ adopts the
   * lexicographic order.}
   *
   * @param fe
   * @param p The area coordinates at which the shape function's gradient is
   * to be evaluated.
   * @return The matrix storing the gradient of each shape function. Its
   * dimension is @p dofs_per_cell*dim .
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  LAPACKFullMatrixExt<RangeNumberType>
  shape_grad_matrix_in_lexicographic_order(
    const FiniteElement<dim, spacedim> &fe,
    const Point<dim>                   &p)
  {
    using FE_Poly_short          = FE_Poly<dim, spacedim>;
    const FE_Poly_short &fe_poly = dynamic_cast<const FE_Poly_short &>(fe);

    return shape_grad_matrix<dim, spacedim, RangeNumberType>(
      fe, fe_poly.get_poly_space_numbering_inverse(), p);
  }


  /**
   * Evaluate a list of shape functions at the specified area coordinates. The
   * shape functions are arranged in the order specified by @p dof_permutation.
   *
   * @param fe
   * @param dof_permutation The numbering for accessing the shape functions in
   * the specified order.
   * @param p The area coordinates at which the shape functions are to be
   * evaluated.
   * @return A list of shape function values.
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  Vector<RangeNumberType>
  shape_values(const FiniteElement<dim, spacedim> &fe,
               const std::vector<unsigned int>    &dof_permutation,
               const Point<dim>                   &p)
  {
    Vector<RangeNumberType> shape_values_vector(fe.dofs_per_cell);

    for (unsigned int i = 0; i < fe.dofs_per_cell; i++)
      shape_values_vector(i) =
        static_cast<RangeNumberType>(fe.shape_value(dof_permutation[i], p));

    return shape_values_vector;
  }


  /**
   * Evaluate a list of shape functions at the specified area coordinates in
   * the default DoF order.
   *
   * \mynote{The support points, shape functions and DoFs in the finite
   * element are enumerated in the hierarchic order for the continuous
   * element @p FE_Q, while the discontinuous element @p FE_DGQ adopts the
   * lexicographic order.}
   *
   * @param fe
   * @param p The area coordinates at which the shape functions are to be
   * evaluated.
   * @return A list of shape function values.
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  Vector<RangeNumberType>
  shape_values_in_default_dof_order(const FiniteElement<dim, spacedim> &fe,
                                    const Point<dim>                   &p)
  {
    Vector<RangeNumberType> shape_values_vector(fe.dofs_per_cell);

    for (unsigned int i = 0; i < fe.dofs_per_cell; i++)
      shape_values_vector(i) =
        static_cast<RangeNumberType>(fe.shape_value(i, p));

    return shape_values_vector;
  }


  /**
   * Evaluate a list of shape functions at the specified area coordinates in
   * the lexicographic order.
   *
   * \mynote{The support points, shape functions and DoFs in the finite
   * element are enumerated in the hierarchic order for the continuous
   * element @p FE_Q, while the discontinuous element @p FE_DGQ adopts the
   * lexicographic order.}
   *
   * @param fe
   * @param p The area coordinates at which the shape functions are to be
   * evaluated.
   * @return A list of shape function values.
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  Vector<RangeNumberType>
  shape_values_in_lexicographic_order(const FiniteElement<dim, spacedim> &fe,
                                      const Point<dim>                   &p)
  {
    using FE_Poly_short          = FE_Poly<dim, spacedim>;
    const FE_Poly_short &fe_poly = dynamic_cast<const FE_Poly_short &>(fe);

    // Use the inverse numbering of polynomial space to restore the tensor
    // product ordering of shape functions.
    return shape_values<dim, spacedim, RangeNumberType>(
      fe, fe_poly.get_poly_space_numbering_inverse(), p);
  }


  /**
   * Collect two coordinate components from the list of points in 3D space.
   *
   * \mynote{This function is useful in constructing the surface metric tensor
   * or surface normal vector.}
   *
   * @param points A list of points in 3D space
   * @param first_component Index for the first coordinate component to be
   * collected
   * @param second_component Index for the second coordinate component to be
   * collected
   * @return A matrix storing the two coordinate components for all points. It
   * has a dimension @p 2*points.size().
   */
  template <int spacedim, typename RangeNumberType = double>
  LAPACKFullMatrixExt<RangeNumberType>
  collect_two_components_from_point3(
    const std::vector<Point<spacedim, RangeNumberType>> &points,
    const unsigned int                                   first_component,
    const unsigned int                                   second_component)
  {
    Assert(first_component < spacedim, ExcInternalError());
    Assert(second_component < spacedim, ExcInternalError());

    LAPACKFullMatrixExt<RangeNumberType> two_component_coords(2, points.size());

    for (unsigned int i = 0; i < points.size(); i++)
      {
        two_component_coords(0, i) = points[i](first_component);
        two_component_coords(1, i) = points[i](second_component);
      }

    return two_component_coords;
  }


  /**
   * Collect two coordinate components from the list of points in 3D space.
   * Only the first @p effective_point_num will be used.
   *
   * \mynote{This function is useful in constructing the surface metric tensor
   * or surface normal vector.}
   *
   * @param points A list of points in 3D space
   * @param effective_point_num Number of points to be used.
   * @param first_component Index for the first coordinate component to be
   * collected
   * @param second_component Index for the second coordinate component to be
   * collected
   * @return A matrix storing the two coordinate components for all points. It
   * has a dimension @p 2*points.size().
   */
  template <int spacedim, typename RangeNumberType = double>
  LAPACKFullMatrixExt<RangeNumberType>
  collect_two_components_from_point3(
    const std::vector<Point<spacedim, RangeNumberType>> &points,
    const unsigned int                                   effective_point_num,
    const unsigned int                                   first_component,
    const unsigned int                                   second_component)
  {
    Assert(first_component < spacedim, ExcInternalError());
    Assert(second_component < spacedim, ExcInternalError());
    Assert(effective_point_num <= points.size(), ExcInternalError());

    LAPACKFullMatrixExt<RangeNumberType> two_component_coords(
      2, effective_point_num);

    for (unsigned int i = 0; i < effective_point_num; i++)
      {
        two_component_coords(0, i) = points[i](first_component);
        two_component_coords(1, i) = points[i](second_component);
      }

    return two_component_coords;
  }


  /**
   * Collect coordinate components from the list of points in
   * \f$\mathbb{R}^d\f$. The obtained coordinate matrix has this format:
   * \f[
   * \begin{pmatrix}
   * x_1(1) & \cdots & x_1(k) \\
   * \vdots & \vdots & \vdots \\
   * x_d(1) & \cdots & x_d(k)
   * \end{pmatrix}
   * \f]
   * where \f$k\f$ is the number of points and \f$x(i)\f$ is the i-th point in
   * the list.
   *
   * \mynote{This function will be used for calculating the Jacobian matrix.
   * Let \f$DN\f$ be the matrix of first order derivatives of shape functions
   * and \f$P\f$ be the resulted coordinate matrix. Then
   * \[
   * J = P \cdot DN
   * \]}
   *
   * @param points
   * @return
   */
  template <int spacedim, typename RangeNumberType = double>
  LAPACKFullMatrixExt<RangeNumberType>
  collect_components_from_points(
    const std::vector<Point<spacedim, RangeNumberType>> &points)
  {
    LAPACKFullMatrixExt<RangeNumberType> point_coords(spacedim, points.size());

    for (unsigned int i = 0; i < points.size(); i++)
      {
        for (unsigned int j = 0; j < spacedim; j++)
          {
            point_coords(j, i) = points[i](j);
          }
      }

    return point_coords;
  }


  /**
   * This overloaded function only use the first @p point_num number of points.
   *
   * @pre
   * @post
   * @tparam spacedim
   * @param points
   * @param point_num
   * @return
   */
  template <int spacedim, typename RangeNumberType = double>
  LAPACKFullMatrixExt<RangeNumberType>
  collect_components_from_points(
    const std::vector<Point<spacedim, RangeNumberType>> &points,
    const unsigned int                                   point_num)
  {
    LAPACKFullMatrixExt<RangeNumberType> point_coords(spacedim, point_num);

    for (unsigned int i = 0; i < point_num; i++)
      {
        for (unsigned int j = 0; j < spacedim; j++)
          {
            point_coords(j, i) = points[i](j);
          }
      }

    return point_coords;
  }


  /**
   * Get the list of unit support points in the finite element, which is
   * ordered according to the specified permutation. The results are obtained
   * via the return value.
   *
   * \ingroup support_points_manip
   *
   * @param fe
   * @param dof_permutation
   * @return
   */
  template <int dim, int spacedim = dim>
  std::vector<Point<dim>>
  get_unit_support_points_with_permutation(
    const FiniteElement<dim, spacedim> &fe,
    const std::vector<unsigned int>    &dof_permutation)
  {
    Assert(fe.has_support_points(),
           ExcMessage("The finite element should have support points."));

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    AssertDimension(dofs_per_cell, dof_permutation.size());

    std::vector<Point<dim>> permuted_unit_support_points(dofs_per_cell);

    /**
     * Get the list of support points in the unit cell in the default DoF
     * ordering.
     */
    const std::vector<Point<dim>> &unit_support_points =
      fe.get_unit_support_points();

    for (unsigned int i = 0; i < dofs_per_cell; i++)
      {
        permuted_unit_support_points[i] =
          unit_support_points[dof_permutation[i]];
      }

    return permuted_unit_support_points;
  }


  /**
   * Get the list of unit support points in the finite element, which is
   * ordered according to the specified permutation. The results are obtained
   * as argument by reference.
   *
   * \ingroup support_points_manip
   *
   * @param fe
   * @param dof_permutation
   * @param permuted_unit_support_points Returned list of permuted unit
   * support points, the memory of which should be preallocated.
   */
  template <int dim, int spacedim = dim>
  void
  get_unit_support_points_with_permutation(
    const FiniteElement<dim, spacedim> &fe,
    const std::vector<unsigned int>    &dof_permutation,
    std::vector<Point<dim>>            &permuted_unit_support_points)
  {
    Assert(fe.has_support_points(),
           ExcMessage("The finite element should have support points."));

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    AssertDimension(dofs_per_cell, dof_permutation.size());
    AssertDimension(dofs_per_cell, permuted_unit_support_points.size());

    /**
     * Get the list of support points in the unit cell in the default DoF
     * ordering.
     */
    const std::vector<Point<dim>> &unit_support_points =
      fe.get_unit_support_points();

    for (unsigned int i = 0; i < dofs_per_cell; i++)
      {
        permuted_unit_support_points[i] =
          unit_support_points[dof_permutation[i]];
      }
  }


  /**
   * Get the list of unit support points in the lexicographic order in the
   * finite element. The results are obtained via the return value.
   *
   * \ingroup support_points_manip
   *
   * @param fe
   * @return
   */
  template <int dim, int spacedim = dim>
  std::vector<Point<dim>>
  get_lexicographic_unit_support_points(const FiniteElement<dim, spacedim> &fe)
  {
    Assert(fe.has_support_points(),
           ExcMessage("The finite element should have support points."));

    const unsigned int      dofs_per_cell = fe.dofs_per_cell;
    std::vector<Point<dim>> lexicographic_unit_support_points(dofs_per_cell);

    using FE_Poly_short          = FE_Poly<dim, spacedim>;
    const FE_Poly_short &fe_poly = dynamic_cast<const FE_Poly_short &>(fe);

    lexicographic_unit_support_points =
      get_unit_support_points_with_permutation(
        fe, fe_poly.get_poly_space_numbering_inverse());

    return lexicographic_unit_support_points;
  }


  /**
   * Get the list of unit support points in the lexicographic order in the
   * finite element. The results are obtained via argument by reference.
   *
   * \ingroup support_points_manip
   *
   * @param fe
   * @param lexicographic_unit_support_points Returned list of unit support
   * points in the lexicographic order, the memory of which should be
   * preallocated.
   */
  template <int dim, int spacedim = dim>
  void
  get_lexicographic_unit_support_points(
    const FiniteElement<dim, spacedim> &fe,
    std::vector<Point<dim>>            &lexicographic_unit_support_points)
  {
    Assert(fe.has_support_points(),
           ExcMessage("The finite element should have support points."));

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    AssertDimension(dofs_per_cell, lexicographic_unit_support_points.size());

    using FE_Poly_short          = FE_Poly<dim, spacedim>;
    const FE_Poly_short &fe_poly = dynamic_cast<const FE_Poly_short &>(fe);

    get_unit_support_points_with_permutation(
      fe,
      fe_poly.get_poly_space_numbering_inverse(),
      lexicographic_unit_support_points);
  }


  /**
   * Calculate a list of support points in the real cell in the order
   * specified by @p dof_permutation. The results are returned via the return
   * value.
   *
   * \ingroup support_points_manip
   *
   * @param cell
   * @param fe
   * @param mapping Geometric mapping object used for transforming support
   * points from the unit cell to the real cell.
   * @param dof_permutation The numbering for accessing the support points in
   * the specified order.
   * @return A list of support points in the real cell.
   *
   * \mynote{N.B. Each support point in the real cell has the space
   * dimension
   * @p spacedim, while each support point in the unit cell has the manifold
   * dimension @p dim.}
   */
  template <int dim, int spacedim = dim>
  std::vector<Point<spacedim>>
  get_support_points_in_real_cell(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const FiniteElement<dim, spacedim>                         &fe,
    const MappingQ<dim, spacedim>                              &mapping,
    const std::vector<unsigned int>                            &dof_permutation)
  {
    Assert(fe.has_support_points(),
           ExcMessage("The finite element should have support points."));

    const unsigned int           dofs_per_cell = fe.dofs_per_cell;
    std::vector<Point<spacedim>> support_points_in_real_cell(dofs_per_cell);

    /**
     * Get the list of support points in the unit cell in the default DoF
     * ordering.
     */
    const std::vector<Point<dim>> &unit_support_points =
      fe.get_unit_support_points();

    /**
     * Transform the support points from unit cell to real cell via the
     * @p mapping object. The support points in the original default DoF
     * ordering are permuted according to @p dof_permutation.
     */
    for (unsigned int i = 0; i < dofs_per_cell; i++)
      {
        support_points_in_real_cell[i] = mapping.transform_unit_to_real_cell(
          cell, unit_support_points.at(dof_permutation[i]));
      }

    return support_points_in_real_cell;
  }


  /**
   * Calculate a list of support points in the real cell in the order
   * specified by @p dof_permutation. The results are returned via argument
   * by reference.
   *
   * \ingroup support_points_manip
   *
   * @param cell
   * @param fe
   * @param mapping
   * @param dof_permutation
   * @param support_points_in_real_cell
   */
  template <int dim, int spacedim = dim>
  void
  get_support_points_in_real_cell(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const FiniteElement<dim, spacedim>                         &fe,
    const MappingQ<dim, spacedim>                              &mapping,
    const std::vector<unsigned int>                            &dof_permutation,
    std::vector<Point<spacedim>> &support_points_in_real_cell)
  {
    Assert(fe.has_support_points(),
           ExcMessage("The finite element should have support points."));

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    AssertDimension(dofs_per_cell, support_points_in_real_cell.size());

    /**
     * Get the list of support points in the unit cell in the default DoF
     * ordering.
     */
    const std::vector<Point<dim>> &unit_support_points =
      fe.get_unit_support_points();

    /**
     * Transform the support points from unit cell to real cell via the
     * @p mapping object. The support points in the original default DoF
     * ordering are permuted according to @p dof_permutation.
     */
    for (unsigned int i = 0; i < dofs_per_cell; i++)
      {
        support_points_in_real_cell[i] = mapping.transform_unit_to_real_cell(
          cell, unit_support_points.at(dof_permutation[i]));
      }
  }


  /**
   * Calculate a list of support points in the real cell in the default
   * DoF order.
   *
   * \ingroup support_points_manip
   *
   * @param cell
   * @param fe
   * @param mapping Geometric mapping object used for transforming support
   * points from the unit cell to the real cell.
   * @return A list of support points in the real cell in the default DoF
   * order.
   *
   * \mynote{N.B. Each support point in the real cell has the space dimension
   * @p spacedim, while each support point in the unit cell has the manifold
   * dimension @p dim.}
   */
  template <int dim, int spacedim>
  std::vector<Point<spacedim>>
  get_support_points_in_default_dof_order_in_real_cell(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const FiniteElement<dim, spacedim>                         &fe,
    const MappingQ<dim, spacedim>                              &mapping)
  {
    Assert(fe.has_support_points(),
           ExcMessage("The finite element should have support points."));

    const unsigned int           dofs_per_cell = fe.dofs_per_cell;
    std::vector<Point<spacedim>> support_points_in_real_cell(dofs_per_cell);

    // Get the list of support points in the unit cell in the default
    // hierarchical ordering.
    const std::vector<Point<dim>> &unit_support_points =
      fe.get_unit_support_points();

    // Transform the support points from unit cell to real cell.
    for (unsigned int i = 0; i < dofs_per_cell; i++)
      {
        support_points_in_real_cell[i] =
          mapping.transform_unit_to_real_cell(cell, unit_support_points[i]);
      }

    return support_points_in_real_cell;
  }


  /**
   * Calculate a list of support points in the real cell in the default
   * DoF order.
   *
   * \ingroup support_points_manip
   *
   * @param cell
   * @param fe
   * @param mapping Geometric mapping object used for transforming support
   * points from the unit cell to the real cell.
   * @param support_points_in_real_cell A list of support points in the real
   * cell in the hierarchic order, the memory of which should be preallocated.
   *
   * \mynote{N.B. Each support point in the real cell has the space dimension
   * @p spacedim, while each support point in the unit cell has the manifold
   * dimension @p dim.}
   */
  template <int dim, int spacedim>
  void
  get_support_points_in_default_dof_order_in_real_cell(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const FiniteElement<dim, spacedim>                         &fe,
    const MappingQ<dim, spacedim>                              &mapping,
    std::vector<Point<spacedim>> &support_points_in_reall_cell)
  {
    const unsigned int dofs_per_cell = fe.dofs_per_cell;

    Assert(support_points_in_reall_cell.size() == dofs_per_cell,
           ExcDimensionMismatch(support_points_in_reall_cell.size(),
                                dofs_per_cell));

    Assert(fe.has_support_points(),
           ExcMessage("The finite element should have support points."));

    // Get the list of support points in the unit cell in the default
    // hierarchical ordering.
    const std::vector<Point<dim>> &unit_support_points =
      fe.get_unit_support_points();

    // Transform the support points from unit cell to real cell.
    for (unsigned int i = 0; i < dofs_per_cell; i++)
      {
        support_points_in_reall_cell[i] =
          mapping.transform_unit_to_real_cell(cell, unit_support_points[i]);
      }
  }


  /**
   * Calculate a list of support points in the real cell in the lexicographic
   * order. The results are obtained via the return value.
   *
   * \ingroup support_points_manip
   *
   * @param cell
   * @param fe
   * @param mapping Geometric mapping object used for transforming support
   * points from the unit cell to the real cell.
   * @return A list of support points in the real cell in the lexicographic
   * order.
   *
   * \mynote{N.B. Each support point in the real cell has the space dimension
   * @p spacedim, while each support point in the unit cell has the manifold
   * dimension @p dim.}
   */
  template <int dim, int spacedim>
  std::vector<Point<spacedim>>
  get_lexicographic_support_points_in_real_cell(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const FiniteElement<dim, spacedim>                         &fe,
    const MappingQ<dim, spacedim>                              &mapping)
  {
    Assert(fe.has_support_points(),
           ExcMessage("The finite element should have support points."));

    const unsigned int           dofs_per_cell = fe.dofs_per_cell;
    std::vector<Point<spacedim>> support_points_in_real_cell(dofs_per_cell);

    using FE_Poly_short          = FE_Poly<dim, spacedim>;
    const FE_Poly_short &fe_poly = dynamic_cast<const FE_Poly_short &>(fe);

    support_points_in_real_cell = get_support_points_in_real_cell(
      cell, fe, mapping, fe_poly.get_poly_space_numbering_inverse());

    return support_points_in_real_cell;
  }


  /**
   * Calculate a list of support points in the real cell in the lexicographic
   * order. The results are obtained via argument by reference.
   *
   * \ingroup support_points_manip
   *
   * @param cell
   * @param fe
   * @param mapping
   * @param support_points_in_real_cell Returned list of support points in the
   * lexicographic order in the real cell, the memory of which should be
   * preallocated.
   */
  template <int dim, int spacedim>
  void
  get_lexicographic_support_points_in_real_cell(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const FiniteElement<dim, spacedim>                         &fe,
    const MappingQ<dim, spacedim>                              &mapping,
    std::vector<Point<spacedim>> &support_points_in_real_cell)
  {
    Assert(fe.has_support_points(),
           ExcMessage("The finite element should have support points."));

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    AssertDimension(dofs_per_cell, support_points_in_real_cell.size());

    using FE_Poly_short          = FE_Poly<dim, spacedim>;
    const FE_Poly_short &fe_poly = dynamic_cast<const FE_Poly_short &>(fe);

    get_support_points_in_real_cell(cell,
                                    fe,
                                    mapping,
                                    fe_poly.get_poly_space_numbering_inverse(),
                                    support_points_in_real_cell);
  }


  template <int dim, int spacedim = dim>
  unsigned int
  get_dofs_per_face_for_fe(const FiniteElement<dim, spacedim> &fe)
  {
    if (fe.conforms(FiniteElementData<dim>::H1))
      {
        Assert(
          fe.dofs_per_face > 0,
          ExcMessage(
            "H1 finite element should have non-zero number of DoFs per face!"));

        return fe.dofs_per_face;
      }
    else if (fe.conforms(FiniteElementData<dim>::L2))
      {
        Assert(fe.dofs_per_face == 0,
               ExcMessage("L2 finite element should have zero DoFs per face!"));

        switch (dim)
          {
              case 1: {
                return 1;
              }
              case 2: {
                return fe.degree + 1;
              }
              case 3: {
                return (fe.degree + 1) * (fe.degree + 1);
              }
              default: {
                Assert(false, ExcNotImplemented());

                return 0;
              }
          }
      }
    else
      {
        Assert(false, ExcNotImplemented());
        return 0;
      }
  }


  /**
   * Get the list of vertex coordinates from a list of unit support points in
   * the lexicographic order. The results are obtained via the return value.
   *
   * If @p is_counter_clockwise_ordered is @p true when @p dim==2, the last
   * two vertices will be swapped.
   *
   * \ingroup support_points_manip
   *
   * @param fe
   * @return
   */
  template <int dim, int spacedim>
  std::array<Point<dim>, GeometryInfo<dim>::vertices_per_cell>
  get_vertices_from_lexicographic_unit_support_points(
    const FiniteElement<dim, spacedim> &fe,
    const bool                          is_counter_clockwise_ordered = false)
  {
    /**
     * Get the list of unit support points in the lexicographic order.
     */
    std::vector<Point<dim>> unit_support_points(
      get_lexicographic_unit_support_points(fe));

    std::array<Point<dim>, GeometryInfo<dim>::vertices_per_cell> vertices;

    const unsigned int dofs_per_face = get_dofs_per_face_for_fe(fe);

    switch (dim)
      {
          case 1: {
            vertices[0] = unit_support_points[0];
            vertices[1] = unit_support_points[unit_support_points.size() - 1];

            break;
          }
          case 2: {
            vertices[0] = unit_support_points[0];
            vertices[1] = unit_support_points[dofs_per_face - 1];

            if (is_counter_clockwise_ordered)
              {
                /**
                 * Swap the last two vertices so that all the vertices are
                 * ordered counter clockwise.
                 */
                vertices[2] =
                  unit_support_points[unit_support_points.size() - 1];
                vertices[3] = unit_support_points[unit_support_points.size() -
                                                  1 - (dofs_per_face - 1)];
              }
            else
              {
                vertices[2] = unit_support_points[unit_support_points.size() -
                                                  1 - (dofs_per_face - 1)];
                vertices[3] =
                  unit_support_points[unit_support_points.size() - 1];
              }

            break;
          }
          default: {
            Assert(false, ExcNotImplemented());

            break;
          }
      }

    return vertices;
  }


  /**
   * Get the list of vertex coordinates from a list of unit support points in
   * the lexicographic order. The results are obtained via argument by
   * reference.
   *
   * If @p is_counter_clockwise_ordered is @p true when @p dim==2, the last
   * two vertices will be swapped.
   *
   * \ingroup support_points_manip
   *
   * @param fe
   * @param vertices
   * @param is_counter_clockwise_ordered
   */
  template <int dim, int spacedim>
  void
  get_vertices_from_lexicographic_unit_support_points(
    const FiniteElement<dim, spacedim>                           &fe,
    std::array<Point<dim>, GeometryInfo<dim>::vertices_per_cell> &vertices,
    const bool is_counter_clockwise_ordered = false)
  {
    /**
     * Get the list of unit support points in the lexicographic order.
     */
    std::vector<Point<dim>> unit_support_points(
      get_lexicographic_unit_support_points(fe));

    const unsigned int dofs_per_face = get_dofs_per_face_for_fe(fe);

    switch (dim)
      {
          case 1: {
            vertices[0] = unit_support_points[0];
            vertices[1] = unit_support_points[unit_support_points.size() - 1];

            break;
          }
          case 2: {
            vertices[0] = unit_support_points[0];
            vertices[1] = unit_support_points[dofs_per_face - 1];

            if (is_counter_clockwise_ordered)
              {
                vertices[2] =
                  unit_support_points[unit_support_points.size() - 1];
                vertices[3] = unit_support_points[unit_support_points.size() -
                                                  1 - (dofs_per_face - 1)];
              }
            else
              {
                vertices[2] = unit_support_points[unit_support_points.size() -
                                                  1 - (dofs_per_face - 1)];
                vertices[3] =
                  unit_support_points[unit_support_points.size() - 1];
              }

            break;
          }
          default: {
            Assert(false, ExcNotImplemented());

            break;
          }
      }
  }


  /**
   * Get the list of vertex coordinates from a list of support points in the
   * real cell in the lexicographic order. The results are obtained via the
   * return value.
   *
   * If @p is_counter_clockwise_ordered is @p true when @p dim==2, the last
   * two vertices will be swapped.
   *
   * \ingroup support_points_manip
   *
   * @param cell
   * @param fe
   * @param mapping
   * @param is_counter_clockwise_ordered
   * @return
   */
  template <int dim, int spacedim>
  std::array<Point<spacedim>, GeometryInfo<dim>::vertices_per_cell>
  get_vertices_from_lexicographic_support_points_in_real_cell(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const FiniteElement<dim, spacedim>                         &fe,
    const Mapping<dim, spacedim>                               &mapping,
    const bool is_counter_clockwise_ordered = false)
  {
    Assert(fe.has_support_points(),
           ExcMessage("The finite element should have support points."));

    /**
     * Extract the list of support points at vertices in the unit cell.
     */
    const unsigned int vertices_per_cell = GeometryInfo<dim>::vertices_per_cell;
    std::array<Point<dim>, vertices_per_cell> vertices_in_unit_cell;
    get_vertices_from_lexicographic_unit_support_points(
      fe, vertices_in_unit_cell, is_counter_clockwise_ordered);

    /**
     * Map the support points from the unit cell to the real cell.
     */
    std::array<Point<spacedim>, vertices_per_cell> vertices_in_real_cell;
    for (unsigned int v = 0; v < vertices_per_cell; v++)
      {
        vertices_in_real_cell[v] =
          mapping.transform_unit_to_real_cell(cell, vertices_in_unit_cell[v]);
      }

    return vertices_in_real_cell;
  }


  /**
   * Get the list of vertex coordinates from a list of support points in the
   * real cell in the lexicographic order. The results are obtained via
   * argument by reference.
   *
   * If @p is_counter_clockwise_ordered is @p true when @p dim==2, the last
   * two vertices will be swapped.
   *
   * \ingroup support_points_manip
   *
   * @param cell
   * @param fe
   * @param mapping
   * @param vertices_in_real_cell
   * @param is_counter_clockwise_ordered
   */
  template <int dim, int spacedim>
  void
  get_vertices_from_lexicographic_support_points_in_real_cell(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const FiniteElement<dim, spacedim>                         &fe,
    const Mapping<dim, spacedim>                               &mapping,
    std::array<Point<spacedim>, GeometryInfo<dim>::vertices_per_cell>
              &vertices_in_real_cell,
    const bool is_counter_clockwise_ordered = false)
  {
    const unsigned int vertices_per_cell = GeometryInfo<dim>::vertices_per_cell;

    Assert(fe.has_support_points(),
           ExcMessage("The finite element should have support points."));
    AssertDimension(vertices_in_real_cell.size(), vertices_per_cell);

    /**
     * Extract the list of support points at vertices in the unit cell.
     */
    std::array<Point<dim>, vertices_per_cell> vertices_in_unit_cell;
    get_vertices_from_lexicographic_unit_support_points(
      fe, vertices_in_unit_cell, is_counter_clockwise_ordered);

    /**
     * Map the support points from the unit cell to the real cell.
     */
    for (unsigned int v = 0; v < vertices_per_cell; v++)
      {
        vertices_in_real_cell[v] =
          mapping.transform_unit_to_real_cell(cell, vertices_in_unit_cell[v]);
      }
  }


  /**
   * Get the list of DoF indices in the current cell. It is obtained via the
   * return value.
   *
   * @param cell
   * @return
   */
  template <int dim, int spacedim>
  std::vector<types::global_dof_index>
  get_lexicographic_dof_indices(
    const typename DoFHandler<dim, spacedim>::cell_iterator &cell)
  {
    const FiniteElement<dim, spacedim> &fe = cell->get_fe();
    Assert(fe.has_support_points(),
           ExcMessage("The finite element should have support points."));

    /**
     * Extract the list of DoF indices in the current cell.
     */
    std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
    std::vector<types::global_dof_index> lexicographic_dof_indices(
      fe.dofs_per_cell);

    cell->get_dof_indices(dof_indices);

    using FE_Poly_short          = FE_Poly<dim, spacedim>;
    const FE_Poly_short &fe_poly = dynamic_cast<const FE_Poly_short &>(fe);

    permute_vector(dof_indices,
                   fe_poly.get_poly_space_numbering_inverse(),
                   lexicographic_dof_indices);

    return lexicographic_dof_indices;
  }


  /**
   * Get the list of DoF indices in the current cell. It is obtained via
   * argument by reference.
   *
   * @param cell
   * @param lexicographic_dof_indices
   */
  template <int dim, int spacedim>
  void
  get_lexicographic_dof_indices(
    const typename DoFHandler<dim, spacedim>::cell_iterator &cell,
    std::vector<types::global_dof_index> &lexicographic_dof_indices)
  {
    const FiniteElement<dim, spacedim> &fe = cell->get_fe();
    Assert(fe.has_support_points(),
           ExcMessage("The finite element should have support points."));

    AssertDimension(lexicographic_dof_indices.size(), fe.dofs_per_cell);

    /**
     * Extract the list of DoF indices in the current cell.
     */
    std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
    cell->get_dof_indices(dof_indices);

    using FE_Poly_short          = FE_Poly<dim, spacedim>;
    const FE_Poly_short &fe_poly = dynamic_cast<const FE_Poly_short &>(fe);

    permute_vector(dof_indices,
                   fe_poly.get_poly_space_numbering_inverse(),
                   lexicographic_dof_indices);
  }


  /**
   * Get a list of vertex DoF indices, which are directly extracted from the
   * list of DoF indices in the lexicographic order. The result is obtained in
   * the return value.
   *
   * @param cell
   * @param is_counter_clockwise_ordered
   * @return
   */
  template <int dim, int spacedim>
  std::vector<types::global_dof_index>
  get_vertex_dof_indices_from_lexicographic_dof_indices(
    const typename DoFHandler<dim, spacedim>::cell_iterator &cell,
    const bool is_counter_clockwise_ordered = false)
  {
    const FiniteElement<dim, spacedim> &fe = cell->get_fe();

    /**
     * Get the list of DoF indices in the lexicographic order.
     */
    std::vector<types::global_dof_index> lexicographic_dof_indices(
      fe.dofs_per_cell);
    get_lexicographic_dof_indices(cell, lexicographic_dof_indices);

    std::vector<types::global_dof_index> vertex_dof_indices(
      GeometryInfo<dim>::vertices_per_cell);

    const unsigned int dofs_per_face = get_dofs_per_face_for_fe(fe);

    switch (dim)
      {
          case 1: {
            vertex_dof_indices[0] = lexicographic_dof_indices[0];
            vertex_dof_indices[1] =
              lexicographic_dof_indices[lexicographic_dof_indices.size() - 1];

            break;
          }
          case 2: {
            vertex_dof_indices[0] = lexicographic_dof_indices[0];
            vertex_dof_indices[1] =
              lexicographic_dof_indices[dofs_per_face - 1];

            if (is_counter_clockwise_ordered)
              {
                vertex_dof_indices[2] =
                  lexicographic_dof_indices[lexicographic_dof_indices.size() -
                                            1];
                vertex_dof_indices[3] =
                  lexicographic_dof_indices[lexicographic_dof_indices.size() -
                                            1 - (dofs_per_face - 1)];
              }
            else
              {
                vertex_dof_indices[2] =
                  lexicographic_dof_indices[lexicographic_dof_indices.size() -
                                            1 - (dofs_per_face - 1)];
                vertex_dof_indices[3] =
                  lexicographic_dof_indices[lexicographic_dof_indices.size() -
                                            1];
              }

            break;
          }
          default: {
            Assert(false, ExcNotImplemented());

            break;
          }
      }

    return vertex_dof_indices;
  }


  /**
   * Get a list of vertex DoF indices, which are directly extracted from the
   * list of DoF indices in the lexicographic order. The result is obtained
   * via argument by reference.
   *
   * @param cell
   * @param vertex_dof_indices
   * @param is_counter_clockwise_ordered
   */
  template <int dim, int spacedim = dim>
  void
  get_vertex_dof_indices_from_lexicographic_dof_indices(
    const typename DoFHandler<dim, spacedim>::cell_iterator &cell,
    std::array<types::global_dof_index, GeometryInfo<dim>::vertices_per_cell>
              &vertex_dof_indices,
    const bool is_counter_clockwise_ordered = false)
  {
    const FiniteElement<dim, spacedim> &fe = cell->get_fe();

    /**
     * Get the list of DoF indices in the lexicographic order.
     */
    std::vector<types::global_dof_index> lexicographic_dof_indices(
      fe.dofs_per_cell);
    get_lexicographic_dof_indices<dim, spacedim>(cell,
                                                 lexicographic_dof_indices);

    const unsigned int dofs_per_face = get_dofs_per_face_for_fe(fe);

    switch (dim)
      {
          case 1: {
            vertex_dof_indices[0] = lexicographic_dof_indices[0];
            vertex_dof_indices[1] =
              lexicographic_dof_indices[lexicographic_dof_indices.size() - 1];

            break;
          }
          case 2: {
            vertex_dof_indices[0] = lexicographic_dof_indices[0];
            vertex_dof_indices[1] =
              lexicographic_dof_indices[dofs_per_face - 1];

            if (is_counter_clockwise_ordered)
              {
                vertex_dof_indices[2] =
                  lexicographic_dof_indices[lexicographic_dof_indices.size() -
                                            1];
                vertex_dof_indices[3] =
                  lexicographic_dof_indices[lexicographic_dof_indices.size() -
                                            1 - (dofs_per_face - 1)];
              }
            else
              {
                vertex_dof_indices[2] =
                  lexicographic_dof_indices[lexicographic_dof_indices.size() -
                                            1 - (dofs_per_face - 1)];
                vertex_dof_indices[3] =
                  lexicographic_dof_indices[lexicographic_dof_indices.size() -
                                            1];
              }

            break;
          }
          default: {
            Assert(false, ExcNotImplemented());

            break;
          }
      }
  }


  /**
   * Return a list of global DoF indices, which are located at the list of
   * vertices in the cell respectively. The result is obtained via the return
   * value.
   *
   * @param cell
   * @param mapping
   * @param threshold
   * @return
   */
  template <int dim, int spacedim>
  std::array<types::global_dof_index, GeometryInfo<dim>::vertices_per_cell>
  get_vertex_dof_indices_in_cell(
    const typename DoFHandler<dim, spacedim>::cell_iterator &cell,
    const Mapping<dim, spacedim>                            &mapping,
    const double                                             threshold = 1e-12)
  {
    const unsigned int vertices_per_cell = GeometryInfo<dim>::vertices_per_cell;
    const FiniteElement<dim, spacedim> &fe = cell->get_fe();

    Assert(fe.has_support_points(), ExcInternalError());

    std::array<types::global_dof_index, vertices_per_cell>
      cell_vertex_dof_indices;

    if (fe.conforms(FiniteElementData<dim>::H1))
      {
        /**
         * When the finite element conforms to \f$H_1\f$, e.g. @p FE_Q, the
         * vertex DoF indices can be directly obtained by calling the member
         * function @p DoFAccessor::vertex_dof_index.
         */

        /**
         * Assert there is only one DoF associated with each vertex.
         */
        Assert(fe.dofs_per_vertex == 1,
               ExcMessage("There should be only one DoF associated a vertex!"));

        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; v++)
          {
            cell_vertex_dof_indices[v] = cell->vertex_dof_index(v, 0);
          }
      }
    else if (fe.conforms(FiniteElementData<dim>::L2))
      {
        /**
         * Handle the case when the finite element conforms to \f$L_2\f$,
         * e.g. @p FE_DGQ, where there are no DoFs associated with vertices.
         */

        /**
         * Assert there is no DoF associated with each vertex.
         */
        Assert(fe.dofs_per_vertex == 0,
               ExcMessage("There should be no DoFs associated with a vertex!"));

        /**
         * Get the list of vertex DoF indices which are directly extracted
         * from the list of all DoF indices. N.B. The ordering of this list of
         * vertex DoF indices may not match the ordering of the vertices in
         * the geometry information.
         */
        std::array<types::global_dof_index, vertices_per_cell>
          vertex_dof_indices_from_lexicographic_dof_indices;
        get_vertex_dof_indices_from_lexicographic_dof_indices<dim, spacedim>(
          cell, vertex_dof_indices_from_lexicographic_dof_indices, false);

        /**
         * Calculate the list of vertex support point coordinates in the real
         * cell with the help of the mapping object. It will be compared with
         * the vertex coordinates obtained from the cell geometry.
         *
         * N.B. The ordering of this list of vertex support points corresponds
         * with the ordering of the list of the above vertex DoF indices.
         */
        std::array<Point<spacedim>, vertices_per_cell>
          vertex_support_points_in_real_cell;
        get_vertices_from_lexicographic_support_points_in_real_cell(
          cell, fe, mapping, vertex_support_points_in_real_cell);

        /**
         * Iterate over each vertex support point and check that to which
         * vertex in the cell it is equal.
         */
        unsigned int vertex_support_point_local_index;
        for (unsigned int v = 0; v < vertices_per_cell; v++)
          {
            vertex_support_point_local_index =
              get_vertex_local_index_in_cell<dim, spacedim>(
                vertex_support_points_in_real_cell[v], cell, threshold);
            AssertIndexRange(vertex_support_point_local_index,
                             vertices_per_cell);

            cell_vertex_dof_indices[vertex_support_point_local_index] =
              vertex_dof_indices_from_lexicographic_dof_indices[v];
          }
      }
    else
      {
        Assert(false, ExcNotImplemented());
      }

    return cell_vertex_dof_indices;
  }


  /**
   * Return a list of global DoF indices, which are located at the list of
   * vertices in the cell respectively. The result is obtained via argument by
   * reference.
   *
   * @param cell
   * @param mapping
   * @param threshold
   * @param cell_vertex_dof_indices
   */
  template <int dim, int spacedim>
  void
  get_vertex_dof_indices_in_cell(
    const typename DoFHandler<dim, spacedim>::cell_iterator &cell,
    const Mapping<dim, spacedim>                            &mapping,
    std::array<types::global_dof_index, GeometryInfo<dim>::vertices_per_cell>
                &cell_vertex_dof_indices,
    const double threshold = 1e-12)
  {
    const unsigned int vertices_per_cell = GeometryInfo<dim>::vertices_per_cell;
    const FiniteElement<dim, spacedim> &fe = cell->get_fe();

    Assert(fe.has_support_points(), ExcInternalError());

    if (fe.conforms(FiniteElementData<dim>::H1))
      {
        /**
         * When the finite element conforms to \f$H_1\f$, e.g. @p FE_Q, the
         * vertex DoF indices can be directly obtained by calling the member
         * function @p DoFAccessor::vertex_dof_index.
         */

        /**
         * Assert there is only one DoF associated with each vertex.
         */
        Assert(fe.dofs_per_vertex == 1,
               ExcMessage("There should be only one DoF associated a vertex!"));

        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; v++)
          {
            cell_vertex_dof_indices[v] = cell->vertex_dof_index(v, 0);
          }
      }
    else if (fe.conforms(FiniteElementData<dim>::L2))
      {
        /**
         * Handle the case when the finite element conforms to \f$L_2\f$,
         * e.g. @p FE_DGQ, where there are no DoFs associated with vertices.
         */

        /**
         * Assert there is no DoF associated with each vertex.
         */
        Assert(fe.dofs_per_vertex == 0,
               ExcMessage("There should be no DoFs associated with a vertex!"));

        /**
         * Get the list of vertex DoF indices which are directly extracted
         * from the list of all DoF indices. N.B. The ordering of this list of
         * vertex DoF indices may not match the ordering of the vertices in
         * the geometry information.
         */
        std::array<types::global_dof_index, vertices_per_cell>
          vertex_dof_indices_from_lexicographic_dof_indices;
        get_vertex_dof_indices_from_lexicographic_dof_indices(
          cell, vertex_dof_indices_from_lexicographic_dof_indices, false);

        /**
         * Calculate the list of vertex support point coordinates in the real
         * cell with the help of the mapping object. It will be compared with
         * the vertex coordinates obtained from the cell geometry.
         *
         * N.B. The ordering of this list of vertex support points corresponds
         * with the ordering of the list of the above vertex DoF indices.
         */
        std::array<Point<spacedim>, vertices_per_cell>
          vertex_support_points_in_real_cell;
        get_vertices_from_lexicographic_support_points_in_real_cell(
          cell, fe, mapping, vertex_support_points_in_real_cell);

        /**
         * Iterate over each vertex support point and check that to which
         * vertex in the cell it is equal.
         */
        unsigned int vertex_support_point_local_index;
        for (unsigned int v = 0; v < vertices_per_cell; v++)
          {
            vertex_support_point_local_index =
              get_vertex_local_index_in_cell<dim, spacedim>(
                vertex_support_points_in_real_cell[v], cell, threshold);
            AssertIndexRange(vertex_support_point_local_index,
                             vertices_per_cell);

            cell_vertex_dof_indices[vertex_support_point_local_index] =
              vertex_dof_indices_from_lexicographic_dof_indices[v];
          }
      }
    else
      {
        Assert(false, ExcNotImplemented());
      }
  }


  /**
   * Get the DoF index for the specified vertex in the cell.
   *
   * \mynote{When the finite element associated with the cell conforms with
   * @p H1, this function is the same as @p DoFAccessor::vertex_dof_index.
   * When the finite element conforms with @p L2, because there are no DoFs
   * associated with vertices, the matching of support points and vertices
   * should be calculated.}
   *
   * @param cell
   * @param mapping
   * @param local_vertex_index_in_cell
   * @param threshold Threshold for point equality checking
   * @return
   */
  template <int dim, int spacedim>
  typename types::global_dof_index
  get_dof_index_for_vertex_in_cell(
    const typename DoFHandler<dim, spacedim>::cell_iterator &cell,
    const Mapping<dim, spacedim>                            &mapping,
    const unsigned int local_vertex_index_in_cell,
    const double       threshold = 1e-12)
  {
    const unsigned int vertices_per_cell = GeometryInfo<dim>::vertices_per_cell;
    const FiniteElement<dim, spacedim> &fe = cell->get_fe();

    Assert(fe.has_support_points(), ExcInternalError());

    if (fe.conforms(FiniteElementData<dim>::H1))
      {
        /**
         * When the finite element conforms to \f$H_1\f$, e.g. @p FE_Q, the
         * vertex DoF indices can be directly obtained by calling the member
         * function @p DoFAccessor::vertex_dof_index.
         */

        /**
         * Assert there is only one DoF associated with each vertex.
         */
        Assert(fe.dofs_per_vertex == 1,
               ExcMessage("There should be only one DoF associated a vertex!"));


        return cell->vertex_dof_index(local_vertex_index_in_cell, 0);
      }
    else if (fe.conforms(FiniteElementData<dim>::L2))
      {
        /**
         * Handle the case when the finite element conforms to \f$L_2\f$,
         * e.g. @p FE_DGQ, where no DoFs are associated with vertices.
         */

        /**
         * Assert there is no DoF associated with each vertex.
         */
        Assert(fe.dofs_per_vertex == 0,
               ExcMessage("There should be no DoFs associated with a vertex!"));

        /**
         * Get the list of vertex DoF indices in lexicographic order, which
         * are directly extracted from the list of all DoF indices. N.B. The
         * ordering of this list of vertex DoF indices may not match the
         * ordering of the vertices in the geometry information.
         */
        std::array<types::global_dof_index, vertices_per_cell>
          vertex_dof_indices_from_lexicographic_dof_indices;
        get_vertex_dof_indices_from_lexicographic_dof_indices<dim, spacedim>(
          cell, vertex_dof_indices_from_lexicographic_dof_indices, false);

        /**
         * Calculate the list of vertex support point coordinates in the real
         * cell with the help of the mapping object. It will be compared with
         * the vertex coordinates obtained from the cell geometry using the
         * given @p threshold.
         *
         * N.B. The ordering of this list of vertex support points corresponds
         * with the ordering of the list of the above vertex DoF indices.
         */
        std::array<Point<spacedim>, vertices_per_cell>
          vertex_support_points_in_real_cell;
        get_vertices_from_lexicographic_support_points_in_real_cell(
          cell, fe, mapping, vertex_support_points_in_real_cell);

        /**
         * Iterate over each vertex support point and check if it matches the
         * required vertex.
         */
        for (unsigned int v = 0; v < vertices_per_cell; v++)
          {
            if (get_vertex_local_index_in_cell<dim, spacedim>(
                  vertex_support_points_in_real_cell[v], cell, threshold) ==
                local_vertex_index_in_cell)
              {
                return vertex_dof_indices_from_lexicographic_dof_indices[v];
              }
          }

        Assert(false,
               ExcMessage(
                 "There is no support point matching the specified vertex!"));
      }
    else
      {
        Assert(false, ExcNotImplemented());
      }

    return 0;
  }


  /**
   * Detect the cell neighboring type by checking the
   * intersection of the given two lists of vertex indices. The intersection
   * of the global vertex indices is returned in the argument
   * @p common_vertex_indices by reference.
   *
   * \alert{Comparison of vertex indices implies that the two cells should
   * belong to a same triangulation. Otherwise, there will be two sets of
   * vertex indices, which cannot be compared.}
   *
   * \mynote{N.B. The template argument @p dim of this function cannot be
   * automatically inducted by the compiler from its input arguments, since
   * even though the template argument @p GeometryInfo<dim>::vertices_per_cell
   * of the two input arguments @p first_cell_vertex_indices and
   * @p second_cell_vertex_indices contains @p dim, the whole
   * @p GeometryInfo<dim>::vertices_per_cell will be evaluated into an
   * integer by the compiler, so that @p dim will be discarded.}
   *
   * @param first_cell_vertex_indices
   * @param second_cell_vertex_indices
   * @param common_vertex_indices
   * @return
   */
  template <int dim>
  CellNeighboringType
  detect_cell_neighboring_type_for_same_triangulations(
    const std::array<types::global_vertex_index,
                     GeometryInfo<dim>::vertices_per_cell>
      &first_cell_vertex_indices,
    const std::array<types::global_vertex_index,
                     GeometryInfo<dim>::vertices_per_cell>
                                            &second_cell_vertex_indices,
    std::vector<types::global_vertex_index> &common_vertex_indices)
  {
    // The arrays storing vertex indices should be sorted before calling
    // @p std::set_intersection.
    auto first_cell_vertex_indices_sorted  = first_cell_vertex_indices;
    auto second_cell_vertex_indices_sorted = second_cell_vertex_indices;

    std::sort(first_cell_vertex_indices_sorted.begin(),
              first_cell_vertex_indices_sorted.end());
    std::sort(second_cell_vertex_indices_sorted.begin(),
              second_cell_vertex_indices_sorted.end());

    /**
     * Calculate the intersection of the two cells' vertex indices.
     */
    std::set_intersection(first_cell_vertex_indices_sorted.begin(),
                          first_cell_vertex_indices_sorted.end(),
                          second_cell_vertex_indices_sorted.begin(),
                          second_cell_vertex_indices_sorted.end(),
                          std::back_inserter(common_vertex_indices));

    CellNeighboringType cell_neighboring_type;
    switch (common_vertex_indices.size())
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
          break;
      }

    return cell_neighboring_type;
  }


  /**
   * Detect the cell neighboring type for two cells by checking the
   * intersection of their vertex indices. Each of the common global
   * vertex index is duplicated and made into a pair, then pushed into the
   * result vector.
   *
   * \alert{Comparison of vertex indices implies that the two cells should
   * belong to a same triangulation. Otherwise, there will be two sets of
   * vertex indices, which cannot be compared.}
   *
   * @param first_cell_iter
   * @param second_cell_iter
   * @param common_vertex_pair_indices
   * @return
   */
  template <int dim, int spacedim>
  CellNeighboringType
  detect_cell_neighboring_type_for_same_triangulations(
    const typename Triangulation<dim, spacedim>::active_cell_iterator
      &first_cell_iter,
    const typename Triangulation<dim, spacedim>::active_cell_iterator
      &second_cell_iter,
    std::vector<std::pair<unsigned int, unsigned int>>
      &common_vertex_pair_local_indices)
  {
    auto first_cell_vertex_indices =
      get_vertex_indices_in_cell<dim, spacedim>(first_cell_iter);
    auto second_cell_vertex_indices =
      get_vertex_indices_in_cell<dim, spacedim>(second_cell_iter);

    CellNeighboringType                     cell_neighboring_type;
    std::vector<types::global_vertex_index> common_vertex_indices;

    cell_neighboring_type =
      detect_cell_neighboring_type_for_same_triangulations<dim>(
        first_cell_vertex_indices,
        second_cell_vertex_indices,
        common_vertex_indices);

    common_vertex_pair_local_indices.clear();
    unsigned int common_vertex_local_index_in_first_cell;
    unsigned int common_vertex_local_index_in_second_cell;

    for (auto v : common_vertex_indices)
      {
        auto first_vertex_found_pos =
          std::find(first_cell_vertex_indices.cbegin(),
                    first_cell_vertex_indices.cend(),
                    v);
        Assert(first_vertex_found_pos != first_cell_vertex_indices.cend(),
               ExcMessage("Cannot find the common vertex in the first cell!"));

        common_vertex_local_index_in_first_cell =
          first_vertex_found_pos - first_cell_vertex_indices.cbegin();

        auto second_vertex_found_pos =
          std::find(second_cell_vertex_indices.cbegin(),
                    second_cell_vertex_indices.cend(),
                    v);
        Assert(second_vertex_found_pos != second_cell_vertex_indices.cend(),
               ExcMessage("Cannot find the common vertex in the second cell!"));

        common_vertex_local_index_in_second_cell =
          second_vertex_found_pos - second_cell_vertex_indices.cbegin();

        common_vertex_pair_local_indices.push_back(
          std::pair<unsigned int, unsigned int>(
            common_vertex_local_index_in_first_cell,
            common_vertex_local_index_in_second_cell));
      }

    return cell_neighboring_type;
  }


  /**
   * Detect the cell neighboring type for the two cells pointed by the input
   * cell iterators. This function handles the case when the involved two DoF
   * handlers are different where the finite elements are different but the
   * triangulations are the same.
   *
   * In this case, the vertex indices for the two cells are numbered in a same
   * index system. But the DoF indices for the two cells are independently
   * indexed.
   *
   * @param first_cell_iter
   * @param second_cell_iter
   * @param first_cell_mapping
   * @param second_cell_mapping
   * @param common_vertex_pair_dof_indices A vector of pairs of DoF indices.
   * Each pair corresponds to a common vertex. In each pair, the first DoF
   * index is in the first cell with the first DoF handler, while the second
   * DoF index is in the second cell with the second DoF handler.
   * @pram threshold
   * @return Cell neighboring type
   */
  template <int dim, int spacedim = dim>
  CellNeighboringType
  detect_cell_neighboring_type_for_same_triangulations(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator
      &first_cell_iter,
    const typename DoFHandler<dim, spacedim>::active_cell_iterator
                                 &second_cell_iter,
    const Mapping<dim, spacedim> &first_cell_mapping,
    const Mapping<dim, spacedim> &second_cell_mapping,
    std::vector<std::pair<types::global_dof_index, types::global_dof_index>>
                &common_vertex_pair_dof_indices,
    const double threshold = 1e-12)
  {
    const unsigned int vertices_per_cell = GeometryInfo<dim>::vertices_per_cell;

    const FiniteElement<dim, spacedim> &first_cell_fe =
      first_cell_iter->get_fe();
    const FiniteElement<dim, spacedim> &second_cell_fe =
      second_cell_iter->get_fe();

    /**
     * Get the vertex indices in each cell.
     */
    std::array<types::global_vertex_index, vertices_per_cell>
      first_cell_vertex_indices(
        get_vertex_indices_in_cell<dim, spacedim>(first_cell_iter));

    std::array<types::global_vertex_index, vertices_per_cell>
      second_cell_vertex_indices(
        get_vertex_indices_in_cell<dim, spacedim>(second_cell_iter));

    /**
     * Calculate the intersection of the two cells' vertex indices. This
     * operation is meaningful because the triangulations for the two cells
     * are the same.
     */
    std::vector<types::global_vertex_index> vertex_index_intersection;
    detect_cell_neighboring_type_for_same_triangulations<dim>(
      first_cell_vertex_indices,
      second_cell_vertex_indices,
      vertex_index_intersection);

    /**
     * Fill the vector of pairs for @p common_vertex_pair_dof_indices. For each
     * vertex index in the intersection, find the corresponding DoF indices in
     * the two cells respectively.
     */
    common_vertex_pair_dof_indices.clear();
    types::global_dof_index common_vertex_dof_index_in_first_cell;
    types::global_dof_index common_vertex_dof_index_in_second_cell;

    for (auto vertex_index : vertex_index_intersection)
      {
        if (first_cell_fe.dofs_per_cell > 1)
          {
            /**
             * Find the current common vertex in the list of vertex indices
             * for the first cell.
             */
            auto first_cell_vertex_found_pos =
              std::find(first_cell_vertex_indices.cbegin(),
                        first_cell_vertex_indices.cend(),
                        vertex_index);

            Assert(
              first_cell_vertex_found_pos != first_cell_vertex_indices.cend(),
              ExcMessage("Cannot find the common vertex in the first cell!"));

            common_vertex_dof_index_in_first_cell =
              get_dof_index_for_vertex_in_cell(
                first_cell_iter,
                first_cell_mapping,
                first_cell_vertex_found_pos -
                  first_cell_vertex_indices.cbegin(),
                threshold);
          }
        else
          {
            /**
             * Handle the case when there is only one DoF in a cell, i.e.
             * @p FE_DGQ with order 0. Set the common vertex DoF index to be
             * the DoF in the cell's interior, even though it is not
             * associated to the vertex.
             */
            common_vertex_dof_index_in_first_cell =
              first_cell_iter->dof_index(0);
          }

        if (second_cell_fe.dofs_per_cell > 1)
          {
            /**
             * Find the current common vertex in the list of vertex indices
             * for the second cell.
             */
            auto second_cell_vertex_found_pos =
              std::find(second_cell_vertex_indices.cbegin(),
                        second_cell_vertex_indices.cend(),
                        vertex_index);

            Assert(
              second_cell_vertex_found_pos != second_cell_vertex_indices.cend(),
              ExcMessage("Cannot find the common vertex in the second cell!"));

            common_vertex_dof_index_in_second_cell =
              get_dof_index_for_vertex_in_cell(
                second_cell_iter,
                second_cell_mapping,
                second_cell_vertex_found_pos -
                  second_cell_vertex_indices.cbegin(),
                threshold);
          }
        else
          {
            /**
             * Handle the case when the finite element order is 0, i.e. for
             * @p FE_DGQ. Set the common vertex DoF index to be the DoF in
             * the cell's interior, even though it is not associated to the
             * vertex.
             */
            common_vertex_dof_index_in_second_cell =
              second_cell_iter->dof_index(0);
          }

        common_vertex_pair_dof_indices.push_back(
          std::pair<types::global_dof_index, types::global_dof_index>(
            common_vertex_dof_index_in_first_cell,
            common_vertex_dof_index_in_second_cell));
      }

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
          break;
      }

    return cell_neighboring_type;
  }



  /**
   * Detect the cell neighboring type for two cells which belong to different
   * surface/boundary triangulations.
   *
   * These two surface triangulations are constructed respectively from a
   * common volume mesh by specifying a set of boundary ids. Therefore, the
   * map from surface/boundary mesh to volume mesh is needed.
   *
   * \alert{The map from surface/boundary mesh to volume mesh is only for the
   * coarse mesh, due to the behavior of @p GridGenerator::extract_boundary_mesh.}
   *
   * @param first_cell_iter
   * @param second_cell_iter
   * @param map_from_first_boundary_mesh_to_volume_mesh
   * @param map_from_second_boundary_mesh_to_volume_mesh
   * @param common_vertex_pair_indices A vector of pairs of vertex local
   * indices in the two cells in the surface mesh. Each pair corresponds to a
   * common vertex shared by the two cells, if any. The first element in each
   * pair is the local index of the common vertex in the first cell, and the
   * second element is that in the second cell.
   * @return Cell neighboring type
   */
  template <int dim, int spacedim>
  CellNeighboringType
  detect_cell_neighboring_type_for_different_triangulations(
    const typename Triangulation<dim, spacedim>::active_cell_iterator
      &first_cell_iter,
    const typename Triangulation<dim, spacedim>::active_cell_iterator
      &second_cell_iter,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_first_boundary_mesh_to_volume_mesh,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_second_boundary_mesh_to_volume_mesh,
    std::vector<std::pair<unsigned int, unsigned int>>
      &common_vertex_pair_local_indices)
  {
    /**
     * Get the iterators to the faces in the original volume triangulation.
     */
    auto first_face_iter_pos =
      map_from_first_boundary_mesh_to_volume_mesh.find(first_cell_iter);
    Assert(
      first_face_iter_pos != map_from_first_boundary_mesh_to_volume_mesh.end(),
      ExcMessage(
        "The associated face iterator in the volume triangulation cannot be found for the first cell iterator in the boundary mesh!"));
    auto first_face_iter = first_face_iter_pos->second;

    auto second_face_iter_pos =
      map_from_second_boundary_mesh_to_volume_mesh.find(second_cell_iter);
    Assert(
      second_face_iter_pos !=
        map_from_second_boundary_mesh_to_volume_mesh.end(),
      ExcMessage(
        "The associated face iterator in the volume triangulation cannot be found for the second cell iterator in the boundary mesh!"));
    auto second_face_iter = second_face_iter_pos->second;

    /**
     * Get the vertex indices in each face in the original volume
     * triangulation.
     */
    auto first_face_vertex_indices =
      get_vertex_indices_in_face<dim + 1, spacedim>(first_face_iter);
    auto second_face_vertex_indices =
      get_vertex_indices_in_face<dim + 1, spacedim>(second_face_iter);

    /**
     * Calculate the intersection of the two faces' vertex indices. This
     * operation is meaningful because the two faces are both pulled back into
     * the original volume triangulation.
     */
    CellNeighboringType cell_neighboring_type;
    std::vector<types::global_vertex_index>
      common_vertex_indices_in_volume_mesh;
    cell_neighboring_type =
      detect_cell_neighboring_type_for_same_triangulations<dim>(
        first_face_vertex_indices,
        second_face_vertex_indices,
        common_vertex_indices_in_volume_mesh);

    /**
     * According to deal.ii's documentation about @p extract_boundary_mesh,
     *
     * > The order of vertices of surface cells at the boundary and the
     * corresponding volume faces may not match in order to ensure that each
     * surface cell is associated with an outward facing normal. As a
     * consequence, if you want to match quantities on the faces of the
     * domain cells and on the cells of the surface mesh, you may have to
     * translate between vertex locations or quadrature points.
     *
     * Hence, the local index of each common vertex in its corresponding face
     * in the volume mesh is not the same as its local index in the cell in
     * the surface mesh. Therefore, the correlation mapping between cell
     * vertices in the surface mesh and those in the face in the volume mesh
     * should be constructed by matching vertex coordinates.
     */

    common_vertex_pair_local_indices.clear();
    unsigned int common_vertex_local_index_in_first_cell;
    unsigned int common_vertex_local_index_in_second_cell;

    for (auto v : common_vertex_indices_in_volume_mesh)
      {
        auto first_face_vertex_found_pos =
          std::find(first_face_vertex_indices.cbegin(),
                    first_face_vertex_indices.cend(),
                    v);
        Assert(first_face_vertex_found_pos != first_face_vertex_indices.cend(),
               ExcMessage("Cannot find the common vertex in the first face!"));

        /**
         * Get the coordinates of the current common vertex. \mynote{The
         * point returned from the member function @p TriaAccessor::vertex is
         * a reference.}
         */
        const Point<spacedim> &common_vertex_coords = first_face_iter->vertex(
          first_face_vertex_found_pos - first_face_vertex_indices.cbegin());

        /**
         * Find the current common vertex in the first cell in the surface
         * mesh by coordinate matching.
         */
        common_vertex_local_index_in_first_cell =
          get_vertex_local_index_in_cell<dim, spacedim>(common_vertex_coords,
                                                        first_cell_iter);
        AssertIndexRange(common_vertex_local_index_in_first_cell,
                         GeometryInfo<dim>::vertices_per_cell);

        /**
         * Find the current common vertex in the second cell in the surface
         * mesh by coordinate matching.
         */
        common_vertex_local_index_in_second_cell =
          get_vertex_local_index_in_cell<dim, spacedim>(common_vertex_coords,
                                                        second_cell_iter);
        AssertIndexRange(common_vertex_local_index_in_second_cell,
                         GeometryInfo<dim>::vertices_per_cell);

        common_vertex_pair_local_indices.push_back(
          std::pair<unsigned int, unsigned int>(
            common_vertex_local_index_in_first_cell,
            common_vertex_local_index_in_second_cell));
      }

    return cell_neighboring_type;
  }


  /**
   * Detect the cell neighboring type for the two cells pointed by the input
   * cell iterators. This function handles the case when the involved two DoF
   * handlers are different where the finite elements can be either identical
   * or different, but the triangulations are always different.
   *
   * In this case, neither vertex indices nor DoF indices for the two cells
   * are numbered in a same index system.
   *
   * These two surface triangulations are constructed respectively from a
   * common volume mesh by specifying a set of boundary ids. Therefore, the
   * map from surface/boundary mesh to volume mesh is needed.
   *
   * \alert{The map from surface/boundary mesh to volume mesh is only for the
   * coarse mesh, due to the behavior of @p GridGenerator::extract_boundary_mesh.}
   *
   * \alert{The template parameters @p dim and @p spacedim are for the
   * boundary mesh. For the original volume mesh, the corresponding dimensions
   * should be @p dim+1 and @p spacedim.}
   *
   * @param first_cell_iter
   * @param second_cell_iter
   * @param first_cell_mapping
   * @param second_cell_mapping
   * @param map_from_first_boundary_mesh_to_volume_mesh
   * @param map_from_second_boundary_mesh_to_volume_mesh
   * @param common_vertex_pair_dof_indices A vector of pairs of global DoF
   * indices in the corresponding DoF handlers. Each pair corresponds to a
   * common vertex shred by the two cells, if any. The first DoF index in the
   * pair is in the first cell with the first DoF handler, while the second
   * DoF index in the pair is in the second cell with the second DoF handler.
   * @param threshold
   * @return Cell neighboring type
   */
  template <int dim, int spacedim = dim>
  CellNeighboringType
  detect_cell_neighboring_type_for_different_triangulations(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator
      &first_cell_iter,
    const typename DoFHandler<dim, spacedim>::active_cell_iterator
                                 &second_cell_iter,
    const Mapping<dim, spacedim> &first_cell_mapping,
    const Mapping<dim, spacedim> &second_cell_mapping,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_first_boundary_mesh_to_volume_mesh,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_second_boundary_mesh_to_volume_mesh,
    std::vector<std::pair<types::global_dof_index, types::global_dof_index>>
                &common_vertex_pair_dof_indices,
    const double threshold = 1e-12)
  {
    const FiniteElement<dim, spacedim> &first_cell_fe =
      first_cell_iter->get_fe();
    const FiniteElement<dim, spacedim> &second_cell_fe =
      second_cell_iter->get_fe();

    /**
     * Get the iterators to the faces in the original volume triangulation.
     */
    auto first_face_iter_pos =
      map_from_first_boundary_mesh_to_volume_mesh.find(first_cell_iter);
    Assert(
      first_face_iter_pos != map_from_first_boundary_mesh_to_volume_mesh.end(),
      ExcMessage(
        "The associated face iterator in the volume triangulation cannot be found for the first cell iterator in the boundary mesh!"));
    auto first_face_iter = first_face_iter_pos->second;

    auto second_face_iter_pos =
      map_from_second_boundary_mesh_to_volume_mesh.find(second_cell_iter);
    Assert(
      second_face_iter_pos !=
        map_from_second_boundary_mesh_to_volume_mesh.end(),
      ExcMessage(
        "The associated face iterator in the volume triangulation cannot be found for the second cell iterator in the boundary mesh!"));
    auto second_face_iter = second_face_iter_pos->second;

    /**
     * Get the vertex indices in each face in the original volume
     * triangulation.
     */
    auto first_face_vertex_indices =
      get_vertex_indices_in_face<dim + 1, spacedim>(first_face_iter);
    auto second_face_vertex_indices =
      get_vertex_indices_in_face<dim + 1, spacedim>(second_face_iter);

    /**
     * Calculate the intersection of the two faces' vertex indices. This
     * operation is meaningful because the two faces are both pulled back into
     * the original volume triangulation.
     */
    CellNeighboringType cell_neighboring_type;
    std::vector<types::global_vertex_index>
      common_vertex_indices_in_volume_mesh;
    cell_neighboring_type =
      detect_cell_neighboring_type_for_same_triangulations<dim>(
        first_face_vertex_indices,
        second_face_vertex_indices,
        common_vertex_indices_in_volume_mesh);

    /**
     * According to deal.ii's documentation about @p extract_boundary_mesh,
     *
     * > The order of vertices of surface cells at the boundary and the
     * corresponding volume faces may not match in order to ensure that each
     * surface cell is associated with an outward facing normal. As a
     * consequence, if you want to match quantities on the faces of the
     * domain cells and on the cells of the surface mesh, you may have to
     * translate between vertex locations or quadrature points.
     *
     * Hence, the local index of each common vertex in its corresponding face
     * in the volume mesh is not the same as its local index in the cell in
     * the surface mesh. Therefore, the correlation mapping between cell
     * vertices in the surface mesh and those in the face in the volume mesh
     * should be constructed by matching vertex coordinates.
     */

    common_vertex_pair_dof_indices.clear();
    types::global_dof_index common_vertex_dof_index_in_first_cell;
    types::global_dof_index common_vertex_dof_index_in_second_cell;

    for (auto vertex_index_in_volume_mesh :
         common_vertex_indices_in_volume_mesh)
      {
        /**
         * Get the coordinates of the current common vertex.
         */
        auto first_face_vertex_found_pos =
          std::find(first_face_vertex_indices.cbegin(),
                    first_face_vertex_indices.cend(),
                    vertex_index_in_volume_mesh);

        Assert(first_face_vertex_found_pos != first_face_vertex_indices.cend(),
               ExcMessage("Cannot find the common vertex in the first face!"));

        /**
         * Get the coordinates of the current common vertex. \mynote{The
         * point returned from the member function @p TriaAccessor::vertex is
         * a reference.}
         */
        const Point<spacedim> &common_vertex_coords = first_face_iter->vertex(
          first_face_vertex_found_pos - first_face_vertex_indices.cbegin());

        if (first_cell_fe.dofs_per_cell > 1)
          {
            /**
             * Find the current common vertex in the first boundary cell by
             * coordinate matching.
             */
            const unsigned int vertex_local_index_in_first_cell =
              get_vertex_local_index_in_cell<dim, spacedim>(
                common_vertex_coords, first_cell_iter);
            AssertIndexRange(vertex_local_index_in_first_cell,
                             GeometryInfo<dim>::vertices_per_cell);

            /**
             * Get the DoF index associated with the common vertex in the
             * first cell.
             */
            common_vertex_dof_index_in_first_cell =
              get_dof_index_for_vertex_in_cell(first_cell_iter,
                                               first_cell_mapping,
                                               vertex_local_index_in_first_cell,
                                               threshold);
          }
        else
          {
            /**
             * Handle the case when the finite element order is 0, i.e. for
             * @p FE_DGQ. Set the common vertex DoF index to be the DoF in
             * the cell's interior, even though it is not associated to the
             * vertex.
             */
            common_vertex_dof_index_in_first_cell =
              first_cell_iter->dof_index(0);
          }

        if (second_cell_fe.dofs_per_cell > 1)
          {
            /**
             * Find the current common vertex in the second boundary cell by
             * coordinates.
             */
            const unsigned int vertex_local_index_in_second_cell =
              get_vertex_local_index_in_cell<dim, spacedim>(
                common_vertex_coords, second_cell_iter);
            AssertIndexRange(vertex_local_index_in_second_cell,
                             GeometryInfo<dim>::vertices_per_cell);

            /**
             * Get the DoF index associated with the common vertex in the
             * second cell.
             */
            common_vertex_dof_index_in_second_cell =
              get_dof_index_for_vertex_in_cell(
                second_cell_iter,
                second_cell_mapping,
                vertex_local_index_in_second_cell,
                threshold);
          }
        else
          {
            /**
             * Handle the case when the finite element order is 0, i.e. for
             * @p FE_DGQ. Set the common vertex DoF index to be the DoF in
             * the cell's interior, even though it is not associated to the
             * vertex.
             */
            common_vertex_dof_index_in_second_cell =
              second_cell_iter->dof_index(0);
          }

        /**
         * Add the pair of DoF indices associated with the common vertex in
         * the two boundary cells to the list.
         */
        common_vertex_pair_dof_indices.push_back(
          std::pair<types::global_dof_index, types::global_dof_index>(
            common_vertex_dof_index_in_first_cell,
            common_vertex_dof_index_in_second_cell));
      }

    switch (common_vertex_indices_in_volume_mesh.size())
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
          break;
      }

    return cell_neighboring_type;
  }


  /**
   * Detect the cell neighboring type for two cells by checking the
   * intersection of the DoF indices associated with their vertices. Because
   * there are vertex DoFs, the finite elements contained in the DoF handlers
   * should be @p H1. The intersection of the DoF indices is returned via
   * argument by reference.
   *
   * \alert{Comparison of DoF indices implies that the two cells should belong
   * to a same DoFHandler. Otherwise, there will be two sets of DoF indices,
   * which cannot be compared.}
   *
   * \mynote{N.B. The template argument @p dim of this function cannot be
   * automatically inducted by the compiler from its input arguments, since
   * even though the template argument @p GeometryInfo<dim>::vertices_per_cell
   * of the two input arguments @p first_cell_vertex_dof_indices and
   * @p second_cell_vertex_dof_indices contains @p dim, the whole
   * @p GeometryInfo<dim>::vertices_per_cell will be evaluated into an
   * integer by the compiler, so that @p dim will be discarded.}
   *
   * @param first_cell_vertex_dof_indices
   * @param second_cell_vertex_dof_indices
   * @param vertex_dof_index_intersection Before calling this function, this
   * variable should be cleared.
   * @return Cell neighboring type
   */
  template <int dim>
  CellNeighboringType
  detect_cell_neighboring_type_for_same_h1_dofhandlers(
    const std::array<types::global_dof_index,
                     GeometryInfo<dim>::vertices_per_cell>
      &first_cell_vertex_dof_indices,
    const std::array<types::global_dof_index,
                     GeometryInfo<dim>::vertices_per_cell>
                                         &second_cell_vertex_dof_indices,
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

    /**
     * Calculate the intersection of the two cells' vertex DoF indices.
     */
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
          break;
      }

    return cell_neighboring_type;
  }


  /**
   * Detect the cell neighboring type for the two cells pointed by the input
   * cell iterators. This function handles the case when the involved two DoF
   * handlers are the same, i.e. both the associated finite elements and
   * triangulations are the same.
   *
   * In this case, the vertex indices for the two cells are numbered in a same
   * index system. Same situation holds for DoF indices for the two cells.
   * Hence, when the finite element is @p H1, such as @p FE_Q, in each pair
   * of the returned vector @p common_vertex_pair_dof_indices, the two elements
   * are the same.
   *
   * When the finite element type is @p H1, e.g. @p FE_Q, we get the common
   * vertex DoF indices by taking the intersection of vertex indices in the
   * two cells.
   *
   * In principle, we can call the function
   * @p detect_cell_neighboring_type_for_same_triangulations as what is done
   * for the other case @p L2 element. However, it involves comparison of
   * point coordinates, which is less efficient.
   *
   * @param first_cell_iter
   * @param second_cell_iter
   * @param first_cell_mapping
   * @param second_cell_mapping
   * @param common_vertex_pair_dof_indices A vector of pairs of DoF indices.
   * Each pair corresponds to a common vertex and in each pair, the first DoF
   * index is in the first cell, while the second DoF index is in the second
   * cell.
   * @param threshold
   * @return
   */
  template <int dim, int spacedim = dim>
  CellNeighboringType
  detect_cell_neighboring_type_for_same_dofhandlers(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator
      &first_cell_iter,
    const typename DoFHandler<dim, spacedim>::active_cell_iterator
                                 &second_cell_iter,
    const Mapping<dim, spacedim> &first_cell_mapping,
    const Mapping<dim, spacedim> &second_cell_mapping,
    std::vector<std::pair<types::global_dof_index, types::global_dof_index>>
                &common_vertex_pair_dof_indices,
    const double threshold = 1e-12)
  {
    /**
     * Get the finite element, which should be the same for the two cell
     * iterators.
     */
    const FiniteElement<dim, spacedim> &fe = first_cell_iter->get_fe();

    if (fe.conforms(FiniteElementData<dim>::H1))
      {
        const unsigned int vertices_per_cell =
          GeometryInfo<dim>::vertices_per_cell;

        common_vertex_pair_dof_indices.clear();

        /**
         * Get the list of vertex DoF indices in the two cells
         */
        std::array<types::global_dof_index, vertices_per_cell>
          first_cell_vertex_dof_indices(get_vertex_dof_indices_in_cell(
            first_cell_iter, first_cell_mapping, threshold));

        std::array<types::global_dof_index, vertices_per_cell>
          second_cell_vertex_dof_indices(get_vertex_dof_indices_in_cell(
            second_cell_iter, second_cell_mapping, threshold));

        std::vector<types::global_dof_index> vertex_dof_index_intersection;

        detect_cell_neighboring_type_for_same_h1_dofhandlers<dim>(
          first_cell_vertex_dof_indices,
          second_cell_vertex_dof_indices,
          vertex_dof_index_intersection);

        /**
         * Fill the vector of pairs for @p common_vertex_dof_indices.
         */
        for (auto dof_index : vertex_dof_index_intersection)
          {
            common_vertex_pair_dof_indices.push_back(
              std::pair<types::global_dof_index, types::global_dof_index>(
                dof_index, dof_index));
          }

        CellNeighboringType cell_neighboring_type;
        switch (common_vertex_pair_dof_indices.size())
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
              break;
          }

        return cell_neighboring_type;
      }
    else if (fe.conforms(FiniteElementData<dim>::L2))
      {
        return detect_cell_neighboring_type_for_same_triangulations(
          first_cell_iter,
          second_cell_iter,
          first_cell_mapping,
          second_cell_mapping,
          common_vertex_pair_dof_indices);
      }
    else
      {
        Assert(false, ExcNotImplemented());

        return CellNeighboringType::None;
      }
  }


  /**
   * Detect the neighboring type of two cells based on their vertex indices.
   *
   * @param method_for_cell_neighboring_type
   * @param first_cell_iter
   * @param second_cell_iter
   * @param map_from_first_boundary_mesh_to_volume_mesh
   * @param map_from_second_boundary_mesh_to_volume_mesh
   * @param common_vertex_pair_local_indices
   * @return
   */
  template <int dim, int spacedim>
  CellNeighboringType
  detect_cell_neighboring_type(
    const DetectCellNeighboringTypeMethod method_for_cell_neighboring_type,
    const typename Triangulation<dim, spacedim>::cell_iterator &first_cell_iter,
    const typename Triangulation<dim, spacedim>::cell_iterator
      &second_cell_iter,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_first_boundary_mesh_to_volume_mesh,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_second_boundary_mesh_to_volume_mesh,
    std::vector<std::pair<unsigned int, unsigned int>>
      &common_vertex_pair_local_indices)
  {
    common_vertex_pair_local_indices.clear();

    CellNeighboringType cell_neighboring_type;

    switch (method_for_cell_neighboring_type)
      {
          case DetectCellNeighboringTypeMethod::SameTriangulations: {
            cell_neighboring_type =
              detect_cell_neighboring_type_for_same_triangulations<dim,
                                                                   spacedim>(
                first_cell_iter,
                second_cell_iter,
                common_vertex_pair_local_indices);

            break;
          }
          case DetectCellNeighboringTypeMethod::DifferentTriangulations: {
            cell_neighboring_type =
              detect_cell_neighboring_type_for_different_triangulations<
                dim,
                spacedim>(first_cell_iter,
                          second_cell_iter,
                          map_from_first_boundary_mesh_to_volume_mesh,
                          map_from_second_boundary_mesh_to_volume_mesh,
                          common_vertex_pair_local_indices);

            break;
          }
          default: {
            Assert(false,
                   ExcMessage(
                     "Invalid cell neighboring type detection method!"));
            cell_neighboring_type = CellNeighboringType::None;

            break;
          }
      }

    return cell_neighboring_type;
  }


  /**
   * Calculate the surface Jacobian determinant at the quadrature point
   * specified by its index.
   *
   * @param k3_index \f$k_3\f$ term index
   * @param quad_no Quadrature point index
   * @param shape_grad_matrix_table The data table storing the gradient values
   * of the shape functions. Refer to @p BEMValues::kx_shape_grad_matrix_table_for_same_panel.
   * @param support_points_in_real_cell A list of support points in the real
   * cell in the lexicographic order.
   * @return Surface Jacobian determinant or surface metric tensor
   */
  template <int spacedim>
  double
  surface_jacobian_det(
    const unsigned int                           k3_index,
    const unsigned int                           quad_no,
    const Table<2, LAPACKFullMatrixExt<double>> &shape_grad_matrix_table,
    const std::vector<Point<spacedim>>          &support_points_in_real_cell)
  {
    // Currently, only spacedim=3 is supported.
    Assert(spacedim == 3, ExcInternalError());

    /**
     * Extract the shape function's gradient matrix under the specified
     * \f$k_3\f$ index and quadrature point.
     */
    const LAPACKFullMatrixExt<double> &shape_grad_matrix_at_quad_point =
      shape_grad_matrix_table(k3_index, quad_no);

    double                      jacobian_det_squared = 0.0;
    LAPACKFullMatrixExt<double> jacobian_matrix_2x2(2, 2);
    for (unsigned int i = 0; i < spacedim; i++)
      {
        LAPACKFullMatrixExt<double> support_point_components =
          collect_two_components_from_point3(support_points_in_real_cell,
                                             i,
                                             (i + 1) % spacedim);
        support_point_components.mmult(jacobian_matrix_2x2,
                                       shape_grad_matrix_at_quad_point);
        jacobian_det_squared +=
          Utilities::fixed_power<2>(jacobian_matrix_2x2.determinant2x2());
      }

    return std::sqrt(jacobian_det_squared);
  }


  /**
   * Calculate the surface Jacobian determinant and the normal vector at the
   * quadrature point specified by its index.
   *
   * \mynote{N.B. The reversed lexicographic order appears for \f$K_y\f$
   * when the cell neighboring type is common edge. Then the calculated
   * normal vector \f$n_y\f$ has the opposite direction of the real one,
   * which should be negated in the subsequent calculation.}
   *
   * @param k3_index \f$k_3\f$ term index
   * @param quad_no Quadrature point index
   * @param mapping_shape_grad_matrix_table The data table storing the
   * gradient values
   * of the shape functions. Refer to
   * BEMValues::kx_shape_grad_matrix_table_for_same_panel.
   * @param mapping_index Index to the mapping object for the current cell.
   * @param support_points_in_real_cell A list of support points in the real
   * cell in the lexicographic order.
   * @param normal_vector
   * @param is_normal_vector_negated Whether the direction of the computed
   * normal vector should be negated.
   * @return Surface Jacobian determinant or surface metric tensor
   */
  template <int spacedim, typename RangeNumberType = double>
  RangeNumberType
  surface_jacobian_det_and_normal_vector(
    const unsigned int k3_index,
    const unsigned int quad_no,
    const Table<3, LAPACKFullMatrixExt<RangeNumberType>>
                      &mapping_shape_grad_matrix_table,
    const unsigned int mapping_index,
    const unsigned int mapping_n_shape_functions,
    const std::vector<Point<spacedim, RangeNumberType>>
                                         &support_points_in_real_cell,
    Tensor<1, spacedim, RangeNumberType> &normal_vector,
    const bool                            is_normal_vector_negated = false)
  {
    // Currently, only @p spacedim=3 is supported.
    Assert(spacedim == 3, ExcInternalError());

    /**
     * Extract the shape function's gradient matrix under the specified
     * mapping index, \f$k_3\f$ index and quadrature point index. The first
     * dimension of the gradient matrix is the shape function index and the
     * second dimension is coordinate component index in the unit cell.
     */
    const LAPACKFullMatrixExt<RangeNumberType>
      &mapping_shape_grad_matrix_at_quad_point =
        mapping_shape_grad_matrix_table(mapping_index, k3_index, quad_no);

    RangeNumberType surface_jacobian_det = RangeNumberType();
    LAPACKFullMatrixExt<RangeNumberType> jacobian_matrix_2x2(2, 2);
    RangeNumberType surface_jacobian_det_components[spacedim];
    for (unsigned int i = 0; i < spacedim; i++)
      {
        LAPACKFullMatrixExt<RangeNumberType> support_point_components =
          collect_two_components_from_point3(support_points_in_real_cell,
                                             mapping_n_shape_functions,
                                             i,
                                             (i + 1) % spacedim);
        support_point_components.mmult(jacobian_matrix_2x2,
                                       mapping_shape_grad_matrix_at_quad_point);
        surface_jacobian_det_components[i] =
          jacobian_matrix_2x2.determinant2x2();
        surface_jacobian_det +=
          Utilities::fixed_power<2>(surface_jacobian_det_components[i]);
      }

    surface_jacobian_det = std::sqrt(surface_jacobian_det);

    /**
     * This loop transform the vector \f$[J_{01}, J_{12}, J_{20}]/\abs{J}\f$
     * to \f$[J_{12}, J_{20}, J_{01}]/\abs{J}\f$, which is the normal vector.
     */
    for (unsigned int i = 0; i < spacedim; i++)
      {
        if (is_normal_vector_negated)
          {
            normal_vector[i] =
              -surface_jacobian_det_components[(i + 1) % spacedim] /
              surface_jacobian_det;
          }
        else
          {
            normal_vector[i] =
              surface_jacobian_det_components[(i + 1) % spacedim] /
              surface_jacobian_det;
          }
      }

    return surface_jacobian_det;
  }


  /**
   * Compute the Jacobian determinant and normal vector at a given point in the
   * unit cell.
   *
   * The point in the unit cell is not explicitly given, but the mapping shape
   * function's gradient matrix provided has been evaluated at this point.
   */
  template <int spacedim, typename RangeNumberType = double>
  RangeNumberType
  surface_jacobian_det_and_normal_vector(
    const std::vector<Point<spacedim, RangeNumberType>> &mapping_support_points,
    const LAPACKFullMatrixExt<RangeNumberType> &mapping_shape_grad_matrix,
    Tensor<1, spacedim, RangeNumberType>       &normal_vector,
    const bool is_normal_vector_negated = false)
  {
    // Currently, only @p spacedim=3 is supported.
    Assert(spacedim == 3, ExcInternalError());

    RangeNumberType surface_jacobian_det = RangeNumberType();
    LAPACKFullMatrixExt<RangeNumberType> jacobian_matrix_2x2(2, 2);
    RangeNumberType surface_jacobian_det_components[spacedim];
    for (unsigned int i = 0; i < spacedim; i++)
      {
        LAPACKFullMatrixExt<RangeNumberType> support_point_components =
          collect_two_components_from_point3(mapping_support_points,
                                             i,
                                             (i + 1) % spacedim);
        support_point_components.mmult(jacobian_matrix_2x2,
                                       mapping_shape_grad_matrix);
        surface_jacobian_det_components[i] =
          jacobian_matrix_2x2.determinant2x2();
        surface_jacobian_det +=
          Utilities::fixed_power<2>(surface_jacobian_det_components[i]);
      }

    surface_jacobian_det = std::sqrt(surface_jacobian_det);

    /**
     * This loop transform the vector \f$[J_{01}, J_{12}, J_{20}]/\abs{J}\f$
     * to \f$[J_{12}, J_{20}, J_{01}]/\abs{J}\f$, which is the normal vector.
     */
    for (unsigned int i = 0; i < spacedim; i++)
      {
        if (is_normal_vector_negated)
          {
            normal_vector[i] =
              -surface_jacobian_det_components[(i + 1) % spacedim] /
              surface_jacobian_det;
          }
        else
          {
            normal_vector[i] =
              surface_jacobian_det_components[(i + 1) % spacedim] /
              surface_jacobian_det;
          }
      }

    return surface_jacobian_det;
  }


  /**
   * Calculate the covariant transformation matrix for mapping the gradient in
   * local coordinate chart to global coordinates.
   *
   * The formula for calculating the covariant transformation matrix:
   * \f[
   * J G^{-1} = J (J^T J)^{-1}.
   * \f]
   * N.B. \f$J\f$ is the Jacobian matrix in \f$\mathbb{R}^{{\rm
   * spacedim}\times{\rm dim}}\f$. Therefore, the covariant transformation
   * matrix has the same sizes as \f$J\f$.
   *
   * @param k3_index
   * @param quad_no
   * @param mapping_shape_grad_matrix_table
   * @param mapping_index
   * @param mapping_n_shape_functions
   * @param support_points_in_real_cell
   * @return
   */
  template <int spacedim, typename RangeNumberType = double>
  LAPACKFullMatrixExt<RangeNumberType>
  surface_covariant_transformation(
    const unsigned int k3_index,
    const unsigned int quad_no,
    const Table<3, LAPACKFullMatrixExt<RangeNumberType>>
                      &mapping_shape_grad_matrix_table,
    const unsigned int mapping_index,
    const unsigned int mapping_n_shape_functions,
    const std::vector<Point<spacedim, RangeNumberType>>
      &support_points_in_real_cell)
  {
    // Currently, only @p spacedim=3 is supported.
    Assert(spacedim == 3, ExcInternalError());

    /**
     * Extract the shape function's gradient matrix under the specified
     * \f$k_3\f$ index and quadrature point, which will then be used for
     * calculating the Jacobian matrix.
     */
    const LAPACKFullMatrixExt<RangeNumberType>
      &mapping_shape_grad_matrix_at_quad_point =
        mapping_shape_grad_matrix_table(mapping_index, k3_index, quad_no);

    LAPACKFullMatrixExt<RangeNumberType> support_point_components =
      collect_components_from_points(support_points_in_real_cell,
                                     mapping_n_shape_functions);

    const unsigned int dim = mapping_shape_grad_matrix_at_quad_point.n();
    AssertDimension(dim, 2);

    LAPACKFullMatrixExt<RangeNumberType> jacobian_matrix(spacedim, dim);
    support_point_components.mmult(jacobian_matrix,
                                   mapping_shape_grad_matrix_at_quad_point);

    /**
     * Metric tensor
     */
    LAPACKFullMatrixExt<RangeNumberType> G(dim, dim);
    /**
     * Inverse of the metric tensor
     */
    LAPACKFullMatrixExt<RangeNumberType> G_inv(dim, dim);
    /**
     * \f$G=J^T J\f$
     */
    jacobian_matrix.Tmmult(G, jacobian_matrix);
    G_inv.invert(G);

    LAPACKFullMatrixExt<RangeNumberType> covariant(spacedim, dim);
    jacobian_matrix.mmult(covariant, G_inv);

    return covariant;
  }

  /**
   * Coordinate transformation of the specified quadrature point in the unit
   * cell to the real cell based on a list of support points in the real cell.
   * This version runs on the host.
   *
   * @param k3_index
   * @param quad_no Quadrature point index
   * @param mapping_shape_value_table Data table for the mapping shape
   * function values. Refer to BEMValues::kx_shape_value_table_for_same_panel.
   * @param mapping_support_points_in_real_cell A list of support points in
   * the real cell in the lexicographic order.
   * @return Point coordinates in the real cell, which has the spatial
   * dimension @p spacedim.
   */
  template <int spacedim, typename RangeNumberType = double>
  Point<spacedim, RangeNumberType>
  transform_quad_point_from_unit_to_permuted_real_cell(
    const unsigned int               k3_index,
    const unsigned int               quad_no,
    const Table<4, RangeNumberType> &mapping_shape_value_table,
    const unsigned int               mapping_index,
    const unsigned int               mapping_n_shape_functions,
    const std::vector<Point<spacedim, RangeNumberType>>
      &mapping_support_points_in_real_cell)
  {
    // @p mapping_n_shape_functions is the number of shape functions in the
    // actual mapping object, while the vector @p mapping_support_points_in_real_cell
    // is preallocated with memory for holding the shape functions in the
    // highest order mapping object. Therefore, we make this assertion.
    Assert(mapping_n_shape_functions <=
             mapping_support_points_in_real_cell.size(),
           ExcInternalError());

    Point<spacedim, RangeNumberType> real_coords;

    /**
     * Linear combination of support point coordinates and evaluation of
     * mapping shape functions at the specified area coordinates.
     */
    for (unsigned int i = 0; i < mapping_n_shape_functions; i++)
      {
        real_coords =
          real_coords +
          mapping_shape_value_table(mapping_index, i, k3_index, quad_no) *
            mapping_support_points_in_real_cell[i];
      }

    return real_coords;
  }


  /**
   * Generate the permutation of the polynomial space inverse numbering by
   * starting from the specified corner in the forward direction. The
   * numbering is returned from the function as the return value.
   *
   * @param fe
   * @param starting_corner Index of the starting corner point. Because there
   * are only four corners in a cell, its value belongs to \f$[0,1,2,3]\f$.
   * @return Numbering of the permuted DoFs
   */
  template <int dim, int spacedim>
  std::vector<unsigned int>
  generate_forward_dof_permutation(const FiniteElement<dim, spacedim> &fe,
                                   unsigned int starting_corner)
  {
    // Currently, only dim=2 and spacedim=3 are supported.
    Assert((dim == 2) && (spacedim == 3), ExcInternalError());

    using FE_Poly_short          = FE_Poly<dim, spacedim>;
    const FE_Poly_short &fe_poly = dynamic_cast<const FE_Poly_short &>(fe);

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
            dof_numbering_matrix(i, j) = poly_space_inverse_numbering[c];
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
                  dof_permutation[c] = dof_numbering_matrix(i, j);
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
                  dof_permutation[c] = dof_numbering_matrix(i, j);
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
                  dof_permutation[c] = dof_numbering_matrix(i, j);
                  c++;
                }
            }

          break;
        default:
          Assert(false, ExcInternalError());
          break;
      }

    return dof_permutation;
  }


  /**
   * Generate the permutation of the polynomial space inverse numbering by
   * starting from the specified corner in the forward direction. This
   * overloaded version has the returned vector as its argument.
   *
   * @param fe
   * @param starting_corner Index of the starting corner point. Because there
   * are only four corners in a cell, its value belongs to \f$[0,1,2,3]\f$.
   * @param dof_permutation Numbering of the permuted DoFs. Its memory should
   * be pre-allocated.
   */
  template <int dim, int spacedim>
  void
  generate_forward_dof_permutation(const FiniteElement<dim, spacedim> &fe,
                                   unsigned int               starting_corner,
                                   std::vector<unsigned int> &dof_permutation)
  {
    // Currently, only dim=2 and spacedim=3 are supported.
    Assert((dim == 2) && (spacedim == 3), ExcInternalError());

    using FE_Poly_short          = FE_Poly<dim, spacedim>;
    const FE_Poly_short &fe_poly = dynamic_cast<const FE_Poly_short &>(fe);

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
            dof_numbering_matrix(i, j) = poly_space_inverse_numbering[c];
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
                  dof_permutation[c] = dof_numbering_matrix(i, j);
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
                  dof_permutation[c] = dof_numbering_matrix(i, j);
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
                  dof_permutation[c] = dof_numbering_matrix(i, j);
                  c++;
                }
            }

          break;
        default:
          Assert(false, ExcInternalError());
          break;
      }
  }


  template <int dim, int spacedim>
  std::vector<unsigned int>
  generate_forward_mapping_support_point_permutation(
    const MappingQExt<dim, spacedim> &mapping,
    unsigned int                      starting_corner)
  {
    // Currently, only dim=2 and spacedim=3 are supported.
    Assert((dim == 2) && (spacedim == 3), ExcInternalError());
    const int poly_degree = mapping.polynomial_degree;

    std::vector<unsigned int> poly_space_inverse_numbering(
      FETools::lexicographic_to_hierarchic_numbering<dim>(poly_degree));
    std::vector<unsigned int> support_point_permutation(
      poly_space_inverse_numbering.size());

    // Store the inverse numbering into a matrix for further traversing.
    unsigned int             c = 0;
    FullMatrix<unsigned int> support_point_numbering_matrix(poly_degree + 1,
                                                            poly_degree + 1);
    for (int i = poly_degree; i >= 0; i--)
      {
        for (int j = 0; j <= poly_degree; j++)
          {
            support_point_numbering_matrix(i, j) =
              poly_space_inverse_numbering[c];
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
                  support_point_permutation[c] =
                    support_point_numbering_matrix(i, j);
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
                  support_point_permutation[c] =
                    support_point_numbering_matrix(i, j);
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
                  support_point_permutation[c] =
                    support_point_numbering_matrix(i, j);
                  c++;
                }
            }

          break;
        default:
          Assert(false, ExcInternalError());
          break;
      }

    return support_point_permutation;
  }


  template <int dim, int spacedim>
  void
  generate_forward_mapping_support_point_permutation(
    const MappingQExt<dim, spacedim> &mapping,
    unsigned int                      starting_corner,
    std::vector<unsigned int>        &support_point_permutation)
  {
    // Currently, only dim=2 and spacedim=3 are supported.
    Assert((dim == 2) && (spacedim == 3), ExcInternalError());
    const int poly_degree = mapping.get_degree();

    std::vector<unsigned int> poly_space_inverse_numbering(
      FETools::lexicographic_to_hierarchic_numbering<dim>(poly_degree));

    AssertDimension(support_point_permutation.size(),
                    poly_space_inverse_numbering.size());

    // Store the inverse numbering into a matrix for further traversing.
    unsigned int             c = 0;
    FullMatrix<unsigned int> support_point_numbering_matrix(poly_degree + 1,
                                                            poly_degree + 1);
    for (int i = poly_degree; i >= 0; i--)
      {
        for (int j = 0; j <= poly_degree; j++)
          {
            support_point_numbering_matrix(i, j) =
              poly_space_inverse_numbering[c];
            c++;
          }
      }

    switch (starting_corner)
      {
        case 0:
          support_point_permutation = poly_space_inverse_numbering;

          break;
        case 1:
          c = 0;
          for (int j = poly_degree; j >= 0; j--)
            {
              for (int i = poly_degree; i >= 0; i--)
                {
                  support_point_permutation[c] =
                    support_point_numbering_matrix(i, j);
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
                  support_point_permutation[c] =
                    support_point_numbering_matrix(i, j);
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
                  support_point_permutation[c] =
                    support_point_numbering_matrix(i, j);
                  c++;
                }
            }

          break;
        default:
          Assert(false, ExcInternalError());
          break;
      }
  }


  /**
   * Generate the permutation of the polynomial space inverse numbering by
   * starting from the specified corner in the backward direction. The
   * numbering is returned from this function via the return value.
   *
   * @param fe
   * @param starting_corner Index of the starting corner point. Because there
   * are only four corners in a cell, its value belongs to \f$[0,1,2,3]\f$.
   * @return Numbering of the permuted DoFs
   */
  template <int dim, int spacedim>
  std::vector<unsigned int>
  generate_backward_dof_permutation(const FiniteElement<dim, spacedim> &fe,
                                    unsigned int starting_corner)
  {
    // Currently, only dim=2 and spacedim=3 are supported.
    Assert((dim == 2) && (spacedim == 3), ExcInternalError());

    using FE_Poly_short          = FE_Poly<dim, spacedim>;
    const FE_Poly_short &fe_poly = dynamic_cast<const FE_Poly_short &>(fe);

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
            dof_numbering_matrix(i, j) = poly_space_inverse_numbering[c];
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
                  dof_permutation[c] = dof_numbering_matrix(i, j);
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
                  dof_permutation[c] = dof_numbering_matrix(i, j);
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
                  dof_permutation[c] = dof_numbering_matrix(i, j);
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
                  dof_permutation[c] = dof_numbering_matrix(i, j);
                  c++;
                }
            }

          break;
        default:
          Assert(false, ExcInternalError());
          break;
      }

    return dof_permutation;
  }


  /**
   * Generate the permutation of the polynomial space inverse numbering by
   * starting from the specified corner in the backward direction. This
   * overloaded version has the returned vector as its argument.
   *
   * @param fe
   * @param starting_corner Index of the starting corner point. Because there
   * are only four corners in a cell, its value belongs to \f$[0,1,2,3]\f$.
   * @return dof_permutation Numbering of the permuted DoFs. Its memory should
   * be pre-allocated.
   */
  template <int dim, int spacedim>
  void
  generate_backward_dof_permutation(const FiniteElement<dim, spacedim> &fe,
                                    unsigned int               starting_corner,
                                    std::vector<unsigned int> &dof_permutation)
  {
    // Currently, only dim=2 and spacedim=3 are supported.
    Assert((dim == 2) && (spacedim == 3), ExcInternalError());

    using FE_Poly_short          = FE_Poly<dim, spacedim>;
    const FE_Poly_short &fe_poly = dynamic_cast<const FE_Poly_short &>(fe);

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
            dof_numbering_matrix(i, j) = poly_space_inverse_numbering[c];
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
                  dof_permutation[c] = dof_numbering_matrix(i, j);
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
                  dof_permutation[c] = dof_numbering_matrix(i, j);
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
                  dof_permutation[c] = dof_numbering_matrix(i, j);
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
                  dof_permutation[c] = dof_numbering_matrix(i, j);
                  c++;
                }
            }

          break;
        default:
          Assert(false, ExcInternalError());
          break;
      }
  }


  template <int dim, int spacedim>
  std::vector<unsigned int>
  generate_backward_mapping_support_point_permutation(
    const MappingQExt<dim, spacedim> &mapping,
    unsigned int                      starting_corner)
  {
    // Currently, only dim=2 and spacedim=3 are supported.
    Assert((dim == 2) && (spacedim == 3), ExcInternalError());
    const int poly_degree = mapping.polynomial_degree;

    std::vector<unsigned int> poly_space_inverse_numbering(
      FETools::lexicographic_to_hierarchic_numbering<dim>(poly_degree));
    std::vector<unsigned int> support_point_permutation(
      poly_space_inverse_numbering.size());

    // Store the inverse numbering into a matrix for further traversing.
    unsigned int             c = 0;
    FullMatrix<unsigned int> support_point_numbering_matrix(poly_degree + 1,
                                                            poly_degree + 1);
    for (int i = poly_degree; i >= 0; i--)
      {
        for (int j = 0; j <= poly_degree; j++)
          {
            support_point_numbering_matrix(i, j) =
              poly_space_inverse_numbering[c];
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
                  support_point_permutation[c] =
                    support_point_numbering_matrix(i, j);
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
                  support_point_permutation[c] =
                    support_point_numbering_matrix(i, j);
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
                  support_point_permutation[c] =
                    support_point_numbering_matrix(i, j);
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
                  support_point_permutation[c] =
                    support_point_numbering_matrix(i, j);
                  c++;
                }
            }

          break;
        default:
          Assert(false, ExcInternalError());
          break;
      }

    return support_point_permutation;
  }


  template <int dim, int spacedim>
  void
  generate_backward_mapping_support_point_permutation(
    const MappingQExt<dim, spacedim> &mapping,
    unsigned int                      starting_corner,
    std::vector<unsigned int>        &support_point_permutation)
  {
    // Currently, only dim=2 and spacedim=3 are supported.
    Assert((dim == 2) && (spacedim == 3), ExcInternalError());
    const int poly_degree = mapping.get_degree();

    std::vector<unsigned int> poly_space_inverse_numbering(
      FETools::lexicographic_to_hierarchic_numbering<dim>(poly_degree));

    AssertDimension(support_point_permutation.size(),
                    poly_space_inverse_numbering.size());

    // Store the inverse numbering into a matrix for further traversing.
    unsigned int             c = 0;
    FullMatrix<unsigned int> support_point_numbering_matrix(poly_degree + 1,
                                                            poly_degree + 1);
    for (int i = poly_degree; i >= 0; i--)
      {
        for (int j = 0; j <= poly_degree; j++)
          {
            support_point_numbering_matrix(i, j) =
              poly_space_inverse_numbering[c];
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
                  support_point_permutation[c] =
                    support_point_numbering_matrix(i, j);
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
                  support_point_permutation[c] =
                    support_point_numbering_matrix(i, j);
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
                  support_point_permutation[c] =
                    support_point_numbering_matrix(i, j);
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
                  support_point_permutation[c] =
                    support_point_numbering_matrix(i, j);
                  c++;
                }
            }

          break;
        default:
          Assert(false, ExcInternalError());
          break;
      }
  }
} // namespace BEMTools

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_BEM_BEM_TOOLS_H_
