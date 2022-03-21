/**
 * @file bem_tools.h
 * @brief Introduction of bem_tools.h
 *
 * @date 2022-03-03
 * @author Jihuan Tian
 */
#ifndef INCLUDE_BEM_TOOLS_H_
#define INCLUDE_BEM_TOOLS_H_

#include <deal.II/base/point.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/base/table.h>
#include <deal.II/base/table_indices.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_base.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/full_matrix.templates.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

using namespace dealii;

namespace IdeoBEM
{
  namespace BEMTools
  {
    enum CellNeighboringType
    {
      SamePanel,
      CommonEdge,
      CommonVertex,
      Regular,
      None
    };


    /**
     * This function returns a list of DoF indices in the given cell iterator,
     * which is used for checking if two cells associated the iterators have
     * interaction. This function is called by @p GraphColoring::make_graph_coloring.
     *
     * Reference:
     * http://localhost/dealii-9.1.1-doc/namespaceGraphColoring.html#a670720d11f544a762592112ae5213876
     *
     * @param cell
     * @return
     */
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


    /**
     * Return a list of global vertex indices for all the vertices in the given
     * cell.
     *
     * @param cell
     * @return
     */
    template <int dim, int spacedim>
    std::array<types::global_vertex_index, GeometryInfo<dim>::vertices_per_cell>
    get_vertex_indices(
      const typename Triangulation<dim, spacedim>::cell_iterator &cell)
    {
      std::array<types::global_vertex_index,
                 GeometryInfo<dim>::vertices_per_cell>
        cell_vertex_indices;

      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; v++)
        {
          cell_vertex_indices[v] = cell->vertex_index(v);
        }

      return cell_vertex_indices;
    }


    /**
     * Return a list of global DoF indices associated with all the vertices in
     * the given cell.
     *
     * @param cell
     * @return
     */
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



    /**
     * Detect the cell neighboring type for two cells by checking the
     * intersection of their vertex indices.
     *
     * \mynote{N.B. The template argument @p dim of this function cannot be
     * automatically inducted by the compiler from its input arguments, since
     * even though the template argument @p GeometryInfo<dim>::vertices_per_cell
     * of the two input arguments @p first_cell_vertex_indices and
     * @p second_cell_vertex_indices contains @p dim, the whole
     * @p GeometryInfo<dim>::vertices_per_cell will be evaluated into an
     * integer by the compiler with @p dim discarded.}
     *
     * @param first_cell_vertex_indices
     * @param second_cell_vertex_indices
     * @param vertex_index_intersection
     * @return
     */
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
      // The arrays storing vertex indices should be sorted before calling
      // @p std::set_intersection.
      std::array<types::global_vertex_index,
                 GeometryInfo<dim>::vertices_per_cell>
        first_cell_vertex_indices_sorted(first_cell_vertex_indices);
      std::array<types::global_vertex_index,
                 GeometryInfo<dim>::vertices_per_cell>
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


    /**
     * Detect the cell neighboring type for two cells by checking the
     * intersection of the DoF indices associated with their vertices.
     *
     * \mynote{N.B. The template argument @p dim of this function cannot be
     * automatically inducted by the compiler from its input arguments, since
     * even though the template argument @p GeometryInfo<dim>::vertices_per_cell
     * of the two input arguments @p first_cell_vertex_dof_indices and
     * @p second_cell_vertex_dof_indices contains @p dim, the whole
     * @p GeometryInfo<dim>::vertices_per_cell will be evaluated into an
     * integer by the compiler with @p dim discarded.}
     *
     * @param first_cell_vertex_dof_indices
     * @param second_cell_vertex_dof_indices
     * @param vertex_dof_index_intersection
     * @return
     */
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
     * the
     * shape function gradients is determined by the given numbering @p dof_permuation.
     *
     * \mynote{The support points, shape functions and DoFs in the finite
     * element are enumerated in the hierarchic order by default.}
     *
     * @param fe
     * @param dof_permutation The numbering for accessing the shape functions in
     * the specified order.
     * @param p The area coordinates at which the shape function's gradient is to be
     * evaluated.
     * @return The matrix storing the gradient of each shape function. Its
     * dimension is @p dofs_per_cell*dim.
     */
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


    /**
     * Calculate the matrix which stores shape function gradient values with
     * respect to area coordinates. Each row of the matrix is the gradient of
     * one of the shape functions. The matrix rows corresponding to the shape
     * function gradients are arranged in the default hierarchic order.
     *
     * @param fe
     * @param p The area coordinates at which the shape function's gradient is to be
     * evaluated.
     * @return The matrix storing the gradient of each shape function. Its
     * dimension is @p dofs_per_cell*dim.
     */
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


    /**
     * Calculate the matrix which stores shape function gradient values with
     * respect to area coordinates. Each row of the matrix is the gradient of
     * one of the shape functions. The matrix rows corresponding to the shape
     * function gradients are arranged in the lexicographic order.
     *
     * \mynote{The support points, shape functions and DoFs in the finite
     * element are enumerated in the hierarchic order by default.}
     *
     * @param fe
     * @param p The area coordinates at which the shape function's gradient is to be
     * evaluated.
     * @return The matrix storing the gradient of each shape function. Its
     * dimension is @p dofs_per_cell*dim.
     */
    template <int dim, int spacedim>
    FullMatrix<double>
    shape_grad_matrix_in_lexicographic_order(
      const FiniteElement<dim, spacedim> &fe,
      const Point<dim> &                  p)
    {
      try
        {
          /**
           * Get the lexicographic numbering from
           * @p FE_Poly<PolynomialType,dim,spacedim>::get_poly_space_numbering_inverse.
           * An alternative is to call @p FETools::lexicographic_to_hierarchic_numbering.
           */
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


    /**
     * Evaluate a list of shape functions at the specified area coordinates. The
     * shape functions are arranged in the required order.
     *
     * @param fe
     * @param dof_permutation The numbering for accessing the shape functions in
     * the specified order.
     * @param p The area coordinates at which the shape functions are to be
     * evaluated.
     * @return A list of shape function values.
     */
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


    /**
     * Evaluate a list of shape functions at the specified area coordinates in
     * the default hierarchic order.
     *
     * @param fe
     * @param p The area coordinates at which the shape functions are to be
     * evaluated.
     * @return A list of shape function values.
     */
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


    /**
     * Evaluate a list of shape functions at the specified area coordinates in
     * the lexicographic order.
     *
     * @param fe
     * @param p The area coordinates at which the shape functions are to be
     * evaluated.
     * @return A list of shape function values.
     */
    template <int dim, int spacedim>
    Vector<double>
    shape_values_in_lexicographic_order(const FiniteElement<dim, spacedim> &fe,
                                        const Point<dim> &                  p)
    {
      try
        {
          using FE_Poly = FE_Poly<TensorProductPolynomials<dim>, dim, spacedim>;
          const FE_Poly &fe_poly = dynamic_cast<const FE_Poly &>(fe);

          // Use the inverse numbering of polynomial space to restore the tensor
          // product ordering of shape functions.
          return shape_values(fe,
                              fe_poly.get_poly_space_numbering_inverse(),
                              p);
        }
      catch (const std::bad_cast &e)
        {
          Assert(false, ExcInternalError());
          return Vector<double>();
        }
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
    template <int spacedim>
    FullMatrix<double>
    collect_two_components_from_point3(
      const std::vector<Point<spacedim>> &points,
      const unsigned int                  first_component,
      const unsigned int                  second_component)
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


    /**
     * Calculate a list of support points in the real cell in the order
     * required by the given numbering.
     *
     * @param cell
     * @param fe
     * @param mapping Geometric mapping object used for transforming support
     * points from the unit cell to the real cell.
     * @param dof_permutation The numbering for accessing the support points in
     * the specified order.
     * @return A list of support points in the real cell.
     *
     * \mynote{N.B. Each support point in the real cell has the space dimension
     * @p spacedim, while each support point in the unit cell has the manifold
     * dimension @p dim.}
     */
    template <int dim, int spacedim>
    std::vector<Point<spacedim>>
    get_support_points_in_real_cell(
      const typename Triangulation<dim, spacedim>::cell_iterator &cell,
      const FiniteElement<dim, spacedim> &                        fe,
      const MappingQGeneric<dim, spacedim> &                      mapping,
      const std::vector<unsigned int> &dof_permutation)
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
      std::vector<Point<spacedim>> support_points_in_real_cell(dofs_per_cell);

      Assert(fe.has_support_points(),
             ExcMessage("The finite element should have support points."));

      // Get the list of support points in the unit cell in the default
      // hierarchical ordering.
      const std::vector<Point<dim>> &unit_support_points =
        fe.get_unit_support_points();

      // Transform the support points from unit cell to real cell.
      for (unsigned int i = 0; i < dofs_per_cell; i++)
        {
          support_points_in_real_cell.at(i) =
            mapping.transform_unit_to_real_cell(
              cell, unit_support_points.at(dof_permutation.at(i)));
        }

      return support_points_in_real_cell;
    }

    /**
     * Calculate a list of support points in the real cell in the default
     * hierarchic order.
     *
     * @param cell
     * @param fe
     * @param mapping Geometric mapping object used for transforming support
     * points from the unit cell to the real cell.
     * @return A list of support points in the real cell in the hierarchic order.
     *
     * \mynote{N.B. Each support point in the real cell has the space dimension
     * @p spacedim, while each support point in the unit cell has the manifold
     * dimension @p dim.}
     */
    template <int dim, int spacedim>
    std::vector<Point<spacedim>>
    get_hierarchic_support_points_in_real_cell(
      const typename Triangulation<dim, spacedim>::cell_iterator &cell,
      const FiniteElement<dim, spacedim> &                        fe,
      const MappingQGeneric<dim, spacedim> &                      mapping)
    {
      const unsigned int           dofs_per_cell = fe.dofs_per_cell;
      std::vector<Point<spacedim>> support_points_in_real_cell(dofs_per_cell);

      Assert(fe.has_support_points(),
             ExcMessage("The finite element should have support points."));

      // Get the list of support points in the unit cell in the default
      // hierarchical ordering.
      const std::vector<Point<dim>> &unit_support_points =
        fe.get_unit_support_points();

      // Transform the support points from unit cell to real cell.
      for (unsigned int i = 0; i < dofs_per_cell; i++)
        {
          support_points_in_real_cell.at(i) =
            mapping.transform_unit_to_real_cell(cell,
                                                unit_support_points.at(i));
        }

      return support_points_in_real_cell;
    }


    /**
     * Calculate a list of support points in the real cell in the default
     * hierarchic order.
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
    get_hierarchic_support_points_in_real_cell(
      const typename Triangulation<dim, spacedim>::cell_iterator &cell,
      const FiniteElement<dim, spacedim> &                        fe,
      const MappingQGeneric<dim, spacedim> &                      mapping,
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
          support_points_in_reall_cell.at(i) =
            mapping.transform_unit_to_real_cell(cell,
                                                unit_support_points.at(i));
        }
    }


    /**
     * Calculate a list of support points in the real cell in the lexicographic
     * order.
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
      const FiniteElement<dim, spacedim> &                        fe,
      const MappingQGeneric<dim, spacedim> &                      mapping)
    {
      const unsigned int           dofs_per_cell = fe.dofs_per_cell;
      std::vector<Point<spacedim>> support_points_in_real_cell(dofs_per_cell);

      try
        {
          using FE_Poly = FE_Poly<TensorProductPolynomials<dim>, dim, spacedim>;
          const FE_Poly &fe_poly = dynamic_cast<const FE_Poly &>(fe);

          // The Mapping object should has the same degree as the @p FE_Q object,
          // so that the support points in the real cell requested from the
          // @p mapping object are compatible with the support points of the
          // finite element @p fe.
          Assert(fe_poly.get_degree() == mapping.get_degree(),
                 ExcInternalError());

          Assert(fe.has_support_points(),
                 ExcMessage("The finite element should have support points."));

          std::vector<unsigned int> poly_space_inverse_numbering(
            fe_poly.get_poly_space_numbering_inverse());
          support_points_in_real_cell = get_support_points_in_real_cell(
            cell, fe, mapping, poly_space_inverse_numbering);
        }
      catch (const std::bad_cast &e)
        {
          Assert(false, ExcInternalError());
        }

      return support_points_in_real_cell;
    }


    /**
     * Calculate the surface Jacobian determinant at the specified area
     * coordinates. Support points in the real cell should have been
     * reordered to the lexicographic order before the calculation.
     *
     * @param fe
     * @param support_points_in_real_cell A list of support points in the real
     * cell in the lexicographic order.
     * @param p The area coordinates of the point in the unit cell.
     * @return Surface Jacobian determinant or surface metric tensor
     */
    template <int dim, int spacedim>
    double
    surface_jacobian_det(
      const FiniteElement<dim, spacedim> &fe,
      const std::vector<Point<spacedim>> &support_points_in_real_cell,
      const Point<dim> &                  p)
    {
      /**
       * \mynote{Currently, only @p dim=2 and @p spacedim=3 are supported.}
       */
      Assert((dim == 2) && (spacedim == 3), ExcInternalError());

      Assert(fe.has_support_points(),
             ExcMessage("The finite element should have support points."));

      /**
       * Calculate the shape function's gradient matrix, the rows of which are
       * in the lexicographic order.
       */
      FullMatrix<double> shape_grad_matrix_at_p(
        shape_grad_matrix_in_lexicographic_order(fe, p));

      /**
       * Calculate those 2x2 Jacobian determinants appearing in the surface
       * metric
       * tensor, namely,\f$J_{01}\f$, \f$J_{12}\f$ and \f$J_{20}\f$.
       */
      double             jacobian_det_squared = 0.0;
      FullMatrix<double> jacobian_matrix_2x2(dim, dim);
      for (unsigned int i = 0; i < spacedim; i++)
        {
          FullMatrix<double> support_point_components =
            collect_two_components_from_point3(support_points_in_real_cell,
                                               i,
                                               (i + 1) % spacedim);
          support_point_components.mmult(jacobian_matrix_2x2,
                                         shape_grad_matrix_at_p);
          jacobian_det_squared +=
            std::pow(jacobian_matrix_2x2.determinant(), 2);
        }

      return std::sqrt(jacobian_det_squared);
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
      const unsigned int                  k3_index,
      const unsigned int                  quad_no,
      const Table<2, FullMatrix<double>> &shape_grad_matrix_table,
      const std::vector<Point<spacedim>> &support_points_in_real_cell)
    {
      // Currently, only spacedim=3 is supported.
      Assert(spacedim == 3, ExcInternalError());

      /**
       * Extract the shape function's gradient matrix under the specified
       * \f$k_3\f$ index and quadrature point.
       */
      const FullMatrix<double> &shape_grad_matrix_at_quad_point =
        shape_grad_matrix_table(k3_index, quad_no);

      double             jacobian_det_squared = 0.0;
      FullMatrix<double> jacobian_matrix_2x2(2, 2);
      for (unsigned int i = 0; i < spacedim; i++)
        {
          FullMatrix<double> support_point_components =
            collect_two_components_from_point3(support_points_in_real_cell,
                                               i,
                                               (i + 1) % spacedim);
          support_point_components.mmult(jacobian_matrix_2x2,
                                         shape_grad_matrix_at_quad_point);
          jacobian_det_squared +=
            std::pow(jacobian_matrix_2x2.determinant(), 2);
        }

      return std::sqrt(jacobian_det_squared);
    }


    /**
     * Calculate the surface Jacobian determinant and the normal vector at the
     * specified coordinates in the unit cell. Support points in the real cell
     * should have been reordered to the lexicographic or reversed lexicographic
     * order before the calculation.
     *
     * \mynote{N.B. The reversed lexicographic order appears for \f$K_y\f$ when
     * the cell neighboring type is common edge. Then the calculated normal
     * vector \f$n_y\f$ has the opposite direction of the real one, which should
     * be negated in the subsequent calculation.}
     *
     * @param fe
     * @param support_points_in_real_cell A list of support points in the real
     * cell in the lexicographic or reversed lexicographic order.
     * @param p The area coordinates of the point in the unit cell.
     * @param normal_vector [out] The calculated normal vector. If it belongs to
     * \f$K_y\f$ and the cell neighboring type is common edge, it should be
     * negated in the subsequent calculation.
     * @return Surface Jacobian determinant or surface metric tensor
     */
    template <int dim, int spacedim>
    double
    surface_jacobian_det_and_normal_vector(
      const FiniteElement<dim, spacedim> &fe,
      const std::vector<Point<spacedim>> &support_points_in_real_cell,
      const Point<dim> &                  p,
      Tensor<1, spacedim> &               normal_vector)
    {
      /**
       * \mynote{Currently, only @p dim=2 and @p spacedim=3 are supported.}
       */
      Assert((dim == 2) && (spacedim == 3), ExcInternalError());

      Assert(fe.has_support_points(),
             ExcMessage("The finite element should have support points."));

      /**
       * Calculate the shape function's gradient matrix, the rows of which are
       * in the lexicographic order.
       */
      FullMatrix<double> shape_grad_matrix_at_p(
        shape_grad_matrix_in_lexicographic_order(fe, p));

      /**
       * Calculate those 2x2 Jacobian determinants appearing in the surface
       * metric
       * tensor, namely,\f$J_{01}\f$, \f$J_{12}\f$ and \f$J_{20}\f$. These
       * components are stored in the array @p surface_jacobian_det_components.
       */
      double             surface_jacobian_det = 0.0;
      FullMatrix<double> jacobian_matrix_2x2(dim, dim);
      double             surface_jacobian_det_components[spacedim];
      for (unsigned int i = 0; i < spacedim; i++)
        {
          FullMatrix<double> support_point_components =
            collect_two_components_from_point3(support_points_in_real_cell,
                                               i,
                                               (i + 1) % spacedim);
          support_point_components.mmult(jacobian_matrix_2x2,
                                         shape_grad_matrix_at_p);
          surface_jacobian_det_components[i] =
            jacobian_matrix_2x2.determinant();
          surface_jacobian_det +=
            std::pow(surface_jacobian_det_components[i], 2);
        }

      surface_jacobian_det = std::sqrt(surface_jacobian_det);

      /**
       * This loop transform the vector \f$[J_{01}, J_{12}, J_{20}]/\abs{J}\f$ to
       * \f$[J_{12}, J_{20}, J_{01}]/\abs{J}\f$, which is the normal vector.
       */
      for (unsigned int i = 0; i < spacedim; i++)
        {
          normal_vector[i] =
            surface_jacobian_det_components[(i + 1) % spacedim] /
            surface_jacobian_det;
        }

      return surface_jacobian_det;
    }


    /**
     * Calculate the surface Jacobian determinant and the normal vector at the
     * quadrature point specified by its index.
     *
     * @param k3_index \f$k_3\f$ term index
     * @param quad_no Quadrature point index
     * @param shape_grad_matrix_table The data table storing the gradient values
     * of the shape functions. Refer to @p BEMValues::kx_shape_grad_matrix_table_for_same_panel.
     * @param support_points_in_real_cell A list of support points in the real
     * cell in the lexicographic order.
     * @param normal_vector
     * @return Surface Jacobian determinant or surface metric tensor
     */
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

      /**
       * Extract the shape function's gradient matrix under the specified
       * \f$k_3\f$ index and quadrature point.
       */
      const FullMatrix<double> &shape_grad_matrix_at_quad_point =
        shape_grad_matrix_table(k3_index, quad_no);

      double             surface_jacobian_det = 0.0;
      FullMatrix<double> jacobian_matrix_2x2(2, 2);
      double             surface_jacobian_det_components[spacedim];
      for (unsigned int i = 0; i < spacedim; i++)
        {
          FullMatrix<double> support_point_components =
            collect_two_components_from_point3(support_points_in_real_cell,
                                               i,
                                               (i + 1) % spacedim);
          support_point_components.mmult(jacobian_matrix_2x2,
                                         shape_grad_matrix_at_quad_point);
          surface_jacobian_det_components[i] =
            jacobian_matrix_2x2.determinant();
          surface_jacobian_det +=
            std::pow(surface_jacobian_det_components[i], 2);
        }

      surface_jacobian_det = std::sqrt(surface_jacobian_det);

      /**
       * This loop transform the vector \f$[J_{01}, J_{12}, J_{20}]/\abs{J}\f$ to
       * \f$[J_{12}, J_{20}, J_{01}]/\abs{J}\f$, which is the normal vector.
       */
      for (unsigned int i = 0; i < spacedim; i++)
        {
          normal_vector[i] =
            surface_jacobian_det_components[(i + 1) % spacedim] /
            surface_jacobian_det;
        }

      return surface_jacobian_det;
    }


    /**
     * Coordinate transformation from the unit cell to the real cell based on a
     * list of support points in the real cell.
     *
     * @param fe
     * @param support_points_in_real_cell A list of permuted support points in
     * the real cell in the lexicographic order
     * @param area_coords Area coordinates in the unit cell, which has the
     * manifold dimension @p dim.
     * @return Point coordinates in the real cell, which has the spatial
     * dimension @p spacedim.
     */
    template <int dim, int spacedim>
    Point<spacedim>
    transform_unit_to_permuted_real_cell(
      const FiniteElement<dim, spacedim> &fe,
      const std::vector<Point<spacedim>> &support_points_in_real_cell,
      const Point<dim> &                  area_coords)
    {
      const unsigned int dofs_per_cell = fe.dofs_per_cell;

      /**
       * Since iso-parametric finite element is assumed, we check the equality
       * between the number of DoFs and the number of support points.
       *
       * \mynote{DoFs are related to the basis functions for spanning the
       * solution function space, while support points are related to the shape
       * functions for representing the geometry.}
       */
      Assert(dofs_per_cell == support_points_in_real_cell.size(),
             ExcDimensionMismatch(dofs_per_cell,
                                  support_points_in_real_cell.size()));

      Point<spacedim> real_coords;
      Vector<double>  shape_values_vector(
        shape_values_in_lexicographic_order(fe, area_coords));

      /**
       * Linear combination of support point coordinates and evaluation of shape
       * functions at the specified area coordinates.
       */
      for (unsigned int i = 0; i < dofs_per_cell; i++)
        {
          real_coords = real_coords + shape_values_vector(i) *
                                        support_points_in_real_cell.at(i);
        }

      return real_coords;
    }


    /**
     * Coordinate transformation of the specified quadrature point in the unit
     * cell to the real cell based on a list of support points in the real cell.
     *
     * @param k3_index
     * @param quad_no Quadrature point index
     * @param shape_value_table Data table for the shape function values. Refer
     * to @p BEMValues::kx_shape_value_table_for_same_panel.
     * @param support_points_in_real_cell A list of support points in the real
     * cell in the lexicographic order.
     * @return Point coordinates in the real cell, which has the spatial
     * dimension @p spacedim.
     */
    template <int spacedim>
    Point<spacedim>
    transform_unit_to_permuted_real_cell(
      const unsigned int                  k3_index,
      const unsigned int                  quad_no,
      const Table<3, double> &            shape_value_table,
      const std::vector<Point<spacedim>> &support_points_in_real_cell)
    {
      /**
       * Since iso-parametric finite element is assumed, we check the equality
       * between the number of DoFs and the number of support points. The latter
       * is equal to the size of the first dimension of the data table.
       *
       * \mynote{DoFs are related to the basis functions for spanning the
       * solution function space, while support points are related to the shape
       * functions for representing the geometry.}
       */
      const unsigned int dofs_per_cell = shape_value_table.size(0);

      Assert(dofs_per_cell == support_points_in_real_cell.size(),
             ExcDimensionMismatch(dofs_per_cell,
                                  support_points_in_real_cell.size()));

      Point<spacedim> real_coords;

      /**
       * Linear combination of support point coordinates and evaluation of shape
       * functions at the specified area coordinates.
       */
      for (unsigned int i = 0; i < dofs_per_cell; i++)
        {
          real_coords = real_coords + shape_value_table(i, k3_index, quad_no) *
                                        support_points_in_real_cell.at(i);
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

      try
        {
          std::vector<unsigned int> poly_space_inverse_numbering(
            FETools::lexicographic_to_hierarchic_numbering(fe));
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
                  dof_numbering_matrix(i, j) =
                    poly_space_inverse_numbering.at(c);
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

      try
        {
          std::vector<unsigned int> poly_space_inverse_numbering(
            FETools::lexicographic_to_hierarchic_numbering(fe));

          const int poly_degree = fe.degree;
          Assert((poly_degree + 1) * (poly_degree + 1) == fe.dofs_per_cell,
                 ExcDimensionMismatch((poly_degree + 1) * (poly_degree + 1),
                                      fe.dofs_per_cell));
          Assert(dof_permutation.size() == fe.dofs_per_cell,
                 ExcDimensionMismatch(dof_permutation.size(),
                                      fe.dofs_per_cell));

          // Store the inverse numbering into a matrix for further traversing.
          unsigned int             c = 0;
          FullMatrix<unsigned int> dof_numbering_matrix(poly_degree + 1,
                                                        poly_degree + 1);
          for (int i = poly_degree; i >= 0; i--)
            {
              for (int j = 0; j <= poly_degree; j++)
                {
                  dof_numbering_matrix(i, j) =
                    poly_space_inverse_numbering.at(c);
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

      std::vector<unsigned int> poly_space_inverse_numbering(
        FETools::lexicographic_to_hierarchic_numbering(fe));
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


    /**
     * Generate the permutation of the polynomial space inverse numbering by
     * starting from the specified corner in the backward direction. This
     * overloaded version has the returned vector as its argument.
     *
     * @param fe
     * @param starting_corner Index of the starting corner point. Because there
     * are only four corners in a cell, its value belongs to \f$[0,1,2,3]\f$.
     * @dof_permutation Numbering of the permuted DoFs. Its memory should
     * be pre-allocated.
     */
    template <int dim, int spacedim>
    void
    generate_backward_dof_permutation(
      const FiniteElement<dim, spacedim> &fe,
      unsigned int                        starting_corner,
      std::vector<unsigned int> &         dof_permutation)
    {
      // Currently, only dim=2 and spacedim=3 are supported.
      Assert((dim == 2) && (spacedim == 3), ExcInternalError());

      try
        {
          std::vector<unsigned int> poly_space_inverse_numbering(
            FETools::lexicographic_to_hierarchic_numbering(fe));

          const int poly_degree = fe.degree;
          Assert((poly_degree + 1) * (poly_degree + 1) == fe.dofs_per_cell,
                 ExcDimensionMismatch((poly_degree + 1) * (poly_degree + 1),
                                      fe.dofs_per_cell));
          Assert(dof_permutation.size() == fe.dofs_per_cell,
                 ExcDimensionMismatch(dof_permutation.size(),
                                      fe.dofs_per_cell));

          // Store the inverse numbering into a matrix for further traversing.
          unsigned int             c = 0;
          FullMatrix<unsigned int> dof_numbering_matrix(poly_degree + 1,
                                                        poly_degree + 1);
          for (int i = poly_degree; i >= 0; i--)
            {
              for (int j = 0; j <= poly_degree; j++)
                {
                  dof_numbering_matrix(i, j) =
                    poly_space_inverse_numbering.at(c);
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


    /**
     * Permute a vector by using the given permutation indices to access its
     * elements.
     *
     * @param input_vector
     * @param permutation_indices
     * @return Permuted vector
     */
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
     * Permute a vector by using the given permutation indices to access its
     * elements. The result vector is returned as argument.
     *
     * @param input_vector
     * @param permutation_indices
     * @param permuted_vector Permuted vector, the memory of which should be
     * pre-allocated.
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
  } // namespace BEMTools
} // namespace IdeoBEM


#endif /* INCLUDE_BEM_TOOLS_H_ */
