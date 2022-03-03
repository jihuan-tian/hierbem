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
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/full_matrix.templates.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "bem_tools.h"
#include "bem_values.h"
#include "quadrature.templates.h"
#include "sauter_quadrature.h"

using namespace dealii;

namespace LaplaceBEM
{
  using namespace BEMTools;

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
    /**
     * The intersection set of the vertex DoF indices for the two cells
     * \f$K_x\f$ and \f$K_y\f$.
     */
    std::vector<types::global_dof_index> vertex_dof_index_intersection;

    /**
     * List of support points in the real cell \f$K_x\f$ in the hierarchical
     * order.
     */
    std::vector<Point<3>> kx_support_points_hierarchical;
    /**
     * List of support points in the real cell \f$K_y\f$ in the hierarchical
     * order.
     */
    std::vector<Point<3>> ky_support_points_hierarchical;

    /**
     * Permuted list of support points in the real cell \f$K_x\f$ in the
     * lexicographic order in the same panel case and regular case, and
     * determined by @p kx_local_dof_permutation in the common edge case and
     * common vertex case.
     */
    std::vector<Point<3>> kx_support_points_permuted;
    /**
     * Permuted list of support points in the real cell \f$K_y\f$ in the
     * lexicographic order in the same panel case and regular case, and
     * determined by @p ky_local_dof_permutation in the common edge case and
     * common vertex case.
     */
    std::vector<Point<3>> ky_support_points_permuted;

    /**
     * The list of DoF indices in \f$K_x\f$ which are ordered in the
     * hierarchical order. This is directly retrieved from the function
     * @p DoFHandler::cell_iterator::get_dof_indices.
     */
    std::vector<types::global_dof_index> kx_local_dof_indices_hierarchical;
    /**
     * The list of DoF indices in \f$K_y\f$ which are ordered in the
     * hierarchical order. This is directly retrieved from the function
     * @p DoFHandler::cell_iterator::get_dof_indices.
     */
    std::vector<types::global_dof_index> ky_local_dof_indices_hierarchical;

    /**
     * The numbering used for accessing the list of DoFs in \f$K_x\f$ in the
     * lexicographic order, where the list of DoFs are stored in the
     * hierarchical order.
     */
    std::vector<unsigned int> kx_fe_poly_space_numbering_inverse;
    /**
     * The numbering used for accessing the list of DoFs in \f$K_y\f$ in the
     * lexicographic order, where the list of DoFs are stored in the
     * hierarchical order.
     */
    std::vector<unsigned int> ky_fe_poly_space_numbering_inverse;
    /**
     * The numbering used for accessing the list of DoFs in \f$K_y\f$ in the
     * reversed lexicographic order, where the list of DoFs are stored in the
     * hierarchical order.
     *
     * \mynote{This numbering occurs when \f$K_x\f$ and \f$K_y\f$ share a
     * common edge.}
     */
    std::vector<unsigned int> ky_fe_reversed_poly_space_numbering_inverse;

    /**
     * The numbering used for accessing the list of support points and
     * associated DoF indices in \f$K_x\f$ in the lexicographic order by
     * starting from a specific vertex, where the list of support points and
     * associated DoF indices are stored in the hierarchical order.
     *
     * \mynote{"By starting from a specific vertex" means:
     * 1. In the same panel case, this numbering is not used because the first
     * vertex is the starting point by default.
     * 2. In the common edge case, start from the vertex which is the starting
     * point of the common edge.
     * 3. In the common vertex case, start from the common vertex.
     * 4. In the regular panel case, same as the same panel case.}
     */
    std::vector<unsigned int> kx_local_dof_permutation;
    /**
     * The numbering used for accessing the list of support points and
     * associated DoF indices in \f$K_y\f$ in the lexicographic order or the
     * reversed lexicographic order by starting from a specific vertex, where
     * the list of support points and associated DoF indices are stored in the
     * hierarchical order.
     *
     * \mynote{"By starting from a specific vertex" means:
     * 1. In the same panel case, this numbering is not used because the first
     * vertex is the starting point by default.
     * 2. In the common edge case, start from the vertex which is the starting
     * point of the common edge. And the list of support points and associated
     * DoF indices are accessed in the reversed lexicographic order. Then the
     * cell orientation is reversed and the calculated normal vector should be
     * negated.
     * 3. In the common vertex case, start from the common vertex. And the list
     * of support points and associated DoF indices are accessed in the
     * lexicographic order.
     * 4. In the regular panel case, same as the same panel case.}
     */
    std::vector<unsigned int> ky_local_dof_permutation;

    // The first dimension of the following data tables is the \f$k_3\f$ index.
    /**
     * Jacobian from the unit cell to the real cell \f$K_x\f$ for each
     * \f$k_3\f$ term and at each quadrature point for the same panel case.
     */
    Table<2, double> kx_jacobians_same_panel;
    /**
     * Jacobian from the unit cell to the real cell \f$K_x\f$ for each
     * \f$k_3\f$ term and at each quadrature point for the common edge case.
     */
    Table<2, double> kx_jacobians_common_edge;
    /**
     * Jacobian from the unit cell to the real cell \f$K_x\f$ for each
     * \f$k_3\f$ term and at each quadrature point for the common vertex case.
     */
    Table<2, double> kx_jacobians_common_vertex;
    /**
     * Jacobian from the unit cell to the real cell \f$K_x\f$ for each
     * \f$k_3\f$ term and at each quadrature point for the regular case.
     */
    Table<2, double> kx_jacobians_regular;

    /**
     * Normal vector at each quadrature point in the real cell \f$K_x\f$ for
     * the same panel case.
     */
    Table<2, Tensor<1, 3>> kx_normals_same_panel;
    /**
     * Normal vector at each quadrature point in the real cell \f$K_x\f$ for
     * the common edge case.
     */
    Table<2, Tensor<1, 3>> kx_normals_common_edge;
    /**
     * Normal vector at each quadrature point in the real cell \f$K_x\f$ for
     * the common vertex case.
     */
    Table<2, Tensor<1, 3>> kx_normals_common_vertex;
    /**
     * Normal vector at each quadrature point in the real cell \f$K_x\f$ for
     * the regular case.
     */
    Table<2, Tensor<1, 3>> kx_normals_regular;

    /**
     * Coordinates in the real cell \f$K_x\f$ for each \f$k_3\f$ term and each
     * quadrature point for the same panel case.
     */
    Table<2, Point<3>> kx_quad_points_same_panel;
    /**
     * Coordinates in the real cell \f$K_x\f$ for each \f$k_3\f$ term and each
     * quadrature point for the common edge case.
     */
    Table<2, Point<3>> kx_quad_points_common_edge;
    /**
     * Coordinates in the real cell \f$K_x\f$ for each \f$k_3\f$ term and each
     * quadrature point for the common vertex case.
     */
    Table<2, Point<3>> kx_quad_points_common_vertex;
    /**
     * Coordinates in the real cell \f$K_x\f$ for each \f$k_3\f$ term and each
     * quadrature point for the regular case.
     */
    Table<2, Point<3>> kx_quad_points_regular;


    /**
     * Jacobian from the unit cell to the real cell \f$K_y\f$ for each
     * \f$k_3\f$ term and at each quadrature point for the same panel case.
     */
    Table<2, double> ky_jacobians_same_panel;
    /**
     * Jacobian from the unit cell to the real cell \f$K_y\f$ for each
     * \f$k_3\f$ term and at each quadrature point for the common edge case.
     */
    Table<2, double> ky_jacobians_common_edge;
    /**
     * Jacobian from the unit cell to the real cell \f$K_y\f$ for each
     * \f$k_3\f$ term and at each quadrature point for the common vertex case.
     */
    Table<2, double> ky_jacobians_common_vertex;
    /**
     * Jacobian from the unit cell to the real cell \f$K_y\f$ for each
     * \f$k_3\f$ term and at each quadrature point for the regular case.
     */
    Table<2, double> ky_jacobians_regular;

    /**
     * Normal vector at each quadrature point in the real cell \f$K_y\f$ for
     * the same panel case.
     */
    Table<2, Tensor<1, 3>> ky_normals_same_panel;
    /**
     * Normal vector at each quadrature point in the real cell \f$K_y\f$ for
     * the common edge case.
     */
    Table<2, Tensor<1, 3>> ky_normals_common_edge;
    /**
     * Normal vector at each quadrature point in the real cell \f$K_y\f$ for
     * the common vertex case.
     */
    Table<2, Tensor<1, 3>> ky_normals_common_vertex;
    /**
     * Normal vector at each quadrature point in the real cell \f$K_y\f$ for
     * the regular case.
     */
    Table<2, Tensor<1, 3>> ky_normals_regular;

    /**
     * Coordinates in the real cell \f$K_y\f$ for each \f$k_3\f$ term and each
     * quadrature point for the same panel case.
     */
    Table<2, Point<3>> ky_quad_points_same_panel;
    /**
     * Coordinates in the real cell \f$K_y\f$ for each \f$k_3\f$ term and each
     * quadrature point for the common edge case.
     */
    Table<2, Point<3>> ky_quad_points_common_edge;
    /**
     * Coordinates in the real cell \f$K_y\f$ for each \f$k_3\f$ term and each
     * quadrature point for the common vertex case.
     */
    Table<2, Point<3>> ky_quad_points_common_vertex;
    /**
     * Coordinates in the real cell \f$K_y\f$ for each \f$k_3\f$ term and each
     * quadrature point for the regular case.
     */
    Table<2, Point<3>> ky_quad_points_regular;

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

      // Polynomial space inverse numbering for recovering the lexicographic
      // order.
      kx_fe_poly_space_numbering_inverse =
        FETools::lexicographic_to_hierarchic_numbering(kx_fe);
      ky_fe_poly_space_numbering_inverse =
        FETools::lexicographic_to_hierarchic_numbering(ky_fe);
      generate_backward_dof_permutation(
        ky_fe, 0, ky_fe_reversed_poly_space_numbering_inverse);
    }
  };


  struct PairCellWisePerTaskData
  {
    /**
     * Local matrix for the pair of cells for the DLP kernel.
     */
    FullMatrix<double> dlp_matrix;
    /**
     * Local matrix for the pair of cells for the SLP kernel.
     */
    FullMatrix<double> slp_matrix;
    /**
     * Permuted list of DoF indices in the cell \f$K_x\f$, each element of
     * which is associated with the corresponding element in
     * @p PairCellWiseScratchData::kx_support_points_permuted.
     */
    std::vector<types::global_dof_index> kx_local_dof_indices_permuted;
    /**
     * Permuted list of DoF indices in the cell \f$K_y\f$, each element of
     * which is associated with the corresponding element in
     * @p PairCellWiseScratchData::ky_support_points_permuted.
     */
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


    /**
     * Base class for a BEM Laplace kernel function.
     *
     * \mynote{The template argument @p dim is the spatial dimension instead of the manifold dimension.}
     */
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

      /**
       * Assignment operator
       *
       * @param f
       * @return
       */
      KernelFunction &
      operator=(const KernelFunction &f);


      /**
       * Evaluate the kernel function at the specified point pair \f$(x, y)\f$
       * associated with their normal vectors \f$(n_x, n_y)\f$. In case the
       * kernel function is vector-valued, this function only returns the
       * required @p component in the result vector.
       *
       * @param x
       * @param y
       * @param nx
       * @param ny
       * @param component Component index in the result vector, if the kernel
       * function is vector-valued. If the kernel function is scalar-valued,
       * there is no such issue and @p component is 0 by default.
       * @return
       */
      virtual RangeNumberType
      value(const Point<dim> &    x,
            const Point<dim> &    y,
            const Tensor<1, dim> &nx,
            const Tensor<1, dim> &ny,
            const unsigned int    component = 0) const;


      /**
       * Define the default behavior for evaluation of @p KernelFunction::value
       * at the specified point pair \f$(x, y)\f$ associated with their normal
       * vectors \f$(n_x, n_y)\f$ and all of its components will be retrieved
       * into the result vector @p values.
       *
       * \mynote{Even though @p KernelFunction is an abstract class which cannot
       * be instantiated and its member functions will never be called, the
       * definition of this function will reduce the burden of redefining such
       * function in each child class.}
       *
       * @param x
       * @param y
       * @param nx
       * @param ny
       * @param values The vector holding all components of the function result.
       */
      virtual void
      vector_value(const Point<dim> &       x,
                   const Point<dim> &       y,
                   const Tensor<1, dim> &   nx,
                   const Tensor<1, dim> &   ny,
                   Vector<RangeNumberType> &values) const;


      /**
       * Define the default behavior for evaluation of @p KernelFunction::value
       * at a list of points with associated normal vectors \f$(n_x, n_y)\f$.
       * Only the required @p component in the result vector will be returned,
       * if the kernel function is vector-valued.
       *
       * \mynote{Even though @p KernelFunction is an abstract class which cannot
       * be instantiated and its member functions will never be called, the
       * definition of this function will reduce the burden of redefining such
       * function in each child class.}
       *
       * @param x_points
       * @param y_points
       * @param nx_list
       * @param ny_list
       * @param values
       * @param component
       */
      virtual void
      value_list(const std::vector<Point<dim>> &    x_points,
                 const std::vector<Point<dim>> &    y_points,
                 const std::vector<Tensor<1, dim>> &nx_list,
                 const std::vector<Tensor<1, dim>> &ny_list,
                 std::vector<RangeNumberType> &     values,
                 const unsigned int                 component = 0) const;


      /**
       * Define the default behavior for evaluation of
       * @p KernelFunction::vector_value at a list of points with associated
       * normal vectors \f$(n_x, n_y)\f$.
       *
       * \mynote{Even though @p KernelFunction is an abstract class which cannot
       * be instantiated and its member functions will never be called, the
       * definition of this function will reduce the burden of redefining such
       * function in each child class.}
       *
       * @param x_points
       * @param y_points
       * @param nx_list
       * @param ny_list
       * @param values
       */
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


    /**
     * Default destructor provided by the compiler.
     */
    template <int dim, typename RangeNumberType>
    inline KernelFunction<dim, RangeNumberType>::~KernelFunction() = default;


    template <int dim, typename RangeNumberType>
    KernelFunction<dim, RangeNumberType> &
    KernelFunction<dim, RangeNumberType>::operator=(const KernelFunction &f)
    {
      /**
       * As a pure base class, it does nothing here but only assert the number
       * of components. The following sentence suppresses compiler warnings
       * about the unused input argument f.
       */
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
      /**
       * This function should never be called, since as a member function of the
       * abstract class, it has no literal definition of the function. Hence, we
       * throw an error here.
       */
      Assert(false, ExcPureFunctionCalled());

      return 0;
    }


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
     * Laplace single layer kernel function
     */
    template <int dim, typename RangeNumberType = double>
    class SingleLayerKernel : public KernelFunction<dim, RangeNumberType>
    {
    public:
      SingleLayerKernel()
        : KernelFunction<dim, RangeNumberType>(SingleLayer)
      {}

      /**
       * Evaluate the kernel function.
       *
       * \mynote{With the appended keyword @p override, this function must be
       * explicitly defined.}
       *
       * @param x
       * @param y
       * @param nx
       * @param ny
       * @param component
       * @return
       */
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
      const Point<dim> &    x,
      const Point<dim> &    y,
      const Tensor<1, dim> &nx,
      const Tensor<1, dim> &ny,
      const unsigned int    component) const
    {
      (void)nx;
      (void)ny;
      (void)component;

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
      const Point<dim> &    x,
      const Point<dim> &    y,
      const Tensor<1, dim> &nx,
      const Tensor<1, dim> &ny,
      const unsigned int    component) const
    {
      (void)nx;
      (void)component;

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
      const Tensor<1, dim> &ny,
      const unsigned int    component) const
    {
      (void)ny;
      (void)component;

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
      const unsigned int    component) const
    {
      (void)component;

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
   * Kernel function pulled back to the unit cell.
   *
   * \mynote{The unit cell has the manifold dimension @p dim.}
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  class KernelPulledbackToUnitCell : public Subscriptor
  {
  public:
    /**
     * Manifold dimension
     */
    static const unsigned int dimension = dim;
    const unsigned int        n_components;


    /**
     * Constructor without @p BEMValues precalculation.
     *
     * @param kernel_function Kernel function defined in the original space,
     * which has dimension @p spacedim.
     * @param cell_neighboring_type
     * @param kx_support_points Permuted list of support points in \f$K_x\f$ in
     * the lexicographic order.
     * @param ky_support_points Permuted list of support points in \f$K_y\f$ in
     * the lexicographic order or reversed lexicographic order, which depends on
     * the cell neighboring type.
     * @param kx_fe
     * @param ky_fe
     * @param kx_dof_index Index for accessing the list of DoFs in the
     * lexicographic order in \f$K_x\f$.
     * @param ky_dof_index Index for accessing the list of DoFs in the
     * lexicographic order in \f$K_y\f$.
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
     * Constructor with @p BEMValues precalculation.
     *
     * @param kernel_function Kernel function defined in the original space,
     * which has dimension @p spacedim.
     * @param cell_neighboring_type
     * @param kx_support_points Permuted list of support points in \f$K_x\f$ in
     * the lexicographic order.
     * @param ky_support_points Permuted list of support points in \f$K_y\f$ in
     * the lexicographic order or reversed lexicographic order, which depends on
     * the cell neighboring type.
     * @param kx_fe
     * @param ky_fe
     * @param bem_values Pointer to the precalculated @p BEMValues, which
     * contains shape function values and their gradient values on the
     * quadrature points.
     * @param kx_dof_index Index for accessing the list of DoFs in the
     * lexicographic order in \f$K_x\f$.
     * @param ky_dof_index Index for accessing the list of DoFs in the
     * lexicographic order in \f$K_y\f$.
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
     * Constructor with @p BEMValues precalculation, which is also oriented for
     * parallelization, i.e., it accepts the scratch data as one of its
     * arguments.
     *
     * @param kernel_function Kernel function defined in the original space,
     * which has dimension @p spacedim.
     * @param cell_neighboring_type
     * @param kx_support_points Permuted list of support points in \f$K_x\f$ in
     * the lexicographic order.
     * @param ky_support_points Permuted list of support points in \f$K_y\f$ in
     * the lexicographic order or reversed lexicographic order, which depends on
     * the cell neighboring type.
     * @param kx_fe
     * @param ky_fe
     * @param bem_values Pointer to the precalculated @p BEMValues, which
     * contains shape function values and their gradient values on the
     * quadrature points.
     * @param scratch
     * @param kx_dof_index Index for accessing the list of DoFs in the
     * lexicographic order in \f$K_x\f$.
     * @param ky_dof_index Index for accessing the list of DoFs in the
     * lexicographic order in \f$K_y\f$.
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


    /**
     * Destructor.
     *
     * \mynote{Since this class has only references to other objects as its
     * members and their memory is not managed by this class, the destructor
     * provided by the compiler is adopted.}
     */
    virtual ~KernelPulledbackToUnitCell();


    KernelPulledbackToUnitCell &
    operator=(const KernelPulledbackToUnitCell &f);


    /**
     * Associate the @p KernelPulledbackToUnitCell with support points and
     * finite element data for a new pair of cells.
     *
     * @param cell_neighboring_type
     * @param kx_support_points Permuted list of support points in \f$K_x\f$ in
     * the lexicographic order.
     * @param ky_support_points Permuted list of support points in \f$K_y\f$ in
     * the lexicographic order or reversed lexicographic order, which depends on
     * the cell neighboring type.
     * @param kx_fe
     * @param ky_fe
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
     * Evaluation of the kernel function at the specified coordinates in the
     * unit cell.
     *
     * \mynote{This version of @p value does not rely on the precalculated
     * @p BEMValues, hence it will calculate Jacobian and normal vector on the
     * fly.}
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

    /**
     * Permuted list of support points in \f$K_x\f$ in the lexicographic order.
     */
    const std::vector<Point<spacedim>> &kx_support_points;
    /**
     * Permuted list of support points in \f$K_y\f$ in the lexicographic order
     * or reversed lexicographic order, which depends on the cell neighboring
     * type.
     */
    const std::vector<Point<spacedim>> &ky_support_points;

    const FiniteElement<dim, spacedim> &kx_fe;
    const FiniteElement<dim, spacedim> &ky_fe;

    const BEMValues<dim, spacedim, RangeNumberType> *bem_values;
    const PairCellWiseScratchData *                  scratch;

    /**
     * The current index for accessing the list of DoFs in the lexicographic
     * order in \f$K_x\f$.
     */
    unsigned int kx_dof_index;
    /**
     * The current index for accessing the list of DoFs in the lexicographic
     * order or reversed lexicographic order in \f$K_y\f$.
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

    /**
     * When the kernel type is Laplace double layer or hyper singular, the
     * kernel function depends on the normal vector \f$K_y\f$. It should be
     * emphasized that when the cell neighboring type for \f$K_x\f$ and
     * \f$K_y\f$ is common edge, because the real support points for \f$K_y\f$
     * are permuted into the reversed lexicographic order, the direction of the
     * calculated \f$n_y\f$ is opposite to the real value. Hence, in such
     * circumstance, \f$n_y\$ should be negated.
     *
     * \mynote{It should be emphasized that what have been actually permuted are
     * the real support points @p kx_support_points and @p ky_support_points,
     * instead of the shape functions on the unit cell.}
     */
    Point<spacedim> x =
      transform_unit_to_permuted_real_cell(kx_fe, kx_support_points, x_hat);
    Point<spacedim> y =
      transform_unit_to_permuted_real_cell(ky_fe, ky_support_points, y_hat);
    RangeNumberType     Jx = 0;
    RangeNumberType     Jy = 0;
    Tensor<1, spacedim> nx, ny;

    /**
     * Calculate surface metric determinants and the normal vectors.
     */
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

          // Negate the normal vector in \f$K_y\f$.
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

          // Negate the normal vector in \f$K_y\f$.
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

    std::vector<unsigned int> kx_poly_space_inverse_numbering =
      FETools::lexicographic_to_hierarchic_numbering(kx_fe);
    std::vector<unsigned int> ky_poly_space_inverse_numbering =
      FETools::lexicographic_to_hierarchic_numbering(ky_fe);

    //        // DEBUG
    //        deallog
    //          << kernel_function.value(x, y, nx, ny, component) << ","
    //          << Jx << ","
    //          << Jy << ","
    //          <<
    //          kx_fe.shape_value(kx_poly_space_inverse_numbering[kx_dof_index],
    //                               x_hat)
    //          << ","
    //          <<
    //          ky_fe.shape_value(ky_poly_space_inverse_numbering[ky_dof_index],
    //                               y_hat)
    //          << std::endl;

    /**
     * Evaluate the original kernel function at the specified pair of points in
     * the real cells with their normal vectors, the result of which is then
     * multiplied by the Jacobians and shape function values.
     */
    return kernel_function.value(x, y, nx, ny, component) * Jx * Jy *
           kx_fe.shape_value(kx_poly_space_inverse_numbering[kx_dof_index],
                             x_hat) *
           ky_fe.shape_value(ky_poly_space_inverse_numbering[ky_dof_index],
                             y_hat);
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

    // Select shape function value table according to the cell neighboring
    // type.
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

    // Negate the normal vector in \f$K_y\f$.
    if (cell_neighboring_type == CommonEdge)
      {
        ny = -ny;
      }

    /**
     * Evaluate the original kernel function at the specified pair of points in
     * the real cells with their normal vectors, the result of which is then
     * multiplied by the Jacobians and shape function values.
     */
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
   * Class for pullback the kernel function on the product of two unit cells
   * to Sauter's parametric space.
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  class KernelPulledbackToSauterSpace : public Subscriptor
  {
  public:
    const unsigned int n_components;

    /**
     * Constructor without @p BEMValues.
     *
     * @param kernel
     * @param cell_neighboring_type
     */
    KernelPulledbackToSauterSpace(
      const KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType> &kernel,
      const CellNeighboringType cell_neighboring_type);


    /**
     * Constructor with @p BEMValues.
     *
     * @param kernel
     * @param cell_neighboring_type
     * @param bem_values
     */
    KernelPulledbackToSauterSpace(
      const KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType> &kernel,
      const CellNeighboringType                        cell_neighboring_type,
      const BEMValues<dim, spacedim, RangeNumberType> *bem_values);


    /**
     * Destructor.
     *
     * \mynote{Since this class has only references to other objects as its
     * members and their memory is not managed by this class, the destructor
     * provided by the compiler is adopted.}
     */
    ~KernelPulledbackToSauterSpace();


    KernelPulledbackToSauterSpace &
    operator=(const KernelPulledbackToSauterSpace &f);


    /**
     * Evaluate the pullback of kernel function on Sauter's parametric space.
     *
     * @param p The coordinates at which the kernel function is to be evaluated.
     * It should be noted that this point has a dimension of @p dim*2.
     */
    RangeNumberType
    value(const Point<dim * 2> p, const unsigned int component = 0) const;


    /**
     * Evaluate the pullback of kernel function on Sauter's parametric space
     * at the quad_no'th quadrature point under the given 4D quadrature rule.
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
            /**
             * Jacobian from the parametric coordinates to the unit coordinates.
             */
            double jacobian_det = p(0) * (1 - p(0)) * (1 - p(0) * p(1));
            /**
             * Transform the parametric coordinates to the unit coordinates.
             */
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
            /**
             * There is no coordinate transformation for the regular case, so
             * directly evaluate the kernel function on the product of unit
             * cells.
             */

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


  /**
   * Apply the Sauter's 4D quadrature rule to the kernel function pulled back to
   * the Sauter's parametric space.
   *
   * @param quad_rule
   * @param f
   * @param component
   * @return
   */
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


  /**
   * Apply the Sauter's 4D quadrature rule to the kernel function pulled back to
   * the Sauter's parametric space. This version uses the precalculated
   * @p BEMValues.
   *
   * @param quad_rule
   * @param f
   * @param component
   * @return
   */
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
   * Get the DoF indices associated with the cell vertices or corners from a
   * list of DoF indices which have been arranged in either the forward or
   * backward lexicographic order. The results are returned in an array as the
   * function's return value.
   *
   * \mynote{There are <code>GeometryInfo<dim>::vertices_per_cell</code>
   * vertices in the returned array, among which the last two vertex DoF indices
   * have been swapped in this function so that the whole list of vertex DoF
   * indices in the returned array are arranged in either the clockwise or
   * counter clockwise order instead of the original zigzag order.}
   *
   * @param fe
   * @param dof_indices List of DoF indices in either the forward or backward
   * lexicographic order.
   * @return List of DoF indices for the cell vertices or corners with the last
   * two swapped.
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
   * Get the DoF indices associated with the cell vertices or corners from a
   * list of DoF indices which have been arranged in either the forward or
   * backward lexicographic order. The results are returned in an array as the
   * last argument of this function.
   *
   * \mynote{There are <code>GeometryInfo<dim>::vertices_per_cell</code>
   * vertices in the returned array, among which the last two vertex DoF indices
   * have been swapped in this function so that the whole list of vertex DoF
   * indices in the returned array are arranged in either the clockwise or
   * counter clockwise order instead of the original zigzag order.}
   *
   * @param fe
   * @param dof_indices List of DoF indices in either the forward or backward
   * lexicographic order.
   * @param vertex_dof_indices [out] List of DoF indices for the cell vertices
   * or corners with the last two swapped.
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
   * Get the DoF index for the starting vertex in the standard configuration of
   * the cell pair aimed for the Galerkin BEM double integration using the
   * Sauter's method.
   *
   * \mynote{There are two cases to be processed here, common edge and common
   * vertex.
   * 1. In the common edge case, there are two DoF indices in the vector
   * <code>vertex_dof_index_intersection</code>. Then their array indices wrt.
   * the vector <code>local_vertex_dof_indices_swapped</code> will be
   * searched. By considering this vector as a closed loop list, the two DoF
   * indices in this vector are successively located and the first one of which
   * is the vertex to start subsequent DoF traversing.
   * 2. In the common vertex case, since there is only one DoF index in the
   * vector @p vertex_dof_index_intersection, this vertex is the starting point.}
   *
   * @param vertex_dof_index_intersection The vector storing the intersection
   * of vertex DoF indices for \f$K_x\f$ and \f$K_y\f$.
   * @param local_vertex_dof_indices_swapped Vertex DoF indices with the last
   * two swapped, which have been obtained from the function
   * @p get_vertex_dof_indices_swapped.
   * @return The array index for the starting vertex, wrt. the original list of
   * vertex DoF indices, i.e. the last two elements of which are not swapped.
   */
  template <int vertices_per_cell>
  unsigned int
  get_start_vertex_dof_index(
    const std::vector<types::global_dof_index> &vertex_dof_index_intersection,
    const std::array<types::global_dof_index, vertices_per_cell>
      &local_vertex_dof_indices_swapped)
  {
    unsigned int starting_vertex_index = 9999;

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

    /**
     * Because the last two elements in the original list of vertex DoF indices
     * have been swapped, we need to correct the starting vertex index when it
     * is one of the last two elements.
     */
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
   * Perform Sauter's quadrature rule on a pair of quadrangular cells, which
   * handles various cases including same panel, common edge, common vertex and
   * regular cell neighboring types. This functions returns the computed local
   * matrix without assembling it to the global matrix.
   *
   * \mynote{In this version, shape function values and their gradient values
   * are not precalculated.}
   *
   * @param kernel_function Laplace kernel function.
   * @param kx_cell_iter Iterator pointing to \f$K_x\f$.
   * @param kx_cell_iter Iterator pointing to \f$K_y\f$.
   * @param kx_mapping Mapping used for \f$K_x\f$. Because a mesher usually generates
   * 1st order grid, if there is no additional manifold specification, the
   * mapping should be 1st order.
   * @param kx_mapping Mapping used for \f$K_y\f$. Because a mesher usually generates
   * 1st order grid, if there is no additional manifold specification, the
   * mapping should be 1st order.
   * @return Local matrix
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

    // Support points of \f$K_x\f$ and \f$K_y\f$ in the default
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

    // Local matrix
    FullMatrix<RangeNumberType> cell_matrix(kx_n_dofs, ky_n_dofs);

    // Polynomial space inverse numbering for recovering the lexicographic
    // order.
    std::vector<unsigned int> kx_fe_poly_space_numbering_inverse =
      FETools::lexicographic_to_hierarchic_numbering(kx_fe);
    std::vector<unsigned int> ky_fe_poly_space_numbering_inverse =
      FETools::lexicographic_to_hierarchic_numbering(ky_fe);

    switch (cell_neighboring_type)
      {
        case SamePanel:
          {
            Assert(vertex_dof_index_intersection.size() == vertices_per_cell,
                   ExcInternalError());

            // Get support points in lexicographic order.
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


            // Iterate over DoFs for test function space in lexicographic
            // order in \f$K_x\f$.
            for (unsigned int i = 0; i < kx_n_dofs; i++)
              {
                // Iterate over DoFs for ansatz function space in tensor
                // product order in \f$K_y\f$.
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
            // 1. Get the DoF indices in lexicographic order for \f$K_x\f$.
            // 2. Get the DoF indices in reversed lexicographic order for
            // \f$K_x\f$.
            // 3. Extract DoF indices only for cell vertices in \f$K_x\f$ and
            // \f$K_y\f$. N.B. The DoF indices for the last two vertices are
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
                get_vertex_dof_indices_swapped(kx_fe,
                                               kx_local_dof_indices_permuted);
            std::array<types::global_dof_index, vertices_per_cell>
              ky_local_vertex_dof_indices_swapped =
                get_vertex_dof_indices_swapped(ky_fe,
                                               ky_local_dof_indices_permuted);

            // Determine the starting vertex index in \f$K_x\f$ and \f$K_y\f$.
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

            // Generate the permutation of DoFs in \f$K_x\f$ and \f$K_y\f$ by
            // starting from <code>kx_starting_vertex_index</code> or
            // <code>ky_starting_vertex_index</code>.
            std::vector<unsigned int> kx_local_dof_permutation =
              generate_forward_dof_permutation(kx_fe, kx_starting_vertex_index);
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


            // Iterate over DoFs for test function space in lexicographic
            // order in \f$K_x\f$.
            for (unsigned int i = 0; i < kx_n_dofs; i++)
              {
                // Iterate over DoFs for ansatz function space in tensor
                // product order in \f$K_y\f$.
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
                get_vertex_dof_indices_swapped(kx_fe,
                                               kx_local_dof_indices_permuted);
            std::array<types::global_dof_index, vertices_per_cell>
              ky_local_vertex_dof_indices_swapped =
                get_vertex_dof_indices_swapped(ky_fe,
                                               ky_local_dof_indices_permuted);

            // Determine the starting vertex index in \f$K_x\f$ and \f$K_y\f$.
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

            // Generate the permutation of DoFs in \f$K_x\f$ and \f$K_y\f$ by
            // starting from <code>kx_starting_vertex_index</code> or
            // <code>ky_starting_vertex_index</code>.
            std::vector<unsigned int> kx_local_dof_permutation =
              generate_forward_dof_permutation(kx_fe, kx_starting_vertex_index);
            std::vector<unsigned int> ky_local_dof_permutation =
              generate_forward_dof_permutation(ky_fe, ky_starting_vertex_index);

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


            // Iterate over DoFs for test function space in lexicographic
            // order in \f$K_x\f$.
            for (unsigned int i = 0; i < kx_n_dofs; i++)
              {
                // Iterate over DoFs for ansatz function space in tensor
                // product order in \f$K_y\f$.
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


            // Iterate over DoFs for test function space in lexicographic
            // order in \f$K_x\f$.
            for (unsigned int i = 0; i < kx_n_dofs; i++)
              {
                // Iterate over DoFs for ansatz function space in tensor
                // product order in \f$K_y\f$.
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

    // Support points of \f$K_x\f$ and \f$K_y\f$ in the default
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

    // Polynomial space inverse numbering for recovering the tensor
    // product ordering.
    std::vector<unsigned int> kx_fe_poly_space_numbering_inverse =
      FETools::lexicographic_to_hierarchic_numbering(kx_fe);
    std::vector<unsigned int> ky_fe_poly_space_numbering_inverse =
      FETools::lexicographic_to_hierarchic_numbering(ky_fe);

    // Quadrature rule to be adopted depending on the cell neighboring
    // type.
    const QGauss<4> *active_quad_rule;

    switch (cell_neighboring_type)
      {
        case SamePanel:
          {
            Assert(vertex_dof_index_intersection.size() == vertices_per_cell,
                   ExcInternalError());

            // Get support points in lexicographic order.
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
            // 1. Get the DoF indices in lexicographic order for \f$K_x\f$.
            // 2. Get the DoF indices in reversed lexicographic order for
            // \f$K_x\f$.
            // 3. Extract DoF indices only for cell vertices in \f$K_x\f$ and
            // \f$K_y\f$. N.B. The DoF indices for the last two vertices are
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
                get_vertex_dof_indices_swapped(kx_fe,
                                               kx_local_dof_indices_permuted);
            std::array<types::global_dof_index, vertices_per_cell>
              ky_local_vertex_dof_indices_swapped =
                get_vertex_dof_indices_swapped(ky_fe,
                                               ky_local_dof_indices_permuted);

            // Determine the starting vertex index in \f$K_x\f$ and \f$K_y\f$.
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

            // Generate the permutation of DoFs in \f$K_x\f$ and \f$K_y\f$ by
            // starting from <code>kx_starting_vertex_index</code> or
            // <code>ky_starting_vertex_index</code>.
            std::vector<unsigned int> kx_local_dof_permutation =
              generate_forward_dof_permutation(kx_fe, kx_starting_vertex_index);
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
                get_vertex_dof_indices_swapped(kx_fe,
                                               kx_local_dof_indices_permuted);
            std::array<types::global_dof_index, vertices_per_cell>
              ky_local_vertex_dof_indices_swapped =
                get_vertex_dof_indices_swapped(ky_fe,
                                               ky_local_dof_indices_permuted);

            // Determine the starting vertex index in \f$K_x\f$ and \f$K_y\f$.
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

            // Generate the permutation of DoFs in \f$K_x\f$ and \f$K_y\f$ by
            // starting from <code>kx_starting_vertex_index</code> or
            // <code>ky_starting_vertex_index</code>.
            std::vector<unsigned int> kx_local_dof_permutation =
              generate_forward_dof_permutation(kx_fe, kx_starting_vertex_index);
            std::vector<unsigned int> ky_local_dof_permutation =
              generate_forward_dof_permutation(ky_fe, ky_starting_vertex_index);

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
          }
      }

    // Iterate over DoFs for test function space in lexicographic
    // order in \f$K_x\f$.
    for (unsigned int i = 0; i < kx_n_dofs; i++)
      {
        // Iterate over DoFs for ansatz function space in tensor
        // product order in \f$K_y\f$.
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

    return cell_matrix;
  }


  /**
   * Perform Sauter's quadrature rule on a pair of quadrangular cells, which
   * handles various cases including same panel, common edge, common vertex and
   * regular cell neighboring types. The computed local matrix values will be
   * assembled into the system matrix, which is passed as the first argument.
   *
   * \mynote{In this version, shape function values and their gradient values
   * are not precalculated.}
   *
   * @param system_matrix The global full matrix to which the local matrix will
   * be assembled.
   * @param kernel_function Laplace kernel function.
   * @param kx_cell_iter Iterator pointing to \f$K_x\f$.
   * @param kx_cell_iter Iterator pointing to \f$K_y\f$.
   * @param kx_mapping Mapping used for \f$K_x\f$. Because a mesher usually generates
   * 1st order grid, if there is no additional manifold specification, the
   * mapping should be 1st order.
   * @param ky_mapping Mapping used for \f$K_y\f$. Because a mesher usually generates
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

    // Support points of \f$K_x\f$ and \f$K_y\f$ in the default
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

    // Local matrix
    FullMatrix<RangeNumberType> cell_matrix(kx_n_dofs, ky_n_dofs);

    // Polynomial space inverse numbering for recovering the lexicographic
    // order.
    std::vector<unsigned int> kx_fe_poly_space_numbering_inverse =
      FETools::lexicographic_to_hierarchic_numbering(kx_fe);
    std::vector<unsigned int> ky_fe_poly_space_numbering_inverse =
      FETools::lexicographic_to_hierarchic_numbering(ky_fe);

    switch (cell_neighboring_type)
      {
        case SamePanel:
          {
            Assert(vertex_dof_index_intersection.size() == vertices_per_cell,
                   ExcInternalError());

            // Get support points in lexicographic order.
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


            // Iterate over DoFs for test function space in lexicographic
            // order in \f$K_x\f$.
            for (unsigned int i = 0; i < kx_n_dofs; i++)
              {
                // Iterate over DoFs for ansatz function space in tensor
                // product order in \f$K_y\f$.
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
            // 1. Get the DoF indices in lexicographic order for \f$K_x\f$.
            // 2. Get the DoF indices in reversed lexicographic order for
            // \f$K_x\f$.
            // 3. Extract DoF indices only for cell vertices in \f$K_x\f$ and
            // \f$K_y\f$. N.B. The DoF indices for the last two vertices are
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
                get_vertex_dof_indices_swapped(kx_fe,
                                               kx_local_dof_indices_permuted);
            std::array<types::global_dof_index, vertices_per_cell>
              ky_local_vertex_dof_indices_swapped =
                get_vertex_dof_indices_swapped(ky_fe,
                                               ky_local_dof_indices_permuted);

            // Determine the starting vertex index in \f$K_x\f$ and \f$K_y\f$.
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

            // Generate the permutation of DoFs in \f$K_x\f$ and \f$K_y\f$ by
            // starting from <code>kx_starting_vertex_index</code> or
            // <code>ky_starting_vertex_index</code>.
            std::vector<unsigned int> kx_local_dof_permutation =
              generate_forward_dof_permutation(kx_fe, kx_starting_vertex_index);
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


            // Iterate over DoFs for test function space in lexicographic
            // order in \f$K_x\f$.
            for (unsigned int i = 0; i < kx_n_dofs; i++)
              {
                // Iterate over DoFs for ansatz function space in tensor
                // product order in \f$K_y\f$.
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
                get_vertex_dof_indices_swapped(kx_fe,
                                               kx_local_dof_indices_permuted);
            std::array<types::global_dof_index, vertices_per_cell>
              ky_local_vertex_dof_indices_swapped =
                get_vertex_dof_indices_swapped(ky_fe,
                                               ky_local_dof_indices_permuted);

            // Determine the starting vertex index in \f$K_x\f$ and \f$K_y\f$.
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

            // Generate the permutation of DoFs in \f$K_x\f$ and \f$K_y\f$ by
            // starting from <code>kx_starting_vertex_index</code> or
            // <code>ky_starting_vertex_index</code>.
            std::vector<unsigned int> kx_local_dof_permutation =
              generate_forward_dof_permutation(kx_fe, kx_starting_vertex_index);
            std::vector<unsigned int> ky_local_dof_permutation =
              generate_forward_dof_permutation(ky_fe, ky_starting_vertex_index);

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


            // Iterate over DoFs for test function space in lexicographic
            // order in \f$K_x\f$.
            for (unsigned int i = 0; i < kx_n_dofs; i++)
              {
                // Iterate over DoFs for ansatz function space in tensor
                // product order in \f$K_y\f$.
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


            // Iterate over DoFs for test function space in lexicographic
            // order in \f$K_x\f$.
            for (unsigned int i = 0; i < kx_n_dofs; i++)
              {
                // Iterate over DoFs for ansatz function space in tensor
                // product order in \f$K_y\f$.
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
   * @param bem_values
   * @param kx_cell_iter Iterator pointing to \f$K_x\f$.
   * @param kx_cell_iter Iterator pointing to \f$K_y\f$.
   * @param kx_mapping Mapping used for \f$K_x\f$. Because a mesher usually generates
   * 1st order grid, if there is no additional manifold specification, the
   * mapping should be 1st order.
   * @param ky_mapping Mapping used for \f$K_y\f$. Because a mesher usually generates
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

    // Support points of \f$K_x\f$ and \f$K_y\f$ in the default
    // hierarchical order.
    std::vector<Point<spacedim>> kx_support_points_hierarchical =
      get_hierarchic_support_points_in_real_cell(kx_cell_iter,
                                                 kx_fe,
                                                 kx_mapping);
    std::vector<Point<spacedim>> ky_support_points_hierarchical =
      get_hierarchic_support_points_in_real_cell(ky_cell_iter,
                                                 ky_fe,
                                                 ky_mapping);

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

    // Polynomial space inverse numbering for recovering the tensor
    // product ordering.
    std::vector<unsigned int> kx_fe_poly_space_numbering_inverse =
      FETools::lexicographic_to_hierarchic_numbering(kx_fe);
    std::vector<unsigned int> ky_fe_poly_space_numbering_inverse =
      FETools::lexicographic_to_hierarchic_numbering(ky_fe);

    // Quadrature rule to be adopted depending on the cell neighboring
    // type.
    const QGauss<4> *active_quad_rule = nullptr;

    switch (cell_neighboring_type)
      {
        case SamePanel:
          {
            Assert(vertex_dof_index_intersection.size() == vertices_per_cell,
                   ExcInternalError());

            // Get support points in the lexicographic order.
            kx_support_points_permuted =
              permute_vector(kx_support_points_hierarchical,
                             kx_fe_poly_space_numbering_inverse);
            ky_support_points_permuted =
              permute_vector(ky_support_points_hierarchical,
                             ky_fe_poly_space_numbering_inverse);

            // Get permuted local DoF indices in the lexicographic order.
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
            /**
             * This part handles the common edge case of Sauter's quadrature
             * rule.
             *
             * 1. Get the DoF indices in the lexicographic order for \f$K_x\f$.
             * 2. Get the DoF indices in the reversed lexicographic order for
             * \f$K_y\f$. Hence, the orientation determined from these DoF
             * indices is opposite to that of \f$K_x\f$.
             * 3. Extract DoF indices only for cell vertices or corners in
             * \f$K_x\f$ and \f$K_y\f$.
             * \mynote{Because the four cell vertices or corners retrieved from
             * the list of DoFs either in the lexicographic or the reversed
             * lexicographic order are in the zigzag form as shown below,
             * @verbatim
             * 2 ----- 3
             * |  Kx   |
             * 0 ----- 1
             * |  Ky   |
             * 2 ----- 3
             * @endverbatim
             * the DoF indices for the last two vertices should be swapped, such
             * that the four vertices in the cell are either in the clockwise or
             * counter clockwise order, as shown below.
             * @verbatim
             * 3 ----- 2
             * |  Kx   |
             * 0 ----- 1
             * |  Ky   |
             * 3 ----- 2
             * @endverbatim
             * 4. Determine the starting vertex on the common edge.
             *
             * Finally, we should keep in mind that the orientation of the cell
             * \f$K_y\f$ is reversed due to the above operation, so the normal
             * vector \f$n_y\f$ calculated from such permuted support points and
             * shape functions should be negated back to the correct direction
             * during the evaluation of the kernel function.
             */
            Assert(vertex_dof_index_intersection.size() ==
                     GeometryInfo<dim>::vertices_per_face,
                   ExcInternalError());

            /**
             * Get permuted local DoF indices in \f$K_x\f$ in the lexicographic
             * order.
             */
            kx_local_dof_indices_permuted =
              permute_vector(kx_local_dof_indices_hierarchical,
                             kx_fe_poly_space_numbering_inverse);
            /**
             * Get permuted local DoF indices in \f$K_y\f$ in the reversed
             * lexicographic order by starting from the first vertex.
             */
            std::vector<unsigned int>
              ky_fe_reversed_poly_space_numbering_inverse =
                generate_backward_dof_permutation(ky_fe, 0);
            ky_local_dof_indices_permuted =
              permute_vector(ky_local_dof_indices_hierarchical,
                             ky_fe_reversed_poly_space_numbering_inverse);

            /**
             * Get the DoF indices for the vertices or corners of \f$K_x\f$
             * with the last two swapped.
             */
            std::array<types::global_dof_index, vertices_per_cell>
              kx_local_vertex_dof_indices_swapped =
                get_vertex_dof_indices_swapped(kx_fe,
                                               kx_local_dof_indices_permuted);
            /**
             * Get the DoF indices for the vertices or corners of \f$K_y\f$
             * with the last two swapped.
             */
            std::array<types::global_dof_index, vertices_per_cell>
              ky_local_vertex_dof_indices_swapped =
                get_vertex_dof_indices_swapped(ky_fe,
                                               ky_local_dof_indices_permuted);

            /**
             * Determine the starting vertex index wrt. the list of vertex DoF
             * indices in \f$K_x\f$.
             */
            unsigned int kx_starting_vertex_index =
              get_start_vertex_dof_index<vertices_per_cell>(
                vertex_dof_index_intersection,
                kx_local_vertex_dof_indices_swapped);
            Assert(kx_starting_vertex_index < vertices_per_cell,
                   ExcInternalError());
            /**
             * Determine the starting vertex index wrt. the list of vertex DoF
             * indices in \f$K_y\f$.
             */
            unsigned int ky_starting_vertex_index =
              get_start_vertex_dof_index<vertices_per_cell>(
                vertex_dof_index_intersection,
                ky_local_vertex_dof_indices_swapped);
            Assert(ky_starting_vertex_index < vertices_per_cell,
                   ExcInternalError());

            /**
             * Generate the permutation of DoF indices in \f$K_x\f$ by starting from
             * the vertex <code>kx_starting_vertex_index</code> in the
             * lexicographic order, i.e. forward traversal.
             */
            std::vector<unsigned int> kx_local_dof_permutation =
              generate_forward_dof_permutation(kx_fe, kx_starting_vertex_index);

            /**
             * Generate the permutation of DoF indices in \f$K_y\f$ by starting from
             * the vertex <code>ky_starting_vertex_index</code> in the
             * reversed lexicographic order, i.e. backward traversal.
             */
            std::vector<unsigned int> ky_local_dof_permutation =
              generate_backward_dof_permutation(ky_fe,
                                                ky_starting_vertex_index);

            /**
             * Get the list of permuted support points in \f$K_x\f$ according
             * to the permutation @p kx_local_dof_permutation.
             */
            kx_support_points_permuted =
              permute_vector(kx_support_points_hierarchical,
                             kx_local_dof_permutation);

            /**
             * Get the list of permuted support points in \f$K_y\f$ according
             * to the permutation @p ky_local_dof_permutation.
             */
            ky_support_points_permuted =
              permute_vector(ky_support_points_hierarchical,
                             ky_local_dof_permutation);

            /**
             * Get the list of permuted DoF indices in \f$K_x\f$ according to
             * the permutation @p kx_local_dof_permutation.
             */
            kx_local_dof_indices_permuted =
              permute_vector(kx_local_dof_indices_hierarchical,
                             kx_local_dof_permutation);

            /**
             * Get the list of permuted DoF indices in \f$K_y\f$ according to
             * the permutation @p ky_local_dof_permutation.
             */
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
            /**
             * This part handles the common vertex case of Sauter's quadrature
             * rule. This is simpler than that of the common edge case because
             * the orientations of \f$K_x\f$ and \f$K_y\f$ are intact.
             */
            Assert(vertex_dof_index_intersection.size() == 1,
                   ExcInternalError());

            /**
             * Get permuted local DoF indices in \f$K_x\f$ in the lexicographic
             * order.
             */
            kx_local_dof_indices_permuted =
              permute_vector(kx_local_dof_indices_hierarchical,
                             kx_fe_poly_space_numbering_inverse);

            /**
             * Get permuted local DoF indices in \f$K_y\f$ in the lexicographic
             * order.
             */
            ky_local_dof_indices_permuted =
              permute_vector(ky_local_dof_indices_hierarchical,
                             ky_fe_poly_space_numbering_inverse);

            /**
             * Get the DoF indices for the vertices or corners of \f$K_x\f$
             * with the last two swapped.
             */
            std::array<types::global_dof_index, vertices_per_cell>
              kx_local_vertex_dof_indices_swapped =
                get_vertex_dof_indices_swapped(kx_fe,
                                               kx_local_dof_indices_permuted);

            /**
             * Get the DoF indices for the vertices or corners of \f$K_y\f$
             * with the last two swapped.
             */
            std::array<types::global_dof_index, vertices_per_cell>
              ky_local_vertex_dof_indices_swapped =
                get_vertex_dof_indices_swapped(ky_fe,
                                               ky_local_dof_indices_permuted);

            /**
             * Determine the starting vertex index wrt. the list of vertex DoF
             * indices in \f$K_x\f$.
             */
            unsigned int kx_starting_vertex_index =
              get_start_vertex_dof_index<vertices_per_cell>(
                vertex_dof_index_intersection,
                kx_local_vertex_dof_indices_swapped);
            Assert(kx_starting_vertex_index < vertices_per_cell,
                   ExcInternalError());

            /**
             * Determine the starting vertex index wrt. the list of vertex DoF
             * indices in \f$K_y\f$.
             */
            unsigned int ky_starting_vertex_index =
              get_start_vertex_dof_index<vertices_per_cell>(
                vertex_dof_index_intersection,
                ky_local_vertex_dof_indices_swapped);
            Assert(ky_starting_vertex_index < vertices_per_cell,
                   ExcInternalError());

            /**
             * Generate the permutation of DoF indices in \f$K_x\f$ by starting from
             * the vertex <code>kx_starting_vertex_index</code> in the
             * lexicographic order, i.e. forward traversal.
             */
            std::vector<unsigned int> kx_local_dof_permutation =
              generate_forward_dof_permutation(kx_fe, kx_starting_vertex_index);

            /**
             * Generate the permutation of DoF indices in \f$K_y\f$ by starting from
             * the vertex <code>ky_starting_vertex_index</code> in the
             * lexicographic order, i.e. forward traversal.
             */
            std::vector<unsigned int> ky_local_dof_permutation =
              generate_forward_dof_permutation(ky_fe, ky_starting_vertex_index);

            /**
             * Get the list of permuted support points in \f$K_x\f$ according
             * to the permutation @p kx_local_dof_permutation.
             */
            kx_support_points_permuted =
              permute_vector(kx_support_points_hierarchical,
                             kx_local_dof_permutation);

            /**
             * Get the list of permuted support points in \f$K_y\f$ according
             * to the permutation @p ky_local_dof_permutation.
             */
            ky_support_points_permuted =
              permute_vector(ky_support_points_hierarchical,
                             ky_local_dof_permutation);

            /**
             * Get the list of permuted DoF indices in \f$K_x\f$ according to
             * the permutation @p kx_local_dof_permutation.
             */
            kx_local_dof_indices_permuted =
              permute_vector(kx_local_dof_indices_hierarchical,
                             kx_local_dof_permutation);

            /**
             * Get the list of permuted DoF indices in \f$K_y\f$ according to
             * the permutation @p ky_local_dof_permutation.
             */
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
            /**
             * This part handles the regular case of Sauter's quadrature rule.
             */
            Assert(vertex_dof_index_intersection.size() == 0,
                   ExcInternalError());

            /**
             * Get permuted local DoF indices in \f$K_x\f$ in the lexicographic
             * order.
             */
            kx_local_dof_indices_permuted =
              permute_vector(kx_local_dof_indices_hierarchical,
                             kx_fe_poly_space_numbering_inverse);

            /**
             * Get permuted local DoF indices in \f$K_y\f$ in the lexicographic
             * order.
             */
            ky_local_dof_indices_permuted =
              permute_vector(ky_local_dof_indices_hierarchical,
                             ky_fe_poly_space_numbering_inverse);

            /**
             * Get the list of permuted support points in \f$K_x\f$ in the
             * lexicographic order.
             */
            kx_support_points_permuted =
              permute_vector(kx_support_points_hierarchical,
                             kx_fe_poly_space_numbering_inverse);

            /**
             * Get the list of permuted support points in \f$K_y\f$ in the
             * lexicographic order.
             */
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

    /**
     * Iterate over DoFs for test function space in \f$K_x\f$.
     */
    for (unsigned int i = 0; i < kx_n_dofs; i++)
      {
        /**
         * Iterate over DoFs for ansatz function space in \f$K_y\f$.
         */
        for (unsigned int j = 0; j < ky_n_dofs; j++)
          {
            /**
             * Pullback the kernel function to unit cell.
             */
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

            /**
             * Pullback the kernel function to Sauter parameter space.
             */
            KernelPulledbackToSauterSpace<dim, spacedim, RangeNumberType>
              kernel_pullback_on_sauter(kernel_pullback_on_unit,
                                        cell_neighboring_type,
                                        &bem_values);

            /**
             * Apply 4d Sauter numerical quadrature.
             */
            cell_matrix(i, j) =
              ApplyQuadratureUsingBEMValues(*active_quad_rule,
                                            kernel_pullback_on_sauter);
          }
      }

    /**
     * Assemble the cell matrix to system matrix.
     */
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
