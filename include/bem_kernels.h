/**
 * @file bem_kernels.h
 * @brief Introduction of bem_kernels.h
 *
 * @date 2022-03-04
 * @author Jihuan Tian
 */
#ifndef INCLUDE_BEM_KERNELS_H_
#define INCLUDE_BEM_KERNELS_H_

#include <deal.II/base/point.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/base/table.h>
#include <deal.II/base/table_indices.h>
#include <deal.II/base/utilities.h>

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
#include "generic_functors.h"

namespace IdeoBEM
{
  using namespace dealii;
  using namespace BEMTools;

  enum KernelType
  {
    SingleLayer,
    DoubleLayer,
    AdjointDoubleLayer,
    HyperSingular,
    HyperSingularRegular,
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

    virtual bool
    is_symmetric() const = 0;
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
      const KernelFunction<spacedim, RangeNumberType> &kernel_function,
      const CellNeighboringType &                      cell_neighboring_type,
      const BEMValues<dim, spacedim, RangeNumberType> *bem_values,
      const PairCellWiseScratchData<dim, spacedim, RangeNumberType> *scratch,
      const unsigned int kx_dof_index = 0,
      const unsigned int ky_dof_index = 0);

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

    void
    set_kx_dof_index(const unsigned int kx_dof_index);
    void
    set_ky_dof_index(const unsigned int ky_dof_index);

    virtual RangeNumberType
    value(const unsigned int k3_index,
          const unsigned int quad_no,
          const unsigned int component = 0) const;

  private:
    const KernelFunction<spacedim, RangeNumberType> &kernel_function;

    CellNeighboringType cell_neighboring_type;

    const BEMValues<dim, spacedim, RangeNumberType> *              bem_values;
    const PairCellWiseScratchData<dim, spacedim, RangeNumberType> *scratch;

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
      const KernelFunction<spacedim, RangeNumberType> &kernel_function,
      const CellNeighboringType &                      cell_neighboring_type,
      const BEMValues<dim, spacedim, RangeNumberType> *bem_values,
      const PairCellWiseScratchData<dim, spacedim, RangeNumberType> *scratch,
      const unsigned int kx_dof_index,
      const unsigned int ky_dof_index)
    : n_components(kernel_function.n_components)
    , kernel_function(kernel_function)
    , cell_neighboring_type(cell_neighboring_type)
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
    bem_values            = f.bem_values;
    scratch               = f.scratch;
    kx_dof_index          = f.kx_dof_index;
    ky_dof_index          = f.ky_dof_index;

    return *this;
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
    const unsigned int k3_index,
    const unsigned int quad_no,
    const unsigned int component) const
  {
    Assert(dim == 2 && spacedim == 3, ExcNotImplemented());

    /**
     * N.B. The shape function value and their gradient value data tables are
     * related to the finite element objects. Because the values in these tables
     * are directly evaluated in the unit cell, which do not depend on the real
     * cells, i.e. when coming to a new cell, these values need not be updated,
     * hence they are members of @p BEMValues instead of @p ScratchData.
     */
    const Table<3, RangeNumberType> *            kx_shape_value_table = nullptr;
    const Table<3, RangeNumberType> *            ky_shape_value_table = nullptr;
    const Table<2, FullMatrix<RangeNumberType>> *kx_shape_grad_matrix_table =
      nullptr;
    const Table<2, FullMatrix<RangeNumberType>> *ky_shape_grad_matrix_table =
      nullptr;

    /**
     * N.B. The covariant transformation matrix data tables are related to the
     * mapping objects instead of the finite element objects. This is natural to
     * understand, the covariant transformation is caused by the coordinate
     * mapping from the unit cell to the real cell.
     *
     * Because the mapping is dependent on real cells, these data should be
     * updated when coming to a new cell. Hence, they are members of
     * @p ScratchData instead of @p BEMValues.
     */
    const Table<2, FullMatrix<RangeNumberType>> *kx_covariants_table = nullptr;
    const Table<2, FullMatrix<RangeNumberType>> *ky_covariants_table = nullptr;

    Point<spacedim>                      x, y;
    RangeNumberType                      Jx = 0;
    RangeNumberType                      Jy = 0;
    Tensor<1, spacedim, RangeNumberType> nx, ny;

    /**
     * Select data tables according to the cell neighboring type.
     */
    switch (cell_neighboring_type)
      {
        case SamePanel:
          kx_shape_value_table =
            &(bem_values->kx_shape_value_table_for_same_panel);
          ky_shape_value_table =
            &(bem_values->ky_shape_value_table_for_same_panel);

          /**
           * When the kernel type is regularized hyper singular, extract
           * covariant transformation matrices from @p ScratchData and
           * gradient of shape functions from @p BEMValues.
           */
          if (kernel_function.kernel_type == HyperSingularRegular)
            {
              kx_shape_grad_matrix_table =
                &(bem_values->kx_shape_grad_matrix_table_for_same_panel);
              ky_shape_grad_matrix_table =
                &(bem_values->ky_shape_grad_matrix_table_for_same_panel);
              kx_covariants_table = &(scratch->kx_covariants_same_panel);
              ky_covariants_table = &(scratch->ky_covariants_same_panel);
            }

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

          /**
           * When the kernel type is regularized hyper singular, extract
           * covariant transformation matrices from @p ScratchData and
           * gradient of shape functions from @p BEMValues.
           */
          if (kernel_function.kernel_type == HyperSingularRegular)
            {
              kx_shape_grad_matrix_table =
                &(bem_values->kx_shape_grad_matrix_table_for_common_edge);
              ky_shape_grad_matrix_table =
                &(bem_values->ky_shape_grad_matrix_table_for_common_edge);
              kx_covariants_table = &(scratch->kx_covariants_common_edge);
              ky_covariants_table = &(scratch->ky_covariants_common_edge);
            }

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

          /**
           * When the kernel type is regularized hyper singular, extract
           * covariant transformation matrices from @p ScratchData and
           * gradient of shape functions from @p BEMValues.
           */
          if (kernel_function.kernel_type == HyperSingularRegular)
            {
              kx_shape_grad_matrix_table =
                &(bem_values->kx_shape_grad_matrix_table_for_common_vertex);
              ky_shape_grad_matrix_table =
                &(bem_values->ky_shape_grad_matrix_table_for_common_vertex);
              kx_covariants_table = &(scratch->kx_covariants_common_vertex);
              ky_covariants_table = &(scratch->ky_covariants_common_vertex);
            }

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

          /**
           * When the kernel type is regularized hyper singular, extract
           * covariant transformation matrices from @p ScratchData and
           * gradient of shape functions from @p BEMValues.
           */
          if (kernel_function.kernel_type == HyperSingularRegular)
            {
              kx_shape_grad_matrix_table =
                &(bem_values->kx_shape_grad_matrix_table_for_regular);
              ky_shape_grad_matrix_table =
                &(bem_values->ky_shape_grad_matrix_table_for_regular);
              kx_covariants_table = &(scratch->kx_covariants_regular);
              ky_covariants_table = &(scratch->ky_covariants_regular);
            }

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

          if (kernel_function.kernel_type == HyperSingularRegular)
            {
              kx_shape_grad_matrix_table = nullptr;
              ky_shape_grad_matrix_table = nullptr;
              kx_covariants_table        = nullptr;
              ky_covariants_table        = nullptr;
            }

          Assert(false, ExcInternalError());
      }

    /**
     * Negate the normal vector in \f$K_y\f$ when the cell neighboring type is
     * common edge. This is because the cell \f$K_y\f$'s orientation has been
     * reversed.
     */
    if (cell_neighboring_type == CommonEdge)
      {
        ny = -ny;
      }

    if (kernel_function.kernel_type == HyperSingularRegular)
      {
        /**
         * Extract the gradient values of the current shape function in the unit
         * cell for \f$K_x\f$ as well as \f$K_y\f$.
         */
        Vector<RangeNumberType> kx_shape_grad_in_unit_cell(dim);
        Vector<RangeNumberType> ky_shape_grad_in_unit_cell(dim);

        for (unsigned int i = 0; i < dim; i++)
          {
            kx_shape_grad_in_unit_cell(i) =
              (*kx_shape_grad_matrix_table)(k3_index, quad_no)(kx_dof_index, i);
            ky_shape_grad_in_unit_cell(i) =
              (*ky_shape_grad_matrix_table)(k3_index, quad_no)(ky_dof_index, i);
          }

        /**
         * Apply covariant transformation to the gradient tensors.
         */
        Vector<RangeNumberType> kx_shape_grad_in_real_cell(spacedim);
        Vector<RangeNumberType> ky_shape_grad_in_real_cell(spacedim);
        (*kx_covariants_table)(k3_index, quad_no)
          .vmult(kx_shape_grad_in_real_cell, kx_shape_grad_in_unit_cell);
        (*ky_covariants_table)(k3_index, quad_no)
          .vmult(ky_shape_grad_in_real_cell, ky_shape_grad_in_unit_cell);

        /**
         * Calculate the surface gradient tensor of the shape functions, which
         * is the cross product of normal vector and the volume gradient vector.
         *
         * \mynote{The cross product operation requires the input vectors be
         * transformed to tensors.}
         */
        Tensor<1, spacedim, RangeNumberType> kx_shape_surface_curl =
          cross_product_3d(
            nx,
            VectorToTensor<spacedim, RangeNumberType, Vector<RangeNumberType>>(
              kx_shape_grad_in_real_cell));
        Tensor<1, spacedim, RangeNumberType> ky_shape_surface_curl =
          cross_product_3d(
            ny,
            VectorToTensor<spacedim, RangeNumberType, Vector<RangeNumberType>>(
              ky_shape_grad_in_real_cell));

        return kernel_function.value(x, y, nx, ny, component) * Jx * Jy *
               scalar_product(kx_shape_surface_curl, ky_shape_surface_curl);
      }
    else
      {
        /**
         * Evaluate the original kernel function at the specified pair of points
         * in the real cells with their normal vectors, the result of which is
         * then multiplied by the Jacobians and shape function values.
         */
        return kernel_function.value(x, y, nx, ny, component) * Jx * Jy *
               (*kx_shape_value_table)(kx_dof_index, k3_index, quad_no) *
               (*ky_shape_value_table)(ky_dof_index, k3_index, quad_no);
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
     * Evaluate the pullback of kernel function on Sauter's parametric space
     * at the quad_no'th quadrature point under the given 4D quadrature rule.
     * @param quad_no quadrature point index
     * @param component
     * @return
     */
    RangeNumberType
    value(const unsigned int quad_no, const unsigned int component = 0) const;

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

    kernel_on_unit_cell   = f.kernel_on_unit_cell;
    cell_neighboring_type = f.cell_neighboring_type;
    bem_values            = f.bem_values;

    return *this;
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
            double jacobian_det = Utilities::fixed_power<3>(p(0));

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
} // namespace IdeoBEM

#endif /* INCLUDE_BEM_KERNELS_H_ */
