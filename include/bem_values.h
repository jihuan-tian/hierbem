/**
 * @file bem_values.h
 * @brief
 *
 * @date 2022-02-23
 * @author Jihuan Tian
 */
#ifndef INCLUDE_BEM_VALUES_H_
#define INCLUDE_BEM_VALUES_H_

#include <deal.II/base/table.h>
#include <deal.II/base/table_indices.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_base.h>
#include <deal.II/fe/fe_tools.h>

#include "bem_tools.h"
#include "sauter_quadrature_tools.h"

namespace IdeoBEM
{
  using namespace dealii;
  using namespace BEMTools;

  /**
   * Values related to a pair of cells (panels) used in Galerkin BEM, which can
   * be considered as a counterpart of the @p FEValues for FEM in deal.ii.
   *
   * \mynote{The values encapsulated in @p BEMValues are shape functions value
   * and their gradient values at each quadrature point, as well as Sauter
   * quadrature rules for different cell neighboring types. These values will
   * be precalculated for improving the performance.}
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  class BEMValues
  {
  public:
    using FE_Poly_short = FE_Poly<TensorProductPolynomials<dim>, dim, spacedim>;

    /**
     * Constructor
     *
     * \mynote{N.B. There is no default constructor for the class
     * @p BEMValues, because all the internal references to finite
     * element objects and quadrature objects should be initialized once the
     * @p BEMValues object is declared.}
     *
     * @param kx_fe
     * @param ky_fe
     * @param quad_rule_for_same_panel
     * @param quad_rule_for_common_edge
     * @param quad_rule_for_common_vertex
     * @param quad_rule_for_regular
     */
    BEMValues(const FiniteElement<dim, spacedim> &kx_fe,
              const FiniteElement<dim, spacedim> &ky_fe,
              const QGauss<4> &                   quad_rule_for_same_panel,
              const QGauss<4> &                   quad_rule_for_common_edge,
              const QGauss<4> &                   quad_rule_for_common_vertex,
              const QGauss<4> &                   quad_rule_for_regular);


    /**
     * Copy constructor for class @p BEMValues.
     *
     * \mynote{The internal references for finite element objects and
     * quadrature objects contained by the current @p BEMValues object will be
     * bound to the same objects associated with the input @p bem_values.
     * Meanwhile, the data tables will be copied.}
     *
     * @param bem_values
     */
    BEMValues(const BEMValues<dim, spacedim, RangeNumberType> &bem_values);


    /**
     * Calculate the table storing shape function values at Sauter quadrature
     * points for the same panel case.
     *
     * @param kx_fe The finite element for \f$K_x\f$
     * @param ky_fe The finite element for \f$K_y\f$
     * @param sauter_quad_rule
     * @param kx_shape_value_table The 1st dimension is the shape function hierarchical numbering;
     * the 2nd dimension is the index for \f$k_3\f$ terms; the 3rd dimension is
     * the
     * quadrature point number. Refer to @p BEMValues::kx_shape_value_table_for_same_panel.
     * @param ky_shape_value_table the 1st dimension is the shape function hierarchical numbering;
     * the 2nd dimension is the index for \f$k_3\f$ terms; the 3rd dimension is
     * the
     * quadrature point number. Refer to @p BEMValues::ky_shape_value_table_for_same_panel.
     */
    void
    shape_values_same_panel(const FiniteElement<dim, spacedim> &kx_fe,
                            const FiniteElement<dim, spacedim> &ky_fe,
                            const QGauss<4> &          sauter_quad_rule,
                            Table<3, RangeNumberType> &kx_shape_value_table,
                            Table<3, RangeNumberType> &ky_shape_value_table);


    /**
     * Calculate the table storing shape function values at Sauter quadrature
     * points for the common edge case.
     *
     * @param kx_fe The finite element for \f$K_x\f$
     * @param ky_fe The finite element for \f$K_y\f$
     * @param sauter_quad_rule
     * @param kx_shape_value_table The 1st dimension is the shape function hierarchical numbering;
     * the 2nd dimension is the index for \f$k_3\f$ terms; the 3rd dimension is
     * the
     * quadrature point number. Refer to @p BEMValues::kx_shape_value_table_for_common_edge.
     * @param ky_shape_value_table The 1st dimension is the shape function hierarchical numbering;
     * the 2nd dimension is the index for \f$k_3\f$ terms; the 3rd dimension is
     * the
     * quadrature point number. Refer to @p BEMValues::ky_shape_value_table_for_common_edge.
     */
    void
    shape_values_common_edge(const FiniteElement<dim, spacedim> &kx_fe,
                             const FiniteElement<dim, spacedim> &ky_fe,
                             const QGauss<4> &          sauter_quad_rule,
                             Table<3, RangeNumberType> &kx_shape_value_table,
                             Table<3, RangeNumberType> &ky_shape_value_table);


    /**
     * Calculate the table storing shape function values at Sauter quadrature
     * points for the common vertex case.
     *
     * @param kx_fe The finite element for \f$K_x\f$
     * @param ky_fe The finite element for \f$K_y\f$
     * @param sauter_quad_rule
     * @param kx_shape_value_table The 1st dimension is the shape function hierarchical numbering;
     * the 2nd dimension is the index for $k_3$ terms; the 3rd dimension is the
     * quadrature point number. Refer to @p BEMValues::kx_shape_value_table_for_common_vertex.
     * @param ky_shape_value_table The 1st dimension is the shape function hierarchical numbering;
     * the 2nd dimension is the index for $k_3$ terms; the 3rd dimension is the
     * quadrature point number. Refer to @p BEMValues::ky_shape_value_table_for_common_vertex.
     */
    void
    shape_values_common_vertex(const FiniteElement<dim, spacedim> &kx_fe,
                               const FiniteElement<dim, spacedim> &ky_fe,
                               const QGauss<4> &          sauter_quad_rule,
                               Table<3, RangeNumberType> &kx_shape_value_table,
                               Table<3, RangeNumberType> &ky_shape_value_table);


    /**
     * Calculate the table storing shape function values at Sauter quadrature
     * points for the regular case.
     *
     * @param kx_fe The finite element for \f$K_x\f$
     * @param ky_fe The finite element for \f$K_y\f$
     * @param sauter_quad_rule
     * @param kx_shape_value_table The 1st dimension is the shape function hierarchical numbering;
     * the 2nd dimension is the index for $k_3$ terms; the 3rd dimension is the
     * quadrature point number. Refer to @p BEMValues::kx_shape_value_table_for_regular.
     * @param ky_shape_value_table The 1st dimension is the shape function hierarchical numbering;
     * the 2nd dimension is the index for $k_3$ terms; the 3rd dimension is the
     * quadrature point number. Refer to @p BEMValues::ky_shape_value_table_for_regular.
     */
    void
    shape_values_regular(const FiniteElement<dim, spacedim> &kx_fe,
                         const FiniteElement<dim, spacedim> &ky_fe,
                         const QGauss<4> &                   sauter_quad_rule,
                         Table<3, RangeNumberType> &kx_shape_value_table,
                         Table<3, RangeNumberType> &ky_shape_value_table);


    /**
     * Calculate the table storing shape function gradient matrices at Sauter
     * quadrature points for the same panel case. N.B. The shape functions are
     * in the lexicographic order and each row of the gradient matrix
     * corresponds to a shape function.
     *
     * @param kx_fe The finite element for \f$K_x\f$
     * @param ky_fe The finite element for \f$K_y\f$
     * @param sauter_quad_rule
     * @param kx_shape_grad_matrix_table The 1st dimension is the index for \f$k_3\f$
     * terms; the 2nd dimension is the quadrature point number. Refer to
     * @p BEMValues::kx_shape_grad_matrix_table_for_regular.
     * @param ky_shape_grad_matrix_table The 1st dimension is the index for \f$k_3\f$
     * terms; the 2nd dimension is the quadrature point number. Refer to
     * @p BEMValues::ky_shape_grad_matrix_table_for_regular.
     */
    void
    shape_grad_matrices_same_panel(
      const FiniteElement<dim, spacedim> &   kx_fe,
      const FiniteElement<dim, spacedim> &   ky_fe,
      const QGauss<4> &                      sauter_quad_rule,
      Table<2, FullMatrix<RangeNumberType>> &kx_shape_grad_matrix_table,
      Table<2, FullMatrix<RangeNumberType>> &ky_shape_grad_matrix_table);


    /**
     * Calculate the table storing shape function gradient matrices at Sauter
     * quadrature points for the common edge case. N.B. The shape functions are
     * in the lexicographic order and each row of the gradient matrix
     * corresponds to a shape function.
     *
     * @param kx_fe The finite element for \f$K_x\f$
     * @param ky_fe The finite element for \f$K_y\f$
     * @param sauter_quad_rule
     * @param kx_shape_grad_matrix_table The 1st dimension is the index for \f$k_3\f$
     * terms; the 2nd dimension is the quadrature point number. Refer to
     * @p BEMValues::kx_shape_grad_matrix_table_for_common_edge.
     * @param ky_shape_grad_matrix_table The 1st dimension is the index for \f$k_3\f$
     * terms; the 2nd dimension is the quadrature point number. Refer to
     * @p BEMValues::ky_shape_grad_matrix_table_for_common_edge.
     */
    void
    shape_grad_matrices_common_edge(
      const FiniteElement<dim, spacedim> &   kx_fe,
      const FiniteElement<dim, spacedim> &   ky_fe,
      const QGauss<4> &                      sauter_quad_rule,
      Table<2, FullMatrix<RangeNumberType>> &kx_shape_grad_matrix_table,
      Table<2, FullMatrix<RangeNumberType>> &ky_shape_grad_matrix_table);


    /**
     * Calculate the table storing shape function gradient matrices at Sauter
     * quadrature points for the common vertex case. N.B. The shape functions
     * are in the lexicographic order and each row of the gradient matrix
     * corresponds to a shape function.
     *
     * @param kx_fe The finite element for \f$K_x\f$
     * @param ky_fe The finite element for \f$K_y\f$
     * @param sauter_quad_rule
     * @param kx_shape_grad_matrix_table The 1st dimension is the index for \f$k_3\f$
     * terms; the 2nd dimension is the quadrature point number. Refer to
     * @p BEMValues::kx_shape_grad_matrix_table_for_common_vertex.
     * @param ky_shape_grad_matrix_table The 1st dimension is the index for \f$k_3\f$
     * terms; the 2nd dimension is the quadrature point number. Refer to
     * @p BEMValues::ky_shape_grad_matrix_table_for_common_vertex.
     */
    void
    shape_grad_matrices_common_vertex(
      const FiniteElement<dim, spacedim> &   kx_fe,
      const FiniteElement<dim, spacedim> &   ky_fe,
      const QGauss<4> &                      sauter_quad_rule,
      Table<2, FullMatrix<RangeNumberType>> &kx_shape_grad_matrix_table,
      Table<2, FullMatrix<RangeNumberType>> &ky_shape_grad_matrix_table);


    /**
     * Calculate the table storing shape function gradient matrices at Sauter
     * quadrature points for the regular case. N.B. The shape functions are
     * in the lexicographic order and each row of the gradient matrix
     * corresponds to a shape function.
     *
     * @param kx_fe The finite element for \f$K_x\f$
     * @param ky_fe The finite element for \f$K_y\f$
     * @param sauter_quad_rule
     * @param kx_shape_grad_matrix_table The 1st dimension is the quadrature point
     * number. Refer to @p BEMValues::kx_shape_grad_matrix_table_for_regular.
     * @param ky_shape_grad_matrix_table The 1st dimension is the quadrature point
     * number. Refer to @p BEMValues::ky_shape_grad_matrix_table_for_regular.
     */
    void
    shape_grad_matrices_regular(
      const FiniteElement<dim, spacedim> &   kx_fe,
      const FiniteElement<dim, spacedim> &   ky_fe,
      const QGauss<4> &                      sauter_quad_rule,
      Table<2, FullMatrix<RangeNumberType>> &kx_shape_grad_matrix_table,
      Table<2, FullMatrix<RangeNumberType>> &ky_shape_grad_matrix_table);


    /**
     * Reference to finite element on the field cell \f$K_x\f$.
     */
    const FiniteElement<dim, spacedim> &kx_fe;
    /**
     * Reference to finite element on the field cell \f$K_y\f$.
     */
    const FiniteElement<dim, spacedim> &ky_fe;
    /**
     * Reference to 4D Sauter quadrature rule for the case that \f$K_x \equiv
     * K_y\f$.
     */
    const QGauss<4> &quad_rule_for_same_panel;
    /**
     * Reference to 4D Sauter quadrature rule for the case that \f$K_x\f$ and
     * \f$K_y\f$ share a common edge.
     */
    const QGauss<4> &quad_rule_for_common_edge;
    /**
     * Reference to 4D Sauter quadrature rule for the case that \f$K_x\f$ and
     * \f$K_y\f$ share a common vertex.
     */
    const QGauss<4> &quad_rule_for_common_vertex;
    /**
     * Reference to 4D Sauter quadrature rule for the case that \f$K_x\f$ and
     * \f$K_y\f$ are separated.
     */
    const QGauss<4> &quad_rule_for_regular;

    /**
     * Data table of shape function values for \f$K_x\f$ in the same panel case.
     * It has three dimensions:
     * 1. shape function index: size=@p dofs_per_cell
     * 2. \f$k_3\f$ term index: size=8
     * 3. Quadrature point index: size=number of quadrature points
     */
    Table<3, RangeNumberType> kx_shape_value_table_for_same_panel;
    /**
     * Data table of shape function values for \f$K_y\f$ in the same panel case.
     * It has three dimensions:
     * 1. shape function index: size=@p dofs_per_cell
     * 2. \f$k_3\f$ term index: size=8
     * 3. Quadrature point index: size=number of quadrature points
     */
    Table<3, RangeNumberType> ky_shape_value_table_for_same_panel;
    /**
     * Data table of shape function values for \f$K_x\f$ in the common edge
     * case. It has three dimensions:
     * 1. shape function index: size=@p dofs_per_cell
     * 2. \f$k_3\f$ term index: size=6
     * 3. Quadrature point index: size=number of quadrature points
     */
    Table<3, RangeNumberType> kx_shape_value_table_for_common_edge;
    /**
     * Data table of shape function values for \f$K_y\f$ in the common edge
     * case. It has three dimensions:
     * 1. shape function index: size=@p dofs_per_cell
     * 2. \f$k_3\f$ term index: size=6
     * 3. Quadrature point index: size=number of quadrature points
     */
    Table<3, RangeNumberType> ky_shape_value_table_for_common_edge;
    /**
     * Data table of shape function values for \f$K_x\f$ in the common vertex
     * case. It has three dimensions:
     * 1. shape function index: size=@p dofs_per_cell
     * 2. \f$k_3\f$ term index: size=4
     * 3. Quadrature point index: size=number of quadrature points
     */
    Table<3, RangeNumberType> kx_shape_value_table_for_common_vertex;
    /**
     * Data table of shape function values for \f$K_y\f$ in the common vertex
     * case. It has three dimensions:
     * 1. shape function index: size=@p dofs_per_cell
     * 2. \f$k_3\f$ term index: size=4
     * 3. Quadrature point index: size=number of quadrature points
     */
    Table<3, RangeNumberType> ky_shape_value_table_for_common_vertex;
    /**
     * Data table of shape function values for \f$K_x\f$ in the regular case.
     * It has three dimensions:
     * 1. shape function index: size=@p dofs_per_cell
     * 2. \f$k_3\f$ term index: size=1
     * 3. Quadrature point index: size=number of quadrature points
     */
    Table<3, RangeNumberType> kx_shape_value_table_for_regular;
    /**
     * Data table of shape function values for \f$K_y\f$ in the regular case.
     * It has three dimensions:
     * 1. shape function index: size=@p dofs_per_cell
     * 2. \f$k_3\f$ term index: size=1
     * 3. Quadrature point index: size=number of quadrature points
     */
    Table<3, RangeNumberType> ky_shape_value_table_for_regular;

    /**
     * Data table of shape function's gradient values for \f$K_x\f$ in the same
     * panel case. It has two dimensions:
     * 1. \f$k_3\f$ term index: size=8
     * 2. Quadrature point index: size=number of quadrature points
     * N.B. Each data item in the table is itself a matrix with the dimension
     * \f$dofs_per_cell*spacedim\f$.
     */
    Table<2, FullMatrix<RangeNumberType>>
      kx_shape_grad_matrix_table_for_same_panel;
    /**
     * Data table of shape function's gradient values for \f$K_y\f$ in the same
     * panel case. It has two dimensions:
     * 1. \f$k_3\f$ term index: size=8
     * 2. Quadrature point index: size=number of quadrature points
     * N.B. Each data item in the table is itself a matrix with the dimension
     * \f$dofs_per_cell*spacedim\f$.
     */
    Table<2, FullMatrix<RangeNumberType>>
      ky_shape_grad_matrix_table_for_same_panel;
    /**
     * Data table of shape function's gradient values for \f$K_x\f$ in the
     * common edge case. It has two dimensions:
     * 1. \f$k_3\f$ term index: size=6
     * 2. Quadrature point index: size=number of quadrature points
     * N.B. Each data item in the table is itself a matrix with the dimension
     * \f$dofs_per_cell*spacedim\f$.
     */
    Table<2, FullMatrix<RangeNumberType>>
      kx_shape_grad_matrix_table_for_common_edge;
    /**
     * Data table of shape function's gradient values for \f$K_y\f$ in the
     * common edge case. It has two dimensions:
     * 1. \f$k_3\f$ term index: size=6
     * 2. Quadrature point index: size=number of quadrature points
     * N.B. Each data item in the table is itself a matrix with the dimension
     * \f$dofs_per_cell*spacedim\f$.
     */
    Table<2, FullMatrix<RangeNumberType>>
      ky_shape_grad_matrix_table_for_common_edge;
    /**
     * Data table of shape function's gradient values for \f$K_x\f$ in the
     * common vertex case. It has two dimensions:
     * 1. \f$k_3\f$ term index: size=4
     * 2. Quadrature point index: size=number of quadrature points
     * N.B. Each data item in the table is itself a matrix with the dimension
     * \f$dofs_per_cell*spacedim\f$.
     */
    Table<2, FullMatrix<RangeNumberType>>
      kx_shape_grad_matrix_table_for_common_vertex;
    /**
     * Data table of shape function's gradient values for \f$K_y\f$ in the
     * common vertex case. It has two dimensions:
     * 1. \f$k_3\f$ term index: size=4
     * 2. Quadrature point index: size=number of quadrature points
     * N.B. Each data item in the table is itself a matrix with the dimension
     * \f$dofs_per_cell*spacedim\f$.
     */
    Table<2, FullMatrix<RangeNumberType>>
      ky_shape_grad_matrix_table_for_common_vertex;
    /**
     * Data table of shape function's gradient values for \f$K_x\f$ in the
     * regular case. It has two dimensions:
     * 1. \f$k_3\f$ term index: size=1
     * 2. Quadrature point index: size=number of quadrature points
     * N.B. Each data item in the table is itself a matrix with the dimension
     * \f$dofs_per_cell*spacedim\f$.
     */
    Table<2, FullMatrix<RangeNumberType>>
      kx_shape_grad_matrix_table_for_regular;
    /**
     * Data table of shape function's gradient values for \f$K_y\f$ in the
     * regular case. It has two dimensions:
     * 1. \f$k_3\f$ term index: size=1
     * 2. Quadrature point index: size=number of quadrature points
     * N.B. Each data item in the table is itself a matrix with the dimension
     * \f$dofs_per_cell*spacedim\f$.
     */
    Table<2, FullMatrix<RangeNumberType>>
      ky_shape_grad_matrix_table_for_regular;

    /**
     * Fill the data tables for shape function values.
     */
    void
    fill_shape_value_tables();

    /**
     * Fill the data tables for the gradient values of shape functions.
     */
    void
    fill_shape_grad_matrix_tables();

  protected:
    /**
     * Initialize the data tables for shape function values.
     */
    void
    init_shape_value_tables();

    /**
     * Initialize the data tables for the gradient values of shape functions.
     */
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
    shape_values_same_panel(kx_fe,
                            ky_fe,
                            quad_rule_for_same_panel,
                            kx_shape_value_table_for_same_panel,
                            ky_shape_value_table_for_same_panel);

    shape_values_common_edge(kx_fe,
                             ky_fe,
                             quad_rule_for_common_edge,
                             kx_shape_value_table_for_common_edge,
                             ky_shape_value_table_for_common_edge);

    shape_values_common_vertex(kx_fe,
                               ky_fe,
                               quad_rule_for_common_vertex,
                               kx_shape_value_table_for_common_vertex,
                               ky_shape_value_table_for_common_vertex);

    shape_values_regular(kx_fe,
                         ky_fe,
                         quad_rule_for_regular,
                         kx_shape_value_table_for_regular,
                         ky_shape_value_table_for_regular);
  }


  template <int dim, int spacedim, typename RangeNumberType>
  void
  BEMValues<dim, spacedim, RangeNumberType>::fill_shape_grad_matrix_tables()
  {
    shape_grad_matrices_same_panel(kx_fe,
                                   ky_fe,
                                   quad_rule_for_same_panel,
                                   kx_shape_grad_matrix_table_for_same_panel,
                                   ky_shape_grad_matrix_table_for_same_panel);

    shape_grad_matrices_common_edge(kx_fe,
                                    ky_fe,
                                    quad_rule_for_common_edge,
                                    kx_shape_grad_matrix_table_for_common_edge,
                                    ky_shape_grad_matrix_table_for_common_edge);

    shape_grad_matrices_common_vertex(
      kx_fe,
      ky_fe,
      quad_rule_for_common_vertex,
      kx_shape_grad_matrix_table_for_common_vertex,
      ky_shape_grad_matrix_table_for_common_vertex);

    shape_grad_matrices_regular(kx_fe,
                                ky_fe,
                                quad_rule_for_regular,
                                kx_shape_grad_matrix_table_for_regular,
                                ky_shape_grad_matrix_table_for_regular);
  }


  template <int dim, int spacedim, typename RangeNumberType>
  void
  BEMValues<dim, spacedim, RangeNumberType>::shape_values_same_panel(
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

    /**
     * Get the polynomial space inverse numbering for accessing the shape
     * functions in the lexicographic order.
     *
     * \alert{Here I have adopted an assumption that the finite elements are
     * based on tensor product polynomials.}
     */
    const FE_Poly_short &kx_fe_poly =
      dynamic_cast<const FE_Poly_short &>(kx_fe);
    const FE_Poly_short &ky_fe_poly =
      dynamic_cast<const FE_Poly_short &>(ky_fe);

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
            // Transform the quadrature point in the parametric space to
            // the unit cells of \f$K_x\f$ and \f$K_y\f$.
            sauter_same_panel_parametric_coords_to_unit_cells(quad_points[q],
                                                              k,
                                                              kx_quad_point,
                                                              ky_quad_point);

            // Iterate over each shape function on the unit cell of \f$K_x\f$
            // and evaluate it at <code>kx_quad_point</code>. N.B. The
            // shape functions are in the lexicographic order.
            for (unsigned int s = 0; s < kx_dofs_per_cell; s++)
              {
                kx_shape_value_table(s, k, q) =
                  kx_fe.shape_value(kx_poly_space_inverse_numbering[s],
                                    kx_quad_point);
              }

            /**
             *  Iterate over each shape function in the lexicographic order on
             *  the unit cell of \f$K_y\f$ and evaluate it at @p ky_quad_point.
             */
            for (unsigned int s = 0; s < ky_dofs_per_cell; s++)
              {
                ky_shape_value_table(s, k, q) =
                  ky_fe.shape_value(ky_poly_space_inverse_numbering[s],
                                    ky_quad_point);
              }
          }
      }
  }


  template <int dim, int spacedim, typename RangeNumberType>
  void
  BEMValues<dim, spacedim, RangeNumberType>::shape_values_common_edge(
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

    /**
     * Get the polynomial space inverse numbering for accessing the shape
     * functions in the lexicographic order.
     *
     * \alert{Here I have adopted an assumption that the finite elements are
     * based on tensor product polynomials.}
     */
    const FE_Poly_short &kx_fe_poly =
      dynamic_cast<const FE_Poly_short &>(kx_fe);
    const FE_Poly_short &ky_fe_poly =
      dynamic_cast<const FE_Poly_short &>(ky_fe);

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
            // Transform the quadrature point in the parametric space to
            // the unit cells of \f$K_x\f$ and \f$K_y\f$.
            sauter_common_edge_parametric_coords_to_unit_cells(quad_points[q],
                                                               k,
                                                               kx_quad_point,
                                                               ky_quad_point);

            /**
             *  Iterate over each shape function in the lexicographic order on
             *  the unit cell of \f$K_x\f$ and evaluate it at @p kx_quad_point.
             */
            for (unsigned int s = 0; s < kx_dofs_per_cell; s++)
              {
                kx_shape_value_table(s, k, q) =
                  kx_fe.shape_value(kx_poly_space_inverse_numbering[s],
                                    kx_quad_point);
              }

            /**
             *  Iterate over each shape function in the lexicographic order on
             *  the unit cell of \f$K_y\f$ and evaluate it at @p ky_quad_point.
             */
            for (unsigned int s = 0; s < ky_dofs_per_cell; s++)
              {
                ky_shape_value_table(s, k, q) =
                  ky_fe.shape_value(ky_poly_space_inverse_numbering[s],
                                    ky_quad_point);
              }
          }
      }
  }


  template <int dim, int spacedim, typename RangeNumberType>
  void
  BEMValues<dim, spacedim, RangeNumberType>::shape_values_common_vertex(
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

    /**
     * Get the polynomial space inverse numbering for accessing the shape
     * functions in the lexicographic order.
     *
     * \alert{Here I have adopted an assumption that the finite elements are
     * based on tensor product polynomials.}
     */
    const FE_Poly_short &kx_fe_poly =
      dynamic_cast<const FE_Poly_short &>(kx_fe);
    const FE_Poly_short &ky_fe_poly =
      dynamic_cast<const FE_Poly_short &>(ky_fe);

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
            // Transform the quadrature point in the parametric space to
            // the unit cells of \f$K_x\f$ and \f$K_y\f$.
            sauter_common_vertex_parametric_coords_to_unit_cells(quad_points[q],
                                                                 k,
                                                                 kx_quad_point,
                                                                 ky_quad_point);

            // Iterate over each shape function on the unit cell of \f$K_x\f$
            // and evaluate it at <code>kx_quad_point</code>. N.B. The
            // shape functions are in the default hierarchical order.
            for (unsigned int s = 0; s < kx_dofs_per_cell; s++)
              {
                kx_shape_value_table(s, k, q) =
                  kx_fe.shape_value(kx_poly_space_inverse_numbering[s],
                                    kx_quad_point);
              }

            /**
             *  Iterate over each shape function in the lexicographic order on
             *  the unit cell of \f$K_y\f$ and evaluate it at @p ky_quad_point.
             */
            for (unsigned int s = 0; s < ky_dofs_per_cell; s++)
              {
                ky_shape_value_table(s, k, q) =
                  ky_fe.shape_value(ky_poly_space_inverse_numbering[s],
                                    ky_quad_point);
              }
          }
      }
  }


  template <int dim, int spacedim, typename RangeNumberType>
  void
  BEMValues<dim, spacedim, RangeNumberType>::shape_values_regular(
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

    /**
     * Get the polynomial space inverse numbering for accessing the shape
     * functions in the lexicographic order.
     *
     * \alert{Here I have adopted an assumption that the finite elements are
     * based on tensor product polynomials.}
     */
    const FE_Poly_short &kx_fe_poly =
      dynamic_cast<const FE_Poly_short &>(kx_fe);
    const FE_Poly_short &ky_fe_poly =
      dynamic_cast<const FE_Poly_short &>(ky_fe);

    std::vector<unsigned int> kx_poly_space_inverse_numbering(
      kx_fe_poly.get_poly_space_numbering_inverse());
    std::vector<unsigned int> ky_poly_space_inverse_numbering(
      ky_fe_poly.get_poly_space_numbering_inverse());

    // Iterate over each quadrature point.
    for (unsigned int q = 0; q < n_q_points; q++)
      {
        // Transform the quadrature point in the parametric space to the
        // unit cells of \f$K_x\f$ and \f$K_y\f$.
        sauter_regular_parametric_coords_to_unit_cells(quad_points[q],
                                                       kx_quad_point,
                                                       ky_quad_point);

        /**
         *  Iterate over each shape function in the lexicographic order on
         *  the unit cell of \f$K_x\f$ and evaluate it at @p ky_quad_point.
         */
        for (unsigned int s = 0; s < kx_dofs_per_cell; s++)
          {
            kx_shape_value_table(s, 0, q) =
              kx_fe.shape_value(kx_poly_space_inverse_numbering[s],
                                kx_quad_point);
          }

        /**
         *  Iterate over each shape function in the lexicographic order on
         *  the unit cell of \f$K_y\f$ and evaluate it at @p ky_quad_point.
         */
        for (unsigned int s = 0; s < ky_dofs_per_cell; s++)
          {
            ky_shape_value_table(s, 0, q) =
              ky_fe.shape_value(ky_poly_space_inverse_numbering[s],
                                ky_quad_point);
          }
      }
  }


  template <int dim, int spacedim, typename RangeNumberType>
  void
  BEMValues<dim, spacedim, RangeNumberType>::shape_grad_matrices_same_panel(
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

    std::vector<Point<dim * 2>> quad_points = sauter_quad_rule.get_points();

    Point<dim> kx_quad_point;
    Point<dim> ky_quad_point;

    // Iterate over each $k_3$ part.
    for (unsigned k = 0; k < 8; k++)
      {
        // Iterate over each quadrature point.
        for (unsigned int q = 0; q < n_q_points; q++)
          {
            // Transform the quadrature point in the parametric space to the
            // unit cells of \f$K_x\f$ and \f$K_y\f$.
            sauter_same_panel_parametric_coords_to_unit_cells(quad_points[q],
                                                              k,
                                                              kx_quad_point,
                                                              ky_quad_point);

            // Calculate the gradient matrix evaluated at
            // <code>kx_quad_point</code> in  \f$K_x\f$. Matrix rows correspond
            // to shape functions which are in the lexicographic order.
            kx_shape_grad_matrix_table(k, q) =
              BEMTools::shape_grad_matrix_in_lexicographic_order(kx_fe,
                                                                 kx_quad_point);
            // Calculate the gradient matrix evaluated at
            // <code>ky_quad_point</code> in  \f$K_y\f$. Matrix rows correspond
            // to shape functions which are in the lexicographic order.
            ky_shape_grad_matrix_table(k, q) =
              BEMTools::shape_grad_matrix_in_lexicographic_order(ky_fe,
                                                                 ky_quad_point);
          }
      }
  }


  template <int dim, int spacedim, typename RangeNumberType>
  void
  BEMValues<dim, spacedim, RangeNumberType>::shape_grad_matrices_common_edge(
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

    std::vector<Point<dim * 2>> quad_points = sauter_quad_rule.get_points();

    Point<dim> kx_quad_point;
    Point<dim> ky_quad_point;

    // Iterate over each $k_3$ part.
    for (unsigned k = 0; k < 6; k++)
      {
        // Iterate over each quadrature point.
        for (unsigned int q = 0; q < n_q_points; q++)
          {
            // Transform the quadrature point in the parametric space to the
            // unit cells of \f$K_x\f$ and \f$K_y\f$.
            sauter_common_edge_parametric_coords_to_unit_cells(quad_points[q],
                                                               k,
                                                               kx_quad_point,
                                                               ky_quad_point);

            // Calculate the gradient matrix evaluated at
            // <code>kx_quad_point</code> in  \f$K_x\f$. Matrix rows correspond
            // to shape functions which are in the lexicographic order.
            kx_shape_grad_matrix_table(k, q) =
              BEMTools::shape_grad_matrix_in_lexicographic_order(kx_fe,
                                                                 kx_quad_point);
            // Calculate the gradient matrix evaluated at
            // <code>ky_quad_point</code> in  \f$K_y\f$. Matrix rows correspond
            // to shape functions which are in the lexicographic order.
            ky_shape_grad_matrix_table(k, q) =
              BEMTools::shape_grad_matrix_in_lexicographic_order(ky_fe,
                                                                 ky_quad_point);
          }
      }
  }


  template <int dim, int spacedim, typename RangeNumberType>
  void
  BEMValues<dim, spacedim, RangeNumberType>::shape_grad_matrices_common_vertex(
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

    std::vector<Point<dim * 2>> quad_points = sauter_quad_rule.get_points();

    Point<dim> kx_quad_point;
    Point<dim> ky_quad_point;

    // Iterate over each $k_3$ part.
    for (unsigned k = 0; k < 4; k++)
      {
        // Iterate over each quadrature point.
        for (unsigned int q = 0; q < n_q_points; q++)
          {
            // Transform the quadrature point in the parametric space to the
            // unit cells of \f$K_x\f$ and \f$K_y\f$.
            sauter_common_vertex_parametric_coords_to_unit_cells(quad_points[q],
                                                                 k,
                                                                 kx_quad_point,
                                                                 ky_quad_point);

            // Calculate the gradient matrix evaluated at
            // <code>kx_quad_point</code> in  \f$K_x\f$. Matrix rows correspond
            // to shape functions which are in the lexicographic order.
            kx_shape_grad_matrix_table(k, q) =
              BEMTools::shape_grad_matrix_in_lexicographic_order(kx_fe,
                                                                 kx_quad_point);
            // Calculate the gradient matrix evaluated at
            // <code>ky_quad_point</code> in  \f$K_y\f$. Matrix rows correspond
            // to shape functions which are in the lexicographic order.
            ky_shape_grad_matrix_table(k, q) =
              BEMTools::shape_grad_matrix_in_lexicographic_order(ky_fe,
                                                                 ky_quad_point);
          }
      }
  }


  template <int dim, int spacedim, typename RangeNumberType>
  void
  BEMValues<dim, spacedim, RangeNumberType>::shape_grad_matrices_regular(
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

    std::vector<Point<dim * 2>> quad_points = sauter_quad_rule.get_points();

    Point<dim> kx_quad_point;
    Point<dim> ky_quad_point;

    // Iterate over each quadrature point.
    for (unsigned int q = 0; q < n_q_points; q++)
      {
        // Transform the quadrature point in the parametric space to the
        // unit cells of \f$K_x\f$ and \f$K_y\f$.
        sauter_regular_parametric_coords_to_unit_cells(quad_points[q],
                                                       kx_quad_point,
                                                       ky_quad_point);

        // Calculate the gradient matrix evaluated at
        // <code>kx_quad_point</code> in  \f$K_x\f$. Matrix rows correspond to
        // shape functions which are in the lexicographic order.
        kx_shape_grad_matrix_table(0, q) =
          BEMTools::shape_grad_matrix_in_lexicographic_order(kx_fe,
                                                             kx_quad_point);
        // Calculate the gradient matrix evaluated at
        // <code>ky_quad_point</code> in  \f$K_y\f$. Matrix rows correspond to
        // shape functions which are in the lexicographic order.
        ky_shape_grad_matrix_table(0, q) =
          BEMTools::shape_grad_matrix_in_lexicographic_order(ky_fe,
                                                             ky_quad_point);
      }
  }

  /**
   * Structure holding cell-wise local matrix data and DoF indices, which is
   * used for SMP parallel computation of the scaled FEM mass matrix.
   */
  template <int dim, int spacedim = dim, typename RangeNumberType = double>
  struct CellWisePerTaskData
  {
    FullMatrix<RangeNumberType> local_matrix;
    // N.B. Memory should be preallocated for this vector before calling
    // <code>get_dof_indices</code>.
    std::vector<types::global_dof_index> local_dof_indices_for_test_space;
    std::vector<types::global_dof_index> local_dof_indices_for_trial_space;

    /**
     * Constructor. Allocate memory for internal members.
     *
     * @param fe_for_test_space
     * @param fe_for_trial_space
     */
    CellWisePerTaskData(const FiniteElement<dim, spacedim> &fe_for_test_space,
                        const FiniteElement<dim, spacedim> &fe_for_trial_space)
      : local_matrix(fe_for_test_space.dofs_per_cell,
                     fe_for_trial_space.dofs_per_cell)
      , local_dof_indices_for_test_space(fe_for_test_space.dofs_per_cell)
      , local_dof_indices_for_trial_space(fe_for_trial_space.dofs_per_cell)
    {}

    /**
     * Copy constructor
     *
     * @param task_data
     */
    CellWisePerTaskData(
      const CellWisePerTaskData<dim, spacedim, RangeNumberType> &task_data)
      : local_matrix(task_data.local_matrix)
      , local_dof_indices_for_test_space(
          task_data.local_dof_indices_for_test_space)
      , local_dof_indices_for_trial_space(
          task_data.local_dof_indices_for_trial_space)
    {}
  };


  /**
   * Structure holding temporary data which are needed for cell-wise
   * integration, such as for the scaled mass matrix term \f$(v, \alpha \cdot
   * u)\f$.
   */
  template <int dim, int spacedim = dim>
  struct CellWiseScratchData
  {
    FEValues<dim, spacedim> fe_values_for_test_space;
    FEValues<dim, spacedim> fe_values_for_trial_space;

    /**
     * Constructor
     *
     * @param fe_for_test_space
     * @param fe_for_trial_space
     * @param quadrature
     * @param update_flags
     */
    CellWiseScratchData(const FiniteElement<dim, spacedim> &fe_for_test_space,
                        const FiniteElement<dim, spacedim> &fe_for_trial_space,
                        const Quadrature<dim> &             quadrature,
                        const UpdateFlags                   update_flags)
      : fe_values_for_test_space(fe_for_test_space, quadrature, update_flags)
      , fe_values_for_trial_space(fe_for_trial_space, quadrature, update_flags)
    {}


    /**
     * Copy constructor. Because <code>FEValues</code> is neither copyable nor
     * has it copy constructor, this copy constructor is mandatory for
     * replication into each task.
     *
     * @param scratch_data
     */
    CellWiseScratchData(const CellWiseScratchData<dim, spacedim> &scratch_data)
      : fe_values_for_test_space(
          scratch_data.fe_values_for_test_space.get_fe(),
          scratch_data.fe_values_for_test_space.get_quadrature(),
          scratch_data.fe_values_for_test_space.get_update_flags())
      , fe_values_for_trial_space(
          scratch_data.fe_values_for_trial_space.get_fe(),
          scratch_data.fe_values_for_trial_space.get_quadrature(),
          scratch_data.fe_values_for_trial_space.get_update_flags())
    {}
  };


  template <int dim, int spacedim = dim, typename RangeNumberType = double>
  struct CellWiseScratchDataForPotentialEval
  {
    FEValues<dim, spacedim> fe_values_for_trial_space;

    CellWiseScratchDataForPotentialEval(
      const FiniteElement<dim, spacedim> &fe_for_trial_space,
      const Quadrature<dim> &             quadrature,
      const UpdateFlags                   update_flags)
      : fe_values_for_trial_space(fe_for_trial_space, quadrature, update_flags)
    {}

    /**
     * Copy constructor
     */
    CellWiseScratchDataForPotentialEval(
      const CellWiseScratchDataForPotentialEval<dim, spacedim> &scratch_data)
      : fe_values_for_trial_space(
          scratch_data.fe_values_for_trial_space.get_fe(),
          scratch_data.fe_values_for_trial_space.get_quadrature(),
          scratch_data.fe_values_for_trial_space.get_update_flags())
    {}
  };


  template <int dim, int spacedim = dim, typename RangeNumberType = double>
  struct CellWisePerTaskDataForPotentialEval
  {
    Vector<RangeNumberType>              local_vector;
    std::vector<types::global_dof_index> local_dof_indices_for_trial_space;

    CellWisePerTaskDataForPotentialEval(
      const FiniteElement<dim, spacedim> &fe_for_trial_space)
      : local_vector(fe_for_trial_space.dofs_per_cell)
      , local_dof_indices_for_trial_space(fe_for_trial_space.dofs_per_cell)
    {}

    CellWisePerTaskDataForPotentialEval(
      const CellWisePerTaskDataForPotentialEval<dim, spacedim, RangeNumberType>
        &task_data)
      : local_vector(task_data.local_vector)
      , local_dof_indices_for_trial_space(
          task_data.local_dof_indices_for_trial_space)
    {}
  };


  /**
   * Structure holding pair-cell-wise local matrix data and DoF indices, which
   * is used for SMP parallel computation of BEM matrices.
   */
  template <int dim, int spacedim = dim, typename RangeNumberType = double>
  struct PairCellWiseScratchData
  {
    using FE_Poly_short = FE_Poly<TensorProductPolynomials<dim>, dim, spacedim>;

    /**
     * The intersection set of the vertex DoF indices for the two cells
     * \f$K_x\f$ and \f$K_y\f$.
     */
    std::vector<std::pair<types::global_dof_index, types::global_dof_index>>
      common_vertex_dof_indices;

    /**
     * List of support points in the real cell \f$K_x\f$ in the default DoF
     * order.
     */
    std::vector<Point<spacedim, RangeNumberType>>
      kx_support_points_in_default_dof_order;
    /**
     * List of support points in the real cell \f$K_y\f$ in the default DoF
     * order.
     */
    std::vector<Point<spacedim, RangeNumberType>>
      ky_support_points_in_default_dof_order;

    /**
     * Permuted list of support points in the real cell \f$K_x\f$ in the
     * lexicographic order in the same panel case and regular case, and
     * determined by @p kx_local_dof_permutation in the common edge case and
     * common vertex case.
     */
    std::vector<Point<spacedim, RangeNumberType>> kx_support_points_permuted;
    /**
     * Permuted list of support points in the real cell \f$K_y\f$ in the
     * lexicographic order in the same panel case and regular case, and
     * determined by @p ky_local_dof_permutation in the common edge case and
     * common vertex case.
     */
    std::vector<Point<spacedim, RangeNumberType>> ky_support_points_permuted;

    /**
     * The list of DoF indices in \f$K_x\f$ which are ordered in the
     * default DoF order. This is directly retrieved from the function
     * @p DoFHandler::cell_iterator::get_dof_indices.
     */
    std::vector<types::global_dof_index>
      kx_local_dof_indices_in_default_dof_order;
    /**
     * The list of DoF indices in \f$K_y\f$ which are ordered in the
     * default DoF order. This is directly retrieved from the function
     * @p DoFHandler::cell_iterator::get_dof_indices.
     */
    std::vector<types::global_dof_index>
      ky_local_dof_indices_in_default_dof_order;

    /**
     * The numbering used for accessing the list of DoFs in \f$K_x\f$ in the
     * lexicographic order, where the list of DoFs are stored in the default DoF
     * order.
     */
    std::vector<unsigned int> kx_fe_poly_space_numbering_inverse;
    /**
     * The numbering used for accessing the list of DoFs in \f$K_y\f$ in the
     * lexicographic order, where the list of DoFs are stored in the default DoF
     * order.
     */
    std::vector<unsigned int> ky_fe_poly_space_numbering_inverse;
    /**
     * The numbering used for accessing the list of DoFs in \f$K_y\f$ in the
     * reversed lexicographic order, where the list of DoFs are stored in the
     * hierarchical order.
     *
     * \mynote{This numbering occurs only when \f$K_x\f$ and \f$K_y\f$ share a
     * common edge.}
     */
    std::vector<unsigned int> ky_fe_reversed_poly_space_numbering_inverse;

    /**
     * The numbering used for accessing the list of support points and
     * associated DoF indices in \f$K_x\f$ in the lexicographic order by
     * starting from a specific vertex, where the list of support points and
     * associated DoF indices are stored in the default DoF order.
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
     * default DoF order.
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
    Table<dim, RangeNumberType> kx_jacobians_same_panel;
    /**
     * Jacobian from the unit cell to the real cell \f$K_x\f$ for each
     * \f$k_3\f$ term and at each quadrature point for the common edge case.
     */
    Table<dim, RangeNumberType> kx_jacobians_common_edge;
    /**
     * Jacobian from the unit cell to the real cell \f$K_x\f$ for each
     * \f$k_3\f$ term and at each quadrature point for the common vertex case.
     */
    Table<dim, RangeNumberType> kx_jacobians_common_vertex;
    /**
     * Jacobian from the unit cell to the real cell \f$K_x\f$ for each
     * \f$k_3\f$ term and at each quadrature point for the regular case.
     */
    Table<dim, RangeNumberType> kx_jacobians_regular;

    /**
     * Normal vector at each quadrature point in the real cell \f$K_x\f$ for
     * the same panel case.
     */
    Table<dim, Tensor<1, spacedim, RangeNumberType>> kx_normals_same_panel;
    /**
     * Normal vector at each quadrature point in the real cell \f$K_x\f$ for
     * the common edge case.
     */
    Table<dim, Tensor<1, spacedim, RangeNumberType>> kx_normals_common_edge;
    /**
     * Normal vector at each quadrature point in the real cell \f$K_x\f$ for
     * the common vertex case.
     */
    Table<dim, Tensor<1, spacedim, RangeNumberType>> kx_normals_common_vertex;
    /**
     * Normal vector at each quadrature point in the real cell \f$K_x\f$ for
     * the regular case.
     */
    Table<dim, Tensor<1, spacedim, RangeNumberType>> kx_normals_regular;

    /**
     * Coordinates in the real cell \f$K_x\f$ for each \f$k_3\f$ term and each
     * quadrature point for the same panel case.
     */
    Table<dim, Point<spacedim, RangeNumberType>> kx_quad_points_same_panel;
    /**
     * Coordinates in the real cell \f$K_x\f$ for each \f$k_3\f$ term and each
     * quadrature point for the common edge case.
     */
    Table<dim, Point<spacedim, RangeNumberType>> kx_quad_points_common_edge;
    /**
     * Coordinates in the real cell \f$K_x\f$ for each \f$k_3\f$ term and each
     * quadrature point for the common vertex case.
     */
    Table<dim, Point<spacedim, RangeNumberType>> kx_quad_points_common_vertex;
    /**
     * Coordinates in the real cell \f$K_x\f$ for each \f$k_3\f$ term and each
     * quadrature point for the regular case.
     */
    Table<dim, Point<spacedim, RangeNumberType>> kx_quad_points_regular;


    /**
     * Jacobian from the unit cell to the real cell \f$K_y\f$ for each
     * \f$k_3\f$ term and at each quadrature point for the same panel case.
     */
    Table<dim, RangeNumberType> ky_jacobians_same_panel;
    /**
     * Jacobian from the unit cell to the real cell \f$K_y\f$ for each
     * \f$k_3\f$ term and at each quadrature point for the common edge case.
     */
    Table<dim, RangeNumberType> ky_jacobians_common_edge;
    /**
     * Jacobian from the unit cell to the real cell \f$K_y\f$ for each
     * \f$k_3\f$ term and at each quadrature point for the common vertex case.
     */
    Table<dim, RangeNumberType> ky_jacobians_common_vertex;
    /**
     * Jacobian from the unit cell to the real cell \f$K_y\f$ for each
     * \f$k_3\f$ term and at each quadrature point for the regular case.
     */
    Table<dim, RangeNumberType> ky_jacobians_regular;

    /**
     * Normal vector at each quadrature point in the real cell \f$K_y\f$ for
     * the same panel case.
     */
    Table<dim, Tensor<1, spacedim, RangeNumberType>> ky_normals_same_panel;
    /**
     * Normal vector at each quadrature point in the real cell \f$K_y\f$ for
     * the common edge case.
     */
    Table<dim, Tensor<1, spacedim, RangeNumberType>> ky_normals_common_edge;
    /**
     * Normal vector at each quadrature point in the real cell \f$K_y\f$ for
     * the common vertex case.
     */
    Table<dim, Tensor<1, spacedim, RangeNumberType>> ky_normals_common_vertex;
    /**
     * Normal vector at each quadrature point in the real cell \f$K_y\f$ for
     * the regular case.
     */
    Table<dim, Tensor<1, spacedim, RangeNumberType>> ky_normals_regular;

    /**
     * Coordinates in the real cell \f$K_y\f$ for each \f$k_3\f$ term and each
     * quadrature point for the same panel case.
     */
    Table<dim, Point<spacedim, RangeNumberType>> ky_quad_points_same_panel;
    /**
     * Coordinates in the real cell \f$K_y\f$ for each \f$k_3\f$ term and each
     * quadrature point for the common edge case.
     */
    Table<dim, Point<spacedim, RangeNumberType>> ky_quad_points_common_edge;
    /**
     * Coordinates in the real cell \f$K_y\f$ for each \f$k_3\f$ term and each
     * quadrature point for the common vertex case.
     */
    Table<dim, Point<spacedim, RangeNumberType>> ky_quad_points_common_vertex;
    /**
     * Coordinates in the real cell \f$K_y\f$ for each \f$k_3\f$ term and each
     * quadrature point for the regular case.
     */
    Table<dim, Point<spacedim, RangeNumberType>> ky_quad_points_regular;

    /**
     * Constructor
     *
     * @param kx_fe
     * @param ky_fe
     * @param bem_values
     */
    PairCellWiseScratchData(const FiniteElement<dim, spacedim> &kx_fe,
                            const FiniteElement<dim, spacedim> &ky_fe,
                            const BEMValues<dim, spacedim> &    bem_values)
      : common_vertex_dof_indices(0)
      , kx_support_points_in_default_dof_order(kx_fe.dofs_per_cell)
      , ky_support_points_in_default_dof_order(ky_fe.dofs_per_cell)
      , kx_support_points_permuted(kx_fe.dofs_per_cell)
      , ky_support_points_permuted(ky_fe.dofs_per_cell)
      , kx_local_dof_indices_in_default_dof_order(kx_fe.dofs_per_cell)
      , ky_local_dof_indices_in_default_dof_order(ky_fe.dofs_per_cell)
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
      common_vertex_dof_indices.reserve(GeometryInfo<dim>::vertices_per_cell);

      // Polynomial space inverse numbering for recovering the lexicographic
      // order.
      const FE_Poly_short &kx_fe_poly =
        dynamic_cast<const FE_Poly_short &>(kx_fe);
      const FE_Poly_short &ky_fe_poly =
        dynamic_cast<const FE_Poly_short &>(ky_fe);

      kx_fe_poly_space_numbering_inverse =
        kx_fe_poly.get_poly_space_numbering_inverse();
      ky_fe_poly_space_numbering_inverse =
        ky_fe_poly.get_poly_space_numbering_inverse();

      generate_backward_dof_permutation(
        ky_fe, 0, ky_fe_reversed_poly_space_numbering_inverse);
    }


    /**
     * Copy constructor
     *
     * @param scratch
     */
    PairCellWiseScratchData(
      const PairCellWiseScratchData<dim, spacedim, RangeNumberType> &scratch)
      : common_vertex_dof_indices(scratch.common_vertex_dof_indices)
      , kx_support_points_in_default_dof_order(
          scratch.kx_support_points_in_default_dof_order)
      , ky_support_points_in_default_dof_order(
          scratch.ky_support_points_in_default_dof_order)
      , kx_support_points_permuted(scratch.kx_support_points_permuted)
      , ky_support_points_permuted(scratch.ky_support_points_permuted)
      , kx_local_dof_indices_in_default_dof_order(
          scratch.kx_local_dof_indices_in_default_dof_order)
      , ky_local_dof_indices_in_default_dof_order(
          scratch.ky_local_dof_indices_in_default_dof_order)
      , kx_fe_poly_space_numbering_inverse(
          scratch.kx_fe_poly_space_numbering_inverse)
      , ky_fe_poly_space_numbering_inverse(
          scratch.ky_fe_poly_space_numbering_inverse)
      , ky_fe_reversed_poly_space_numbering_inverse(
          scratch.ky_fe_reversed_poly_space_numbering_inverse)
      , kx_local_dof_permutation(scratch.kx_local_dof_permutation)
      , ky_local_dof_permutation(scratch.ky_local_dof_permutation)
      , kx_jacobians_same_panel(scratch.kx_jacobians_same_panel)
      , kx_jacobians_common_edge(scratch.kx_jacobians_common_edge)
      , kx_jacobians_common_vertex(scratch.kx_jacobians_common_vertex)
      , kx_jacobians_regular(scratch.kx_jacobians_regular)
      , kx_normals_same_panel(scratch.kx_normals_same_panel)
      , kx_normals_common_edge(scratch.kx_normals_common_edge)
      , kx_normals_common_vertex(scratch.kx_normals_common_vertex)
      , kx_normals_regular(scratch.kx_normals_regular)
      , kx_quad_points_same_panel(scratch.kx_quad_points_same_panel)
      , kx_quad_points_common_edge(scratch.kx_quad_points_common_edge)
      , kx_quad_points_common_vertex(scratch.kx_quad_points_common_vertex)
      , kx_quad_points_regular(scratch.kx_quad_points_regular)
      , ky_jacobians_same_panel(scratch.ky_jacobians_same_panel)
      , ky_jacobians_common_edge(scratch.ky_jacobians_common_edge)
      , ky_jacobians_common_vertex(scratch.ky_jacobians_common_vertex)
      , ky_jacobians_regular(scratch.ky_jacobians_regular)
      , ky_normals_same_panel(scratch.ky_normals_same_panel)
      , ky_normals_common_edge(scratch.ky_normals_common_edge)
      , ky_normals_common_vertex(scratch.ky_normals_common_vertex)
      , ky_normals_regular(scratch.ky_normals_regular)
      , ky_quad_points_same_panel(scratch.ky_quad_points_same_panel)
      , ky_quad_points_common_edge(scratch.ky_quad_points_common_edge)
      , ky_quad_points_common_vertex(scratch.ky_quad_points_common_vertex)
      , ky_quad_points_regular(scratch.ky_quad_points_regular)
    {}
  };


  template <int dim, int spacedim = dim, typename RangeNumberType = double>
  struct PairCellWisePerTaskData
  {
    /**
     * Local matrix for the pair of cells to be assembled into the global full
     * matrix representation of the boundary integral operator.
     *
     * \comment{Therefore, this data field is only defined for verification.}
     */
    FullMatrix<RangeNumberType> local_pair_cell_matrix;

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

    /**
     * Constructor
     *
     * @param kx_fe
     * @param ky_fe
     */
    PairCellWisePerTaskData(const FiniteElement<dim, spacedim> &kx_fe,
                            const FiniteElement<dim, spacedim> &ky_fe)
      : local_pair_cell_matrix(kx_fe.dofs_per_cell, ky_fe.dofs_per_cell)
      , kx_local_dof_indices_permuted(kx_fe.dofs_per_cell)
      , ky_local_dof_indices_permuted(ky_fe.dofs_per_cell)
    {}


    /**
     * Copy constructor
     *
     * @param task_data
     */
    PairCellWisePerTaskData(
      const PairCellWisePerTaskData<dim, spacedim, RangeNumberType> &task_data)
      : local_pair_cell_matrix(task_data.local_pair_cell_matrix)
      , kx_local_dof_indices_permuted(task_data.kx_local_dof_indices_permuted)
      , ky_local_dof_indices_permuted(task_data.ky_local_dof_indices_permuted)
    {}
  };
} // namespace IdeoBEM

#endif /* INCLUDE_BEM_VALUES_H_ */
