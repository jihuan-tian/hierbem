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
#include <deal.II/fe/fe_tools.h>

#include "bem_tools.h"
#include "quadrature.templates.h"
#include "sauter_quadrature.h"

using namespace dealii;

namespace LaplaceBEM
{
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
     * the 2nd dimension is the index for \f$k_3\f$ terms; the 3rd dimension is the
     * quadrature point number. Refer to @p BEMValues::kx_shape_value_table_for_same_panel.
     * @param ky_shape_value_table the 1st dimension is the shape function hierarchical numbering;
     * the 2nd dimension is the index for \f$k_3\f$ terms; the 3rd dimension is the
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
     * the 2nd dimension is the index for \f$k_3\f$ terms; the 3rd dimension is the
     * quadrature point number. Refer to @p BEMValues::kx_shape_value_table_for_common_edge.
     * @param ky_shape_value_table The 1st dimension is the shape function hierarchical numbering;
     * the 2nd dimension is the index for \f$k_3\f$ terms; the 3rd dimension is the
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
     * Reference to 4D Sauter quadrature rule for the case that \f$K_x \equiv K_y\f$.
     */
    const QGauss<4> &quad_rule_for_same_panel;
    /**
     * Reference to 4D Sauter quadrature rule for the case that \f$K_x\f$ and \f$K_y\f$
     * share a common edge.
     */
    const QGauss<4> &quad_rule_for_common_edge;
    /**
     * Reference to 4D Sauter quadrature rule for the case that \f$K_x\f$ and \f$K_y\f$
     * share a common vertex.
     */
    const QGauss<4> &quad_rule_for_common_vertex;
    /**
     * Reference to 4D Sauter quadrature rule for the case that \f$K_x\f$ and \f$K_y\f$ are
     * separated.
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
     * Data table of shape function values for \f$K_x\f$ in the common edge case.
     * It has three dimensions:
     * 1. shape function index: size=@p dofs_per_cell
     * 2. \f$k_3\f$ term index: size=6
     * 3. Quadrature point index: size=number of quadrature points
     */
    Table<3, RangeNumberType> kx_shape_value_table_for_common_edge;
    /**
     * Data table of shape function values for \f$K_y\f$ in the common edge case.
     * It has three dimensions:
     * 1. shape function index: size=@p dofs_per_cell
     * 2. \f$k_3\f$ term index: size=6
     * 3. Quadrature point index: size=number of quadrature points
     */
    Table<3, RangeNumberType> ky_shape_value_table_for_common_edge;
    /**
     * Data table of shape function values for \f$K_x\f$ in the common vertex case.
     * It has three dimensions:
     * 1. shape function index: size=@p dofs_per_cell
     * 2. \f$k_3\f$ term index: size=4
     * 3. Quadrature point index: size=number of quadrature points
     */
    Table<3, RangeNumberType> kx_shape_value_table_for_common_vertex;
    /**
     * Data table of shape function values for \f$K_y\f$ in the common vertex case.
     * It has three dimensions:
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
     * Data table of shape function's gradient values for \f$K_x\f$ in the same panel case.
     * It has two dimensions:
     * 1. \f$k_3\f$ term index: size=8
     * 2. Quadrature point index: size=number of quadrature points
     * N.B. Each data item in the table is itself a matrix with the dimension
     * \f$dofs_per_cell*spacedim\f$.
     */
    Table<2, FullMatrix<RangeNumberType>>
      kx_shape_grad_matrix_table_for_same_panel;
    /**
     * Data table of shape function's gradient values for \f$K_y\f$ in the same panel case.
     * It has two dimensions:
     * 1. \f$k_3\f$ term index: size=8
     * 2. Quadrature point index: size=number of quadrature points
     * N.B. Each data item in the table is itself a matrix with the dimension
     * \f$dofs_per_cell*spacedim\f$.
     */
    Table<2, FullMatrix<RangeNumberType>>
      ky_shape_grad_matrix_table_for_same_panel;
    /**
     * Data table of shape function's gradient values for \f$K_x\f$ in the common edge case.
     * It has two dimensions:
     * 1. \f$k_3\f$ term index: size=6
     * 2. Quadrature point index: size=number of quadrature points
     * N.B. Each data item in the table is itself a matrix with the dimension
     * \f$dofs_per_cell*spacedim\f$.
     */
    Table<2, FullMatrix<RangeNumberType>>
      kx_shape_grad_matrix_table_for_common_edge;
    /**
     * Data table of shape function's gradient values for \f$K_y\f$ in the common edge case.
     * It has two dimensions:
     * 1. \f$k_3\f$ term index: size=6
     * 2. Quadrature point index: size=number of quadrature points
     * N.B. Each data item in the table is itself a matrix with the dimension
     * \f$dofs_per_cell*spacedim\f$.
     */
    Table<2, FullMatrix<RangeNumberType>>
      ky_shape_grad_matrix_table_for_common_edge;
    /**
     * Data table of shape function's gradient values for \f$K_x\f$ in the common vertex case.
     * It has two dimensions:
     * 1. \f$k_3\f$ term index: size=4
     * 2. Quadrature point index: size=number of quadrature points
     * N.B. Each data item in the table is itself a matrix with the dimension
     * \f$dofs_per_cell*spacedim\f$.
     */
    Table<2, FullMatrix<RangeNumberType>>
      kx_shape_grad_matrix_table_for_common_vertex;
    /**
     * Data table of shape function's gradient values for \f$K_y\f$ in the common vertex case.
     * It has two dimensions:
     * 1. \f$k_3\f$ term index: size=4
     * 2. Quadrature point index: size=number of quadrature points
     * N.B. Each data item in the table is itself a matrix with the dimension
     * \f$dofs_per_cell*spacedim\f$.
     */
    Table<2, FullMatrix<RangeNumberType>>
      ky_shape_grad_matrix_table_for_common_vertex;
    /**
     * Data table of shape function's gradient values for \f$K_x\f$ in the regular case.
     * It has two dimensions:
     * 1. \f$k_3\f$ term index: size=1
     * 2. Quadrature point index: size=number of quadrature points
     * N.B. Each data item in the table is itself a matrix with the dimension
     * \f$dofs_per_cell*spacedim\f$.
     */
    Table<2, FullMatrix<RangeNumberType>>
      kx_shape_grad_matrix_table_for_regular;
    /**
     * Data table of shape function's gradient values for \f$K_y\f$ in the regular case.
     * It has two dimensions:
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

    std::vector<unsigned int> kx_poly_space_inverse_numbering(
      FETools::lexicographic_to_hierarchic_numbering(kx_fe));
    std::vector<unsigned int> ky_poly_space_inverse_numbering(
      FETools::lexicographic_to_hierarchic_numbering(ky_fe));

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

    std::vector<unsigned int> kx_poly_space_inverse_numbering(
      FETools::lexicographic_to_hierarchic_numbering(kx_fe));
    std::vector<unsigned int> ky_poly_space_inverse_numbering(
      FETools::lexicographic_to_hierarchic_numbering(ky_fe));

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

    std::vector<unsigned int> kx_poly_space_inverse_numbering(
      FETools::lexicographic_to_hierarchic_numbering(kx_fe));
    std::vector<unsigned int> ky_poly_space_inverse_numbering(
      FETools::lexicographic_to_hierarchic_numbering(ky_fe));

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

    std::vector<unsigned int> kx_poly_space_inverse_numbering(
      FETools::lexicographic_to_hierarchic_numbering(kx_fe));
    std::vector<unsigned int> ky_poly_space_inverse_numbering(
      FETools::lexicographic_to_hierarchic_numbering(ky_fe));

    // Iterate over each quadrature point.
    for (unsigned int q = 0; q < n_q_points; q++)
      {
        // Transform the quadrature point in the parametric space to the
        // unit cells of \f$K_x\f$ and \f$K_y\f$.
        sauter_regular_parametric_coords_to_unit_cells(quad_points[q],
                                                       kx_quad_point,
                                                       ky_quad_point);

        // Iterate over each shape function on the unit cell of \f$K_x\f$ and
        // evaluate it at <code>kx_quad_point</code>. N.B. The shape
        // functions are in the default hierarchical order.
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
            // <code>kx_quad_point</code> in  \f$K_x\f$. Matrix rows correspond to
            // shape functions which are in the lexicographic order.
            kx_shape_grad_matrix_table(k, q) =
              BEMTools::shape_grad_matrix_in_lexicographic_order(kx_fe,
                                                                 kx_quad_point);
            // Calculate the gradient matrix evaluated at
            // <code>ky_quad_point</code> in  \f$K_y\f$. Matrix rows correspond to
            // shape functions which are in the lexicographic order.
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
            // <code>kx_quad_point</code> in  \f$K_x\f$. Matrix rows correspond to
            // shape functions which are in the lexicographic order.
            kx_shape_grad_matrix_table(k, q) =
              BEMTools::shape_grad_matrix_in_lexicographic_order(kx_fe,
                                                                 kx_quad_point);
            // Calculate the gradient matrix evaluated at
            // <code>ky_quad_point</code> in  \f$K_y\f$. Matrix rows correspond to
            // shape functions which are in the lexicographic order.
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
            // <code>kx_quad_point</code> in  \f$K_x\f$. Matrix rows correspond to
            // shape functions which are in the lexicographic order.
            kx_shape_grad_matrix_table(k, q) =
              BEMTools::shape_grad_matrix_in_lexicographic_order(kx_fe,
                                                                 kx_quad_point);
            // Calculate the gradient matrix evaluated at
            // <code>ky_quad_point</code> in  \f$K_y\f$. Matrix rows correspond to
            // shape functions which are in the lexicographic order.
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
} // namespace LaplaceBEM

#endif /* INCLUDE_BEM_VALUES_H_ */
