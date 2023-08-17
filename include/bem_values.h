/**
 * @file bem_values.h
 * @brief
 *
 * @date 2022-02-23
 * @author Jihuan Tian
 */
#ifndef INCLUDE_BEM_VALUES_H_
#define INCLUDE_BEM_VALUES_H_

#include <deal.II/base/point.h>
#include <deal.II/base/table.h>
#include <deal.II/base/table_indices.h>
#include <deal.II/base/timer.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_data.h>
#include <deal.II/fe/fe_tools.h>

#include <tbb/tbb_thread.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include "bem_tools.hcu"
#include "cpu_table.h"
#include "lapack_full_matrix_ext.h"
#include "sauter_quadrature_tools.h"

namespace HierBEM
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
    using FE_Poly_short = FE_Poly<dim, spacedim>;

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
    BEMValues(
      const FiniteElement<dim, spacedim>                    &kx_fe,
      const FiniteElement<dim, spacedim>                    &ky_fe,
      typename MappingQGeneric<dim, spacedim>::InternalData &kx_mapping_data,
      typename MappingQGeneric<dim, spacedim>::InternalData &ky_mapping_data,
      const QGauss<dim * 2> &quad_rule_for_same_panel,
      const QGauss<dim * 2> &quad_rule_for_common_edge,
      const QGauss<dim * 2> &quad_rule_for_common_vertex,
      const QGauss<dim * 2> &quad_rule_for_regular);


    /**
     * Copy constructor for class @p BEMValues is deleted.
     */
    BEMValues(const BEMValues<dim, spacedim, RangeNumberType> &bem_values) =
      delete;


    /**
     * Calculate the table storing shape function values and derivatives for
     * both finite element and mapping objects at Sauter quadrature points for
     * the same panel case.
     */
    void
    shape_function_values_same_panel();


    /**
     * Calculate the table storing shape function values and derivatives for
     * both finite element and mapping objects at Sauter quadrature points for
     * the common edge case.
     */
    void
    shape_function_values_common_edge();


    /**
     * Calculate the table storing shape function values and derivatives for
     * both finite element and mapping objects at Sauter quadrature points for
     * the common vertex case.
     */
    void
    shape_function_values_common_vertex();


    /**
     * Calculate the table storing shape function values and derivatives for
     * both finite element and mapping objects at Sauter quadrature points for
     * the regular case.
     */
    void
    shape_function_values_regular();

    /**
     * Reference to finite element on the field cell \f$K_x\f$.
     */
    const FiniteElement<dim, spacedim> &kx_fe;
    /**
     * Reference to finite element on the field cell \f$K_y\f$.
     */
    const FiniteElement<dim, spacedim> &ky_fe;
    /**
     * Reference to the internal data in the mapping of \f$K_x\f$.
     */
    typename MappingQGeneric<dim, spacedim>::InternalData &kx_mapping_data;
    /**
     * Reference to the internal data in the mapping of \f$K_y\f$.
     */
    typename MappingQGeneric<dim, spacedim>::InternalData &ky_mapping_data;
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
     * Data table of finite element shape function values for \f$K_x\f$ in the
     * same panel case. It has three dimensions:
     * 1. shape function index: size=@p dofs_per_cell
     * 2. \f$k_3\f$ term index: size=8
     * 3. Quadrature point index: size=number of quadrature points
     */
    Table<3, RangeNumberType> kx_shape_value_table_for_same_panel;
    /**
     * Data table of finite element shape function values for \f$K_y\f$ in the
     * same panel case. It has three dimensions:
     * 1. shape function index: size=@p dofs_per_cell
     * 2. \f$k_3\f$ term index: size=8
     * 3. Quadrature point index: size=number of quadrature points
     */
    Table<3, RangeNumberType> ky_shape_value_table_for_same_panel;
    /**
     * Data table of finite element shape function values for \f$K_x\f$ in the
     * common edge case. It has three dimensions:
     * 1. shape function index: size=@p dofs_per_cell
     * 2. \f$k_3\f$ term index: size=6
     * 3. Quadrature point index: size=number of quadrature points
     */
    Table<3, RangeNumberType> kx_shape_value_table_for_common_edge;
    /**
     * Data table of finite element shape function values for \f$K_y\f$ in the
     * common edge case. It has three dimensions:
     * 1. shape function index: size=@p dofs_per_cell
     * 2. \f$k_3\f$ term index: size=6
     * 3. Quadrature point index: size=number of quadrature points
     */
    Table<3, RangeNumberType> ky_shape_value_table_for_common_edge;
    /**
     * Data table of finite element shape function values for \f$K_x\f$ in the
     * common vertex case. It has three dimensions:
     * 1. shape function index: size=@p dofs_per_cell
     * 2. \f$k_3\f$ term index: size=4
     * 3. Quadrature point index: size=number of quadrature points
     */
    Table<3, RangeNumberType> kx_shape_value_table_for_common_vertex;
    /**
     * Data table of finite element shape function values for \f$K_y\f$ in the
     * common vertex case. It has three dimensions:
     * 1. shape function index: size=@p dofs_per_cell
     * 2. \f$k_3\f$ term index: size=4
     * 3. Quadrature point index: size=number of quadrature points
     */
    Table<3, RangeNumberType> ky_shape_value_table_for_common_vertex;
    /**
     * Data table of finite element shape function values for \f$K_x\f$ in the
     * regular case. It has three dimensions:
     * 1. shape function index: size=@p dofs_per_cell
     * 2. \f$k_3\f$ term index: size=1
     * 3. Quadrature point index: size=number of quadrature points
     */
    Table<3, RangeNumberType> kx_shape_value_table_for_regular;
    /**
     * Data table of finite element shape function values for \f$K_y\f$ in the
     * regular case. It has three dimensions:
     * 1. shape function index: size=@p dofs_per_cell
     * 2. \f$k_3\f$ term index: size=1
     * 3. Quadrature point index: size=number of quadrature points
     */
    Table<3, RangeNumberType> ky_shape_value_table_for_regular;

    /**
     * Data table of mapping shape function values for \f$K_x\f$ in the
     * same panel case. It has three dimensions:
     * 1. shape function index: size=@p dofs_per_cell
     * 2. \f$k_3\f$ term index: size=8
     * 3. Quadrature point index: size=number of quadrature points
     */
    Table<3, RangeNumberType> kx_mapping_shape_value_table_for_same_panel;
    /**
     * Data table of mapping shape function values for \f$K_y\f$ in the
     * same panel case. It has three dimensions:
     * 1. shape function index: size=@p dofs_per_cell
     * 2. \f$k_3\f$ term index: size=8
     * 3. Quadrature point index: size=number of quadrature points
     */
    Table<3, RangeNumberType> ky_mapping_shape_value_table_for_same_panel;
    /**
     * Data table of mapping shape function values for \f$K_x\f$ in the
     * common edge case. It has three dimensions:
     * 1. shape function index: size=@p dofs_per_cell
     * 2. \f$k_3\f$ term index: size=6
     * 3. Quadrature point index: size=number of quadrature points
     */
    Table<3, RangeNumberType> kx_mapping_shape_value_table_for_common_edge;
    /**
     * Data table of mapping shape function values for \f$K_y\f$ in the
     * common edge case. It has three dimensions:
     * 1. shape function index: size=@p dofs_per_cell
     * 2. \f$k_3\f$ term index: size=6
     * 3. Quadrature point index: size=number of quadrature points
     */
    Table<3, RangeNumberType> ky_mapping_shape_value_table_for_common_edge;
    /**
     * Data table of mapping shape function values for \f$K_x\f$ in the
     * common vertex case. It has three dimensions:
     * 1. shape function index: size=@p dofs_per_cell
     * 2. \f$k_3\f$ term index: size=4
     * 3. Quadrature point index: size=number of quadrature points
     */
    Table<3, RangeNumberType> kx_mapping_shape_value_table_for_common_vertex;
    /**
     * Data table of mapping shape function values for \f$K_y\f$ in the
     * common vertex case. It has three dimensions:
     * 1. shape function index: size=@p dofs_per_cell
     * 2. \f$k_3\f$ term index: size=4
     * 3. Quadrature point index: size=number of quadrature points
     */
    Table<3, RangeNumberType> ky_mapping_shape_value_table_for_common_vertex;
    /**
     * Data table of mapping shape function values for \f$K_x\f$ in the
     * regular case. It has three dimensions:
     * 1. shape function index: size=@p dofs_per_cell
     * 2. \f$k_3\f$ term index: size=1
     * 3. Quadrature point index: size=number of quadrature points
     */
    Table<3, RangeNumberType> kx_mapping_shape_value_table_for_regular;
    /**
     * Data table of mapping shape function values for \f$K_y\f$ in the
     * regular case. It has three dimensions:
     * 1. shape function index: size=@p dofs_per_cell
     * 2. \f$k_3\f$ term index: size=1
     * 3. Quadrature point index: size=number of quadrature points
     */
    Table<3, RangeNumberType> ky_mapping_shape_value_table_for_regular;

    /**
     * Data table of finite element shape function's gradient values for
     * \f$K_x\f$ in the same panel case. It has two dimensions:
     * 1. \f$k_3\f$ term index: size=8
     * 2. Quadrature point index: size=number of quadrature points
     * N.B. Each data item in the table is itself a matrix with the dimension
     * \f$dofs_per_cell*dim\f$.
     */
    Table<2, LAPACKFullMatrixExt<RangeNumberType>>
      kx_shape_grad_matrix_table_for_same_panel;
    /**
     * Data table of finite element shape function's gradient values for
     * \f$K_y\f$ in the same panel case. It has two dimensions:
     * 1. \f$k_3\f$ term index: size=8
     * 2. Quadrature point index: size=number of quadrature points
     * N.B. Each data item in the table is itself a matrix with the dimension
     * \f$dofs_per_cell*dim\f$.
     */
    Table<2, LAPACKFullMatrixExt<RangeNumberType>>
      ky_shape_grad_matrix_table_for_same_panel;
    /**
     * Data table of finite element shape function's gradient values for
     * \f$K_x\f$ in the common edge case. It has two dimensions:
     * 1. \f$k_3\f$ term index: size=6
     * 2. Quadrature point index: size=number of quadrature points
     * N.B. Each data item in the table is itself a matrix with the dimension
     * \f$dofs_per_cell*dim\f$.
     */
    Table<2, LAPACKFullMatrixExt<RangeNumberType>>
      kx_shape_grad_matrix_table_for_common_edge;
    /**
     * Data table of finite element shape function's gradient values for
     * \f$K_y\f$ in the common edge case. It has two dimensions:
     * 1. \f$k_3\f$ term index: size=6
     * 2. Quadrature point index: size=number of quadrature points
     * N.B. Each data item in the table is itself a matrix with the dimension
     * \f$dofs_per_cell*dim\f$.
     */
    Table<2, LAPACKFullMatrixExt<RangeNumberType>>
      ky_shape_grad_matrix_table_for_common_edge;
    /**
     * Data table of finite element shape function's gradient values for
     * \f$K_x\f$ in the common vertex case. It has two dimensions:
     * 1. \f$k_3\f$ term index: size=4
     * 2. Quadrature point index: size=number of quadrature points
     * N.B. Each data item in the table is itself a matrix with the dimension
     * \f$dofs_per_cell*dim\f$.
     */
    Table<2, LAPACKFullMatrixExt<RangeNumberType>>
      kx_shape_grad_matrix_table_for_common_vertex;
    /**
     * Data table of finite element shape function's gradient values for
     * \f$K_y\f$ in the common vertex case. It has two dimensions:
     * 1. \f$k_3\f$ term index: size=4
     * 2. Quadrature point index: size=number of quadrature points
     * N.B. Each data item in the table is itself a matrix with the dimension
     * \f$dofs_per_cell*dim\f$.
     */
    Table<2, LAPACKFullMatrixExt<RangeNumberType>>
      ky_shape_grad_matrix_table_for_common_vertex;
    /**
     * Data table of finite element shape function's gradient values for
     * \f$K_x\f$ in the regular case. It has two dimensions:
     * 1. \f$k_3\f$ term index: size=1
     * 2. Quadrature point index: size=number of quadrature points
     * N.B. Each data item in the table is itself a matrix with the dimension
     * \f$dofs_per_cell*dim\f$.
     */
    Table<2, LAPACKFullMatrixExt<RangeNumberType>>
      kx_shape_grad_matrix_table_for_regular;
    /**
     * Data table of finite element shape function's gradient values for
     * \f$K_y\f$ in the regular case. It has two dimensions:
     * 1. \f$k_3\f$ term index: size=1
     * 2. Quadrature point index: size=number of quadrature points
     * N.B. Each data item in the table is itself a matrix with the dimension
     * \f$dofs_per_cell*dim\f$.
     */
    Table<2, LAPACKFullMatrixExt<RangeNumberType>>
      ky_shape_grad_matrix_table_for_regular;

    /**
     * Data table of mapping shape function's gradient values for
     * \f$K_x\f$ in the same panel case. It has two dimensions:
     * 1. \f$k_3\f$ term index: size=8
     * 2. Quadrature point index: size=number of quadrature points
     * N.B. Each data item in the table is itself a matrix with the dimension
     * \f$MappingQGeneric::InternalData.n_shape_functions*dim\f$.
     */
    Table<2, LAPACKFullMatrixExt<RangeNumberType>>
      kx_mapping_shape_grad_matrix_table_for_same_panel;
    /**
     * Data table of mapping shape function's gradient values for
     * \f$K_y\f$ in the same panel case. It has two dimensions:
     * 1. \f$k_3\f$ term index: size=8
     * 2. Quadrature point index: size=number of quadrature points
     * N.B. Each data item in the table is itself a matrix with the dimension
     * \f$MappingQGeneric::InternalData.n_shape_functions*dim\f$.
     */
    Table<2, LAPACKFullMatrixExt<RangeNumberType>>
      ky_mapping_shape_grad_matrix_table_for_same_panel;
    /**
     * Data table of mapping shape function's gradient values for
     * \f$K_x\f$ in the common edge case. It has two dimensions:
     * 1. \f$k_3\f$ term index: size=6
     * 2. Quadrature point index: size=number of quadrature points
     * N.B. Each data item in the table is itself a matrix with the dimension
     * \f$MappingQGeneric::InternalData.n_shape_functions*dim\f$.
     */
    Table<2, LAPACKFullMatrixExt<RangeNumberType>>
      kx_mapping_shape_grad_matrix_table_for_common_edge;
    /**
     * Data table of mapping shape function's gradient values for
     * \f$K_y\f$ in the common edge case. It has two dimensions:
     * 1. \f$k_3\f$ term index: size=6
     * 2. Quadrature point index: size=number of quadrature points
     * N.B. Each data item in the table is itself a matrix with the dimension
     * \f$MappingQGeneric::InternalData.n_shape_functions*dim\f$.
     */
    Table<2, LAPACKFullMatrixExt<RangeNumberType>>
      ky_mapping_shape_grad_matrix_table_for_common_edge;
    /**
     * Data table of mapping shape function's gradient values for
     * \f$K_x\f$ in the common vertex case. It has two dimensions:
     * 1. \f$k_3\f$ term index: size=4
     * 2. Quadrature point index: size=number of quadrature points
     * N.B. Each data item in the table is itself a matrix with the dimension
     * \f$MappingQGeneric::InternalData.n_shape_functions*dim\f$.
     */
    Table<2, LAPACKFullMatrixExt<RangeNumberType>>
      kx_mapping_shape_grad_matrix_table_for_common_vertex;
    /**
     * Data table of mapping shape function's gradient values for
     * \f$K_y\f$ in the common vertex case. It has two dimensions:
     * 1. \f$k_3\f$ term index: size=4
     * 2. Quadrature point index: size=number of quadrature points
     * N.B. Each data item in the table is itself a matrix with the dimension
     * \f$MappingQGeneric::InternalData.n_shape_functions*dim\f$.
     */
    Table<2, LAPACKFullMatrixExt<RangeNumberType>>
      ky_mapping_shape_grad_matrix_table_for_common_vertex;
    /**
     * Data table of mapping shape function's gradient values for
     * \f$K_x\f$ in the regular case. It has two dimensions:
     * 1. \f$k_3\f$ term index: size=1
     * 2. Quadrature point index: size=number of quadrature points
     * N.B. Each data item in the table is itself a matrix with the dimension
     * \f$MappingQGeneric::InternalData.n_shape_functions*dim\f$.
     */
    Table<2, LAPACKFullMatrixExt<RangeNumberType>>
      kx_mapping_shape_grad_matrix_table_for_regular;
    /**
     * Data table of mapping shape function's gradient values for
     * \f$K_y\f$ in the regular case. It has two dimensions:
     * 1. \f$k_3\f$ term index: size=1
     * 2. Quadrature point index: size=number of quadrature points
     * N.B. Each data item in the table is itself a matrix with the dimension
     * \f$MappingQGeneric::InternalData.n_shape_functions*dim\f$.
     */
    Table<2, LAPACKFullMatrixExt<RangeNumberType>>
      ky_mapping_shape_grad_matrix_table_for_regular;

    /**
     * Fill the data tables for the values and derivatives of finite element
     * shape functions and mapping object shape functions.
     */
    void
    fill_shape_function_value_tables();

  protected:
    /**
     * Initialize the data tables for the finite element shape function values.
     */
    void
    init_shape_value_tables();

    /**
     * Initialize the data tables for the mapping shape function values.
     */
    void
    init_mapping_shape_value_tables();

    /**
     * Initialize the data tables for the gradient values of finite element
     * shape functions.
     */
    void
    init_shape_grad_matrix_tables();

    /**
     * Initialize the data tables for the gradient values of shape functions in
     * the mapping object.
     */
    void
    init_mapping_shape_grad_matrix_tables();

    /**
     * Initialize matrices storing the gradient values of shape functions in the
     * mapping object.
     *
     * @param table
     * @param n_shape_functions
     */
    void
    init_internal_matrix_in_mapping_shape_grad_matrix_table(
      Table<2, LAPACKFullMatrixExt<RangeNumberType>> &table,
      const unsigned int                              n_shape_functions);
  };


  template <int dim, int spacedim, typename RangeNumberType>
  BEMValues<dim, spacedim, RangeNumberType>::BEMValues(
    const FiniteElement<dim, spacedim>                    &kx_fe,
    const FiniteElement<dim, spacedim>                    &ky_fe,
    typename MappingQGeneric<dim, spacedim>::InternalData &kx_mapping_data,
    typename MappingQGeneric<dim, spacedim>::InternalData &ky_mapping_data,
    const QGauss<dim * 2> &quad_rule_for_same_panel,
    const QGauss<dim * 2> &quad_rule_for_common_edge,
    const QGauss<dim * 2> &quad_rule_for_common_vertex,
    const QGauss<dim * 2> &quad_rule_for_regular)
    : kx_fe(kx_fe)
    , ky_fe(ky_fe)
    , kx_mapping_data(kx_mapping_data)
    , ky_mapping_data(ky_mapping_data)
    , quad_rule_for_same_panel(quad_rule_for_same_panel)
    , quad_rule_for_common_edge(quad_rule_for_common_edge)
    , quad_rule_for_common_vertex(quad_rule_for_common_vertex)
    , quad_rule_for_regular(quad_rule_for_regular)
  {
    init_shape_value_tables();
    init_mapping_shape_value_tables();
    init_shape_grad_matrix_tables();
    init_mapping_shape_grad_matrix_tables();
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
  BEMValues<dim, spacedim, RangeNumberType>::init_mapping_shape_value_tables()
  {
    kx_mapping_shape_value_table_for_same_panel.reinit(TableIndices<3>(
      kx_mapping_data.n_shape_functions, 8, quad_rule_for_same_panel.size()));
    ky_mapping_shape_value_table_for_same_panel.reinit(TableIndices<3>(
      ky_mapping_data.n_shape_functions, 8, quad_rule_for_same_panel.size()));

    kx_mapping_shape_value_table_for_common_edge.reinit(TableIndices<3>(
      kx_mapping_data.n_shape_functions, 6, quad_rule_for_common_edge.size()));
    ky_mapping_shape_value_table_for_common_edge.reinit(TableIndices<3>(
      ky_mapping_data.n_shape_functions, 6, quad_rule_for_common_edge.size()));

    kx_mapping_shape_value_table_for_common_vertex.reinit(
      TableIndices<3>(kx_mapping_data.n_shape_functions,
                      4,
                      quad_rule_for_common_vertex.size()));
    ky_mapping_shape_value_table_for_common_vertex.reinit(
      TableIndices<3>(ky_mapping_data.n_shape_functions,
                      4,
                      quad_rule_for_common_vertex.size()));

    kx_mapping_shape_value_table_for_regular.reinit(TableIndices<3>(
      kx_mapping_data.n_shape_functions, 1, quad_rule_for_regular.size()));
    ky_mapping_shape_value_table_for_regular.reinit(TableIndices<3>(
      ky_mapping_data.n_shape_functions, 1, quad_rule_for_regular.size()));
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
  BEMValues<dim, spacedim, RangeNumberType>::
    init_mapping_shape_grad_matrix_tables()
  {
    kx_mapping_shape_grad_matrix_table_for_same_panel.reinit(
      TableIndices<2>(8, quad_rule_for_same_panel.size()));
    init_internal_matrix_in_mapping_shape_grad_matrix_table(
      kx_mapping_shape_grad_matrix_table_for_same_panel,
      kx_mapping_data.n_shape_functions);

    ky_mapping_shape_grad_matrix_table_for_same_panel.reinit(
      TableIndices<2>(8, quad_rule_for_same_panel.size()));
    init_internal_matrix_in_mapping_shape_grad_matrix_table(
      ky_mapping_shape_grad_matrix_table_for_same_panel,
      ky_mapping_data.n_shape_functions);

    kx_mapping_shape_grad_matrix_table_for_common_edge.reinit(
      TableIndices<2>(6, quad_rule_for_common_edge.size()));
    init_internal_matrix_in_mapping_shape_grad_matrix_table(
      kx_mapping_shape_grad_matrix_table_for_common_edge,
      kx_mapping_data.n_shape_functions);

    ky_mapping_shape_grad_matrix_table_for_common_edge.reinit(
      TableIndices<2>(6, quad_rule_for_common_edge.size()));
    init_internal_matrix_in_mapping_shape_grad_matrix_table(
      ky_mapping_shape_grad_matrix_table_for_common_edge,
      ky_mapping_data.n_shape_functions);

    kx_mapping_shape_grad_matrix_table_for_common_vertex.reinit(
      TableIndices<2>(4, quad_rule_for_common_vertex.size()));
    init_internal_matrix_in_mapping_shape_grad_matrix_table(
      kx_mapping_shape_grad_matrix_table_for_common_vertex,
      kx_mapping_data.n_shape_functions);

    ky_mapping_shape_grad_matrix_table_for_common_vertex.reinit(
      TableIndices<2>(4, quad_rule_for_common_vertex.size()));
    init_internal_matrix_in_mapping_shape_grad_matrix_table(
      ky_mapping_shape_grad_matrix_table_for_common_vertex,
      ky_mapping_data.n_shape_functions);

    kx_mapping_shape_grad_matrix_table_for_regular.reinit(
      TableIndices<2>(1, quad_rule_for_regular.size()));
    init_internal_matrix_in_mapping_shape_grad_matrix_table(
      kx_mapping_shape_grad_matrix_table_for_regular,
      kx_mapping_data.n_shape_functions);

    ky_mapping_shape_grad_matrix_table_for_regular.reinit(
      TableIndices<2>(1, quad_rule_for_regular.size()));
    init_internal_matrix_in_mapping_shape_grad_matrix_table(
      ky_mapping_shape_grad_matrix_table_for_regular,
      ky_mapping_data.n_shape_functions);
  }


  template <int dim, int spacedim, typename RangeNumberType>
  void
  BEMValues<dim, spacedim, RangeNumberType>::
    init_internal_matrix_in_mapping_shape_grad_matrix_table(
      Table<2, LAPACKFullMatrixExt<RangeNumberType>> &table,
      const unsigned int                              n_shape_functions)
  {
    for (unsigned int i = 0; i < table.size(0); i++)
      for (unsigned int j = 0; j < table.size(1); j++)
        {
          table(TableIndices<2>(i, j)).reinit(n_shape_functions, dim);
        }
  }


  template <int dim, int spacedim, typename RangeNumberType>
  void
  BEMValues<dim, spacedim, RangeNumberType>::fill_shape_function_value_tables()
  {
    shape_function_values_same_panel();
    shape_function_values_common_edge();
    shape_function_values_common_vertex();
    shape_function_values_regular();
  }


  template <int dim, int spacedim, typename RangeNumberType>
  void
  BEMValues<dim, spacedim, RangeNumberType>::shape_function_values_same_panel()
  {
    const unsigned int kx_dofs_per_cell = kx_fe.dofs_per_cell;
    const unsigned int ky_dofs_per_cell = ky_fe.dofs_per_cell;
    const unsigned int kx_mapping_n_shape_functions =
      kx_mapping_data.n_shape_functions;
    const unsigned int ky_mapping_n_shape_functions =
      ky_mapping_data.n_shape_functions;
    const unsigned int n_q_points = quad_rule_for_same_panel.size();

    // Make assertion about the length for each dimension of the data table.
    AssertDimension(kx_shape_value_table_for_same_panel.size(0),
                    kx_dofs_per_cell);
    AssertDimension(kx_shape_value_table_for_same_panel.size(1), 8);
    AssertDimension(kx_shape_value_table_for_same_panel.size(2), n_q_points);

    AssertDimension(ky_shape_value_table_for_same_panel.size(0),
                    ky_dofs_per_cell);
    AssertDimension(ky_shape_value_table_for_same_panel.size(1), 8);
    AssertDimension(ky_shape_value_table_for_same_panel.size(2), n_q_points);

    AssertDimension(kx_mapping_shape_value_table_for_same_panel.size(0),
                    kx_mapping_n_shape_functions);
    AssertDimension(kx_mapping_shape_value_table_for_same_panel.size(1), 8);
    AssertDimension(kx_mapping_shape_value_table_for_same_panel.size(2),
                    n_q_points);

    AssertDimension(ky_mapping_shape_value_table_for_same_panel.size(0),
                    ky_mapping_n_shape_functions);
    AssertDimension(ky_mapping_shape_value_table_for_same_panel.size(1), 8);
    AssertDimension(ky_mapping_shape_value_table_for_same_panel.size(2),
                    n_q_points);

    AssertDimension(kx_shape_grad_matrix_table_for_same_panel.size(0), 8);
    AssertDimension(kx_shape_grad_matrix_table_for_same_panel.size(1),
                    n_q_points);

    AssertDimension(ky_shape_grad_matrix_table_for_same_panel.size(0), 8);
    AssertDimension(ky_shape_grad_matrix_table_for_same_panel.size(1),
                    n_q_points);

    AssertDimension(kx_mapping_shape_grad_matrix_table_for_same_panel.size(0),
                    8);
    AssertDimension(kx_mapping_shape_grad_matrix_table_for_same_panel.size(1),
                    n_q_points);

    AssertDimension(ky_mapping_shape_grad_matrix_table_for_same_panel.size(0),
                    8);
    AssertDimension(ky_mapping_shape_grad_matrix_table_for_same_panel.size(1),
                    n_q_points);

    /**
     * Initialize the internal data held in the mapping objects.
     */
    kx_mapping_data.shape_values.resize(kx_mapping_n_shape_functions *
                                        n_q_points);
    kx_mapping_data.shape_derivatives.resize(kx_mapping_n_shape_functions *
                                             n_q_points);
    ky_mapping_data.shape_values.resize(ky_mapping_n_shape_functions *
                                        n_q_points);
    ky_mapping_data.shape_derivatives.resize(ky_mapping_n_shape_functions *
                                             n_q_points);

    /**
     * Quadrature points in the Sauter's parametric space.
     */
    std::vector<Point<dim * 2>> quad_points =
      quad_rule_for_same_panel.get_points();

    /**
     * Get the polynomial space inverse numbering for accessing the shape
     * functions in the lexicographic order.
     *
     * \alert{Here I have adopted an assumption that the finite elements
     * are based on tensor product polynomials.}
     */
    const FE_Poly_short &kx_fe_poly =
      dynamic_cast<const FE_Poly_short &>(kx_fe);
    const FE_Poly_short &ky_fe_poly =
      dynamic_cast<const FE_Poly_short &>(ky_fe);

    std::vector<unsigned int> kx_poly_space_inverse_numbering(
      kx_fe_poly.get_poly_space_numbering_inverse());
    std::vector<unsigned int> ky_poly_space_inverse_numbering(
      ky_fe_poly.get_poly_space_numbering_inverse());

    /**
     * Quadrature points in the unit cells of \f$K_x\f$ and \f$K_y\f$
     * respectively.
     */
    std::vector<Point<dim>> kx_unit_quad_points(n_q_points);
    std::vector<Point<dim>> ky_unit_quad_points(n_q_points);

    // Iterate over each $k_3$ part.
    for (unsigned k = 0; k < 8; k++)
      {
        // Iterate over each quadrature point.
        for (unsigned int q = 0; q < n_q_points; q++)
          {
            // Transform the quadrature point in the parametric space to
            // the unit cells of \f$K_x\f$ and \f$K_y\f$.
            sauter_same_panel_parametric_coords_to_unit_cells(
              quad_points[q],
              k,
              kx_unit_quad_points[q],
              ky_unit_quad_points[q]);

            /**
             * Iterate over each finite element shape function in the
             * lexicographic order on the unit cell of \f$K_x\f$ and evaluate it
             * at the current quadrature point @p kx_unit_quad_points[q].
             */
            for (unsigned int s = 0; s < kx_dofs_per_cell; s++)
              {
                kx_shape_value_table_for_same_panel(TableIndices<3>(s, k, q)) =
                  kx_fe.shape_value(kx_poly_space_inverse_numbering[s],
                                    kx_unit_quad_points[q]);
              }

            /**
             * Iterate over each finite element shape function in the
             * lexicographic order on the unit cell of \f$K_y\f$ and evaluate it
             * at the current quadrature point @p ky_unit_quad_points[q].
             */
            for (unsigned int s = 0; s < ky_dofs_per_cell; s++)
              {
                ky_shape_value_table_for_same_panel(TableIndices<3>(s, k, q)) =
                  ky_fe.shape_value(ky_poly_space_inverse_numbering[s],
                                    ky_unit_quad_points[q]);
              }

            // Calculate the Jacobian matrix evaluated at
            // <code>kx_quad_point</code> in  \f$K_x\f$. Matrix rows correspond
            // to shape functions which are in the lexicographic order.
            kx_shape_grad_matrix_table_for_same_panel(TableIndices<2>(k, q)) =
              BEMTools::shape_grad_matrix_in_lexicographic_order(
                kx_fe, kx_unit_quad_points[q]);
            // Calculate the Jacobian matrix evaluated at
            // <code>ky_quad_point</code> in  \f$K_y\f$. Matrix rows correspond
            // to shape functions which are in the lexicographic order.
            ky_shape_grad_matrix_table_for_same_panel(TableIndices<2>(k, q)) =
              BEMTools::shape_grad_matrix_in_lexicographic_order(
                ky_fe, ky_unit_quad_points[q]);
          }

        /**
         * Compute mapping shape function values and their derivatives in batch.
         *
         * \alert{Even though the internally generated polynomials in the
         * mapping object are in the tensor product order, the shape function
         * values and derivatives are still in the hierarchic order. This can be
         * verified by checking the source code of
         * @p MappingQ<dim, spacedim>::InternalData::compute_shape_function_values.
         * (see
         * http://localhost/dealii-9.4.1-doc/mapping__q_8cc_source.html#l00271)
         *
         * However, this behavior is different from the documentation for the
         * function @p MappingQ< dim, spacedim >::InternalData::shape().}
         */
        // Get the numbering for accessing the support points in the
        // lexicographic ordering which are stored in the hierarchic ordering.
        const std::vector<unsigned int> kx_mapping_poly_space_inverse_numbering(
          FETools::lexicographic_to_hierarchic_numbering<dim>(
            kx_mapping_data.polynomial_degree));
        const std::vector<unsigned int> ky_mapping_poly_space_inverse_numbering(
          FETools::lexicographic_to_hierarchic_numbering<dim>(
            ky_mapping_data.polynomial_degree));

        kx_mapping_data.compute_shape_function_values(kx_unit_quad_points);
        ky_mapping_data.compute_shape_function_values(ky_unit_quad_points);

        for (unsigned int q = 0; q < n_q_points; q++)
          {
            for (unsigned int s = 0; s < kx_mapping_n_shape_functions; s++)
              {
                kx_mapping_shape_value_table_for_same_panel(
                  TableIndices<3>(s, k, q)) =
                  kx_mapping_data.shape(
                    q, kx_mapping_poly_space_inverse_numbering[s]);

                for (unsigned int d = 0; d < dim; d++)
                  {
                    kx_mapping_shape_grad_matrix_table_for_same_panel(
                      TableIndices<2>(k, q))(s, d) =
                      kx_mapping_data.derivative(
                        q, kx_mapping_poly_space_inverse_numbering[s])[d];
                  }
              }

            for (unsigned int s = 0; s < ky_mapping_n_shape_functions; s++)
              {
                ky_mapping_shape_value_table_for_same_panel(
                  TableIndices<3>(s, k, q)) =
                  ky_mapping_data.shape(
                    q, ky_mapping_poly_space_inverse_numbering[s]);

                for (unsigned int d = 0; d < dim; d++)
                  {
                    ky_mapping_shape_grad_matrix_table_for_same_panel(
                      TableIndices<2>(k, q))(s, d) =
                      ky_mapping_data.derivative(
                        q, ky_mapping_poly_space_inverse_numbering[s])[d];
                  }
              }
          }
      }
  }


  template <int dim, int spacedim, typename RangeNumberType>
  void
  BEMValues<dim, spacedim, RangeNumberType>::shape_function_values_common_edge()
  {
    const unsigned int kx_dofs_per_cell = kx_fe.dofs_per_cell;
    const unsigned int ky_dofs_per_cell = ky_fe.dofs_per_cell;
    const unsigned int kx_mapping_n_shape_functions =
      kx_mapping_data.n_shape_functions;
    const unsigned int ky_mapping_n_shape_functions =
      ky_mapping_data.n_shape_functions;
    const unsigned int n_q_points = quad_rule_for_common_edge.size();

    // Make assertion about the length for each dimension of the data table.
    AssertDimension(kx_shape_value_table_for_common_edge.size(0),
                    kx_dofs_per_cell);
    AssertDimension(kx_shape_value_table_for_common_edge.size(1), 6);
    AssertDimension(kx_shape_value_table_for_common_edge.size(2), n_q_points);

    AssertDimension(ky_shape_value_table_for_common_edge.size(0),
                    ky_dofs_per_cell);
    AssertDimension(ky_shape_value_table_for_common_edge.size(1), 6);
    AssertDimension(ky_shape_value_table_for_common_edge.size(2), n_q_points);

    AssertDimension(kx_mapping_shape_value_table_for_common_edge.size(0),
                    kx_mapping_n_shape_functions);
    AssertDimension(kx_mapping_shape_value_table_for_common_edge.size(1), 6);
    AssertDimension(kx_mapping_shape_value_table_for_common_edge.size(2),
                    n_q_points);

    AssertDimension(ky_mapping_shape_value_table_for_common_edge.size(0),
                    ky_mapping_n_shape_functions);
    AssertDimension(ky_mapping_shape_value_table_for_common_edge.size(1), 6);
    AssertDimension(ky_mapping_shape_value_table_for_common_edge.size(2),
                    n_q_points);

    AssertDimension(kx_shape_grad_matrix_table_for_common_edge.size(0), 6);
    AssertDimension(kx_shape_grad_matrix_table_for_common_edge.size(1),
                    n_q_points);

    AssertDimension(ky_shape_grad_matrix_table_for_common_edge.size(0), 6);
    AssertDimension(ky_shape_grad_matrix_table_for_common_edge.size(1),
                    n_q_points);

    AssertDimension(kx_mapping_shape_grad_matrix_table_for_common_edge.size(0),
                    6);
    AssertDimension(kx_mapping_shape_grad_matrix_table_for_common_edge.size(1),
                    n_q_points);

    AssertDimension(ky_mapping_shape_grad_matrix_table_for_common_edge.size(0),
                    6);
    AssertDimension(ky_mapping_shape_grad_matrix_table_for_common_edge.size(1),
                    n_q_points);

    /**
     * Initialize the internal data held in the mapping objects.
     */
    kx_mapping_data.shape_values.resize(kx_mapping_n_shape_functions *
                                        n_q_points);
    kx_mapping_data.shape_derivatives.resize(kx_mapping_n_shape_functions *
                                             n_q_points);
    ky_mapping_data.shape_values.resize(ky_mapping_n_shape_functions *
                                        n_q_points);
    ky_mapping_data.shape_derivatives.resize(ky_mapping_n_shape_functions *
                                             n_q_points);

    /**
     * Quadrature points in the Sauter's parametric space.
     */
    std::vector<Point<dim * 2>> quad_points =
      quad_rule_for_common_edge.get_points();

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
            // Transform the quadrature point in the parametric space to
            // the unit cells of \f$K_x\f$ and \f$K_y\f$.
            sauter_common_edge_parametric_coords_to_unit_cells(
              quad_points[q],
              k,
              kx_unit_quad_points[q],
              ky_unit_quad_points[q]);

            /**
             * Iterate over each finite element shape function in the
             * lexicographic order on the unit cell of \f$K_x\f$ and evaluate it
             * at the current quadrature point @p kx_unit_quad_points[q].
             */
            for (unsigned int s = 0; s < kx_dofs_per_cell; s++)
              {
                kx_shape_value_table_for_common_edge(TableIndices<3>(s, k, q)) =
                  kx_fe.shape_value(kx_poly_space_inverse_numbering[s],
                                    kx_unit_quad_points[q]);
              }

            /**
             * Iterate over each finite element shape function in the
             * lexicographic order on the unit cell of \f$K_y\f$ and evaluate it
             * at the current quadrature point @p ky_unit_quad_points[q].
             */
            for (unsigned int s = 0; s < ky_dofs_per_cell; s++)
              {
                ky_shape_value_table_for_common_edge(TableIndices<3>(s, k, q)) =
                  ky_fe.shape_value(ky_poly_space_inverse_numbering[s],
                                    ky_unit_quad_points[q]);
              }

            // Calculate the Jacobian matrix evaluated at
            // <code>kx_quad_point</code> in  \f$K_x\f$. Matrix rows correspond
            // to shape functions which are in the lexicographic order.
            kx_shape_grad_matrix_table_for_common_edge(TableIndices<2>(k, q)) =
              BEMTools::shape_grad_matrix_in_lexicographic_order(
                kx_fe, kx_unit_quad_points[q]);
            // Calculate the Jacobian matrix evaluated at
            // <code>ky_quad_point</code> in  \f$K_y\f$. Matrix rows correspond
            // to shape functions which are in the lexicographic order.
            ky_shape_grad_matrix_table_for_common_edge(TableIndices<2>(k, q)) =
              BEMTools::shape_grad_matrix_in_lexicographic_order(
                ky_fe, ky_unit_quad_points[q]);
          }

        /**
         * Compute mapping shape function values and their derivatives in batch.
         *
         * \alert{Even though the internally generated polynomials in the
         * mapping object are in the tensor product order, the shape function
         * values and derivatives are still in the hierarchic order. This can be
         * verified by checking the source code of
         * @p MappingQ<dim, spacedim>::InternalData::compute_shape_function_values.
         * (see
         * http://localhost/dealii-9.4.1-doc/mapping__q_8cc_source.html#l00271)
         *
         * However, this behavior is different from the documentation for the
         * function @p MappingQ< dim, spacedim >::InternalData::shape().}
         */
        // Get the numbering for accessing the support points in the
        // lexicographic ordering which are stored in the hierarchic ordering.
        const std::vector<unsigned int> kx_mapping_poly_space_inverse_numbering(
          FETools::lexicographic_to_hierarchic_numbering<dim>(
            kx_mapping_data.polynomial_degree));
        const std::vector<unsigned int> ky_mapping_poly_space_inverse_numbering(
          FETools::lexicographic_to_hierarchic_numbering<dim>(
            ky_mapping_data.polynomial_degree));

        kx_mapping_data.compute_shape_function_values(kx_unit_quad_points);
        ky_mapping_data.compute_shape_function_values(ky_unit_quad_points);

        for (unsigned int q = 0; q < n_q_points; q++)
          {
            for (unsigned int s = 0; s < kx_mapping_n_shape_functions; s++)
              {
                kx_mapping_shape_value_table_for_common_edge(
                  TableIndices<3>(s, k, q)) =
                  kx_mapping_data.shape(
                    q, kx_mapping_poly_space_inverse_numbering[s]);

                for (unsigned int d = 0; d < dim; d++)
                  {
                    kx_mapping_shape_grad_matrix_table_for_common_edge(
                      TableIndices<2>(k, q))(s, d) =
                      kx_mapping_data.derivative(
                        q, kx_mapping_poly_space_inverse_numbering[s])[d];
                  }
              }

            for (unsigned int s = 0; s < ky_mapping_n_shape_functions; s++)
              {
                ky_mapping_shape_value_table_for_common_edge(
                  TableIndices<3>(s, k, q)) =
                  ky_mapping_data.shape(
                    q, ky_mapping_poly_space_inverse_numbering[s]);

                for (unsigned int d = 0; d < dim; d++)
                  {
                    ky_mapping_shape_grad_matrix_table_for_common_edge(
                      TableIndices<2>(k, q))(s, d) =
                      ky_mapping_data.derivative(
                        q, ky_mapping_poly_space_inverse_numbering[s])[d];
                  }
              }
          }
      }
  }


  template <int dim, int spacedim, typename RangeNumberType>
  void
  BEMValues<dim, spacedim, RangeNumberType>::
    shape_function_values_common_vertex()
  {
    const unsigned int kx_dofs_per_cell = kx_fe.dofs_per_cell;
    const unsigned int ky_dofs_per_cell = ky_fe.dofs_per_cell;
    const unsigned int kx_mapping_n_shape_functions =
      kx_mapping_data.n_shape_functions;
    const unsigned int ky_mapping_n_shape_functions =
      ky_mapping_data.n_shape_functions;
    const unsigned int n_q_points = quad_rule_for_common_vertex.size();

    // Make assertion about the length for each dimension of the data table.
    AssertDimension(kx_shape_value_table_for_common_vertex.size(0),
                    kx_dofs_per_cell);
    AssertDimension(kx_shape_value_table_for_common_vertex.size(1), 4);
    AssertDimension(kx_shape_value_table_for_common_vertex.size(2), n_q_points);

    AssertDimension(ky_shape_value_table_for_common_vertex.size(0),
                    ky_dofs_per_cell);
    AssertDimension(ky_shape_value_table_for_common_vertex.size(1), 4);
    AssertDimension(ky_shape_value_table_for_common_vertex.size(2), n_q_points);

    AssertDimension(kx_mapping_shape_value_table_for_common_vertex.size(0),
                    kx_mapping_n_shape_functions);
    AssertDimension(kx_mapping_shape_value_table_for_common_vertex.size(1), 4);
    AssertDimension(kx_mapping_shape_value_table_for_common_vertex.size(2),
                    n_q_points);

    AssertDimension(ky_mapping_shape_value_table_for_common_vertex.size(0),
                    ky_mapping_n_shape_functions);
    AssertDimension(ky_mapping_shape_value_table_for_common_vertex.size(1), 4);
    AssertDimension(ky_mapping_shape_value_table_for_common_vertex.size(2),
                    n_q_points);

    AssertDimension(kx_shape_grad_matrix_table_for_common_vertex.size(0), 4);
    AssertDimension(kx_shape_grad_matrix_table_for_common_vertex.size(1),
                    n_q_points);

    AssertDimension(ky_shape_grad_matrix_table_for_common_vertex.size(0), 4);
    AssertDimension(ky_shape_grad_matrix_table_for_common_vertex.size(1),
                    n_q_points);

    AssertDimension(
      kx_mapping_shape_grad_matrix_table_for_common_vertex.size(0), 4);
    AssertDimension(
      kx_mapping_shape_grad_matrix_table_for_common_vertex.size(1), n_q_points);

    AssertDimension(
      ky_mapping_shape_grad_matrix_table_for_common_vertex.size(0), 4);
    AssertDimension(
      ky_mapping_shape_grad_matrix_table_for_common_vertex.size(1), n_q_points);

    /**
     * Initialize the internal data held in the mapping objects.
     */
    kx_mapping_data.shape_values.resize(kx_mapping_n_shape_functions *
                                        n_q_points);
    kx_mapping_data.shape_derivatives.resize(kx_mapping_n_shape_functions *
                                             n_q_points);
    ky_mapping_data.shape_values.resize(ky_mapping_n_shape_functions *
                                        n_q_points);
    ky_mapping_data.shape_derivatives.resize(ky_mapping_n_shape_functions *
                                             n_q_points);

    /**
     * Quadrature points in the Sauter's parametric space.
     */
    std::vector<Point<dim * 2>> quad_points =
      quad_rule_for_common_vertex.get_points();

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

    /**
     * Quadrature points in the unit cells of \f$K_x\f$ and \f$K_y\f$
     * respectively.
     */
    std::vector<Point<dim>> kx_unit_quad_points(n_q_points);
    std::vector<Point<dim>> ky_unit_quad_points(n_q_points);

    // Iterate over each $k_3$ part.
    for (unsigned k = 0; k < 4; k++)
      {
        // Iterate over each quadrature point.
        for (unsigned int q = 0; q < n_q_points; q++)
          {
            // Transform the quadrature point in the parametric space to
            // the unit cells of \f$K_x\f$ and \f$K_y\f$.
            sauter_common_vertex_parametric_coords_to_unit_cells(
              quad_points[q],
              k,
              kx_unit_quad_points[q],
              ky_unit_quad_points[q]);

            /**
             * Iterate over each finite element shape function in the
             * lexicographic order on the unit cell of \f$K_x\f$ and evaluate it
             * at the current quadrature point @p kx_unit_quad_points[q].
             */
            for (unsigned int s = 0; s < kx_dofs_per_cell; s++)
              {
                kx_shape_value_table_for_common_vertex(
                  TableIndices<3>(s, k, q)) =
                  kx_fe.shape_value(kx_poly_space_inverse_numbering[s],
                                    kx_unit_quad_points[q]);
              }

            /**
             * Iterate over each finite element shape function in the
             * lexicographic order on the unit cell of \f$K_y\f$ and evaluate it
             * at the current quadrature point @p ky_unit_quad_points[q].
             */
            for (unsigned int s = 0; s < ky_dofs_per_cell; s++)
              {
                ky_shape_value_table_for_common_vertex(
                  TableIndices<3>(s, k, q)) =
                  ky_fe.shape_value(ky_poly_space_inverse_numbering[s],
                                    ky_unit_quad_points[q]);
              }

            // Calculate the Jacobian matrix evaluated at
            // <code>kx_quad_point</code> in  \f$K_x\f$. Matrix rows correspond
            // to shape functions which are in the lexicographic order.
            kx_shape_grad_matrix_table_for_common_vertex(
              TableIndices<2>(k, q)) =
              BEMTools::shape_grad_matrix_in_lexicographic_order(
                kx_fe, kx_unit_quad_points[q]);
            // Calculate the Jacobian matrix evaluated at
            // <code>ky_quad_point</code> in  \f$K_y\f$. Matrix rows correspond
            // to shape functions which are in the lexicographic order.
            ky_shape_grad_matrix_table_for_common_vertex(
              TableIndices<2>(k, q)) =
              BEMTools::shape_grad_matrix_in_lexicographic_order(
                ky_fe, ky_unit_quad_points[q]);
          }

        /**
         * Compute mapping shape function values and their derivatives in batch.
         *
         * \alert{Even though the internally generated polynomials in the
         * mapping object are in the tensor product order, the shape function
         * values and derivatives are still in the hierarchic order. This can be
         * verified by checking the source code of
         * @p MappingQ<dim, spacedim>::InternalData::compute_shape_function_values.
         * (see
         * http://localhost/dealii-9.4.1-doc/mapping__q_8cc_source.html#l00271)
         *
         * However, this behavior is different from the documentation for the
         * function @p MappingQ< dim, spacedim >::InternalData::shape().}
         */
        // Get the numbering for accessing the support points in the
        // lexicographic ordering which are stored in the hierarchic ordering.
        const std::vector<unsigned int> kx_mapping_poly_space_inverse_numbering(
          FETools::lexicographic_to_hierarchic_numbering<dim>(
            kx_mapping_data.polynomial_degree));
        const std::vector<unsigned int> ky_mapping_poly_space_inverse_numbering(
          FETools::lexicographic_to_hierarchic_numbering<dim>(
            ky_mapping_data.polynomial_degree));

        kx_mapping_data.compute_shape_function_values(kx_unit_quad_points);
        ky_mapping_data.compute_shape_function_values(ky_unit_quad_points);

        for (unsigned int q = 0; q < n_q_points; q++)
          {
            for (unsigned int s = 0; s < kx_mapping_n_shape_functions; s++)
              {
                kx_mapping_shape_value_table_for_common_vertex(
                  TableIndices<3>(s, k, q)) =
                  kx_mapping_data.shape(
                    q, kx_mapping_poly_space_inverse_numbering[s]);

                for (unsigned int d = 0; d < dim; d++)
                  {
                    kx_mapping_shape_grad_matrix_table_for_common_vertex(
                      TableIndices<2>(k, q))(s, d) =
                      kx_mapping_data.derivative(
                        q, kx_mapping_poly_space_inverse_numbering[s])[d];
                  }
              }

            for (unsigned int s = 0; s < ky_mapping_n_shape_functions; s++)
              {
                ky_mapping_shape_value_table_for_common_vertex(
                  TableIndices<3>(s, k, q)) =
                  ky_mapping_data.shape(
                    q, ky_mapping_poly_space_inverse_numbering[s]);

                for (unsigned int d = 0; d < dim; d++)
                  {
                    ky_mapping_shape_grad_matrix_table_for_common_vertex(
                      TableIndices<2>(k, q))(s, d) =
                      ky_mapping_data.derivative(
                        q, ky_mapping_poly_space_inverse_numbering[s])[d];
                  }
              }
          }
      }
  }


  template <int dim, int spacedim, typename RangeNumberType>
  void
  BEMValues<dim, spacedim, RangeNumberType>::shape_function_values_regular()
  {
    const unsigned int kx_dofs_per_cell = kx_fe.dofs_per_cell;
    const unsigned int ky_dofs_per_cell = ky_fe.dofs_per_cell;
    const unsigned int kx_mapping_n_shape_functions =
      kx_mapping_data.n_shape_functions;
    const unsigned int ky_mapping_n_shape_functions =
      ky_mapping_data.n_shape_functions;
    const unsigned int n_q_points = quad_rule_for_regular.size();

    // Make assertion about the length for each dimension of the data table.
    AssertDimension(kx_shape_value_table_for_regular.size(0), kx_dofs_per_cell);
    AssertDimension(kx_shape_value_table_for_regular.size(1), 1);
    AssertDimension(kx_shape_value_table_for_regular.size(2), n_q_points);

    AssertDimension(ky_shape_value_table_for_regular.size(0), ky_dofs_per_cell);
    AssertDimension(ky_shape_value_table_for_regular.size(1), 1);
    AssertDimension(ky_shape_value_table_for_regular.size(2), n_q_points);

    AssertDimension(kx_mapping_shape_value_table_for_regular.size(0),
                    kx_mapping_n_shape_functions);
    AssertDimension(kx_mapping_shape_value_table_for_regular.size(1), 1);
    AssertDimension(kx_mapping_shape_value_table_for_regular.size(2),
                    n_q_points);

    AssertDimension(ky_mapping_shape_value_table_for_regular.size(0),
                    ky_mapping_n_shape_functions);
    AssertDimension(ky_mapping_shape_value_table_for_regular.size(1), 1);
    AssertDimension(ky_mapping_shape_value_table_for_regular.size(2),
                    n_q_points);

    AssertDimension(kx_shape_grad_matrix_table_for_regular.size(0), 1);
    AssertDimension(kx_shape_grad_matrix_table_for_regular.size(1), n_q_points);

    AssertDimension(ky_shape_grad_matrix_table_for_regular.size(0), 1);
    AssertDimension(ky_shape_grad_matrix_table_for_regular.size(1), n_q_points);

    AssertDimension(kx_mapping_shape_grad_matrix_table_for_regular.size(0), 1);
    AssertDimension(kx_mapping_shape_grad_matrix_table_for_regular.size(1),
                    n_q_points);

    AssertDimension(ky_mapping_shape_grad_matrix_table_for_regular.size(0), 1);
    AssertDimension(ky_mapping_shape_grad_matrix_table_for_regular.size(1),
                    n_q_points);

    /**
     * Initialize the internal data held in the mapping objects.
     */
    kx_mapping_data.shape_values.resize(kx_mapping_n_shape_functions *
                                        n_q_points);
    kx_mapping_data.shape_derivatives.resize(kx_mapping_n_shape_functions *
                                             n_q_points);
    ky_mapping_data.shape_values.resize(ky_mapping_n_shape_functions *
                                        n_q_points);
    ky_mapping_data.shape_derivatives.resize(ky_mapping_n_shape_functions *
                                             n_q_points);

    /**
     * Quadrature points in the Sauter's parametric space.
     */
    std::vector<Point<dim * 2>> quad_points =
      quad_rule_for_regular.get_points();

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

    /**
     * Quadrature points in the unit cells of \f$K_x\f$ and \f$K_y\f$
     * respectively.
     */
    std::vector<Point<dim>> kx_unit_quad_points(n_q_points);
    std::vector<Point<dim>> ky_unit_quad_points(n_q_points);

    // Iterate over each quadrature point.
    for (unsigned int q = 0; q < n_q_points; q++)
      {
        // Transform the quadrature point in the parametric space to the
        // unit cells of \f$K_x\f$ and \f$K_y\f$.
        sauter_regular_parametric_coords_to_unit_cells(quad_points[q],
                                                       kx_unit_quad_points[q],
                                                       ky_unit_quad_points[q]);

        /**
         * Iterate over each finite element shape function in the
         * lexicographic order on the unit cell of \f$K_x\f$ and evaluate it
         * at the current quadrature point @p kx_unit_quad_points[q].
         */
        for (unsigned int s = 0; s < kx_dofs_per_cell; s++)
          {
            kx_shape_value_table_for_regular(TableIndices<3>(s, 0, q)) =
              kx_fe.shape_value(kx_poly_space_inverse_numbering[s],
                                kx_unit_quad_points[q]);
          }

        /**
         * Iterate over each finite element shape function in the
         * lexicographic order on the unit cell of \f$K_y\f$ and evaluate it
         * at the current quadrature point @p ky_unit_quad_points[q].
         */
        for (unsigned int s = 0; s < ky_dofs_per_cell; s++)
          {
            ky_shape_value_table_for_regular(TableIndices<3>(s, 0, q)) =
              ky_fe.shape_value(ky_poly_space_inverse_numbering[s],
                                ky_unit_quad_points[q]);
          }

        // Calculate the Jacobian matrix evaluated at
        // <code>kx_quad_point</code> in  \f$K_x\f$. Matrix rows correspond
        // to shape functions which are in the lexicographic order.
        kx_shape_grad_matrix_table_for_regular(TableIndices<2>(0, q)) =
          BEMTools::shape_grad_matrix_in_lexicographic_order(
            kx_fe, kx_unit_quad_points[q]);
        // Calculate the Jacobian matrix evaluated at
        // <code>ky_quad_point</code> in  \f$K_y\f$. Matrix rows correspond
        // to shape functions which are in the lexicographic order.
        ky_shape_grad_matrix_table_for_regular(TableIndices<2>(0, q)) =
          BEMTools::shape_grad_matrix_in_lexicographic_order(
            ky_fe, ky_unit_quad_points[q]);
      }

    /**
     * Compute mapping shape function values and their derivatives in batch.
     *
     * \alert{Even though the internally generated polynomials in the
     * mapping object are in the tensor product order, the shape function
     * values and derivatives are still in the hierarchic order. This can be
     * verified by checking the source code of
     * @p MappingQ<dim, spacedim>::InternalData::compute_shape_function_values.
     * (see
     * http://localhost/dealii-9.4.1-doc/mapping__q_8cc_source.html#l00271)
     *
     * However, this behavior is different from the documentation for the
     * function @p MappingQ< dim, spacedim >::InternalData::shape().}
     */
    // Get the numbering for accessing the support points in the
    // lexicographic ordering which are stored in the hierarchic ordering.
    const std::vector<unsigned int> kx_mapping_poly_space_inverse_numbering(
      FETools::lexicographic_to_hierarchic_numbering<dim>(
        kx_mapping_data.polynomial_degree));
    const std::vector<unsigned int> ky_mapping_poly_space_inverse_numbering(
      FETools::lexicographic_to_hierarchic_numbering<dim>(
        ky_mapping_data.polynomial_degree));

    kx_mapping_data.compute_shape_function_values(kx_unit_quad_points);
    ky_mapping_data.compute_shape_function_values(ky_unit_quad_points);

    for (unsigned int q = 0; q < n_q_points; q++)
      {
        for (unsigned int s = 0; s < kx_mapping_n_shape_functions; s++)
          {
            kx_mapping_shape_value_table_for_regular(TableIndices<3>(s, 0, q)) =
              kx_mapping_data.shape(q,
                                    kx_mapping_poly_space_inverse_numbering[s]);

            for (unsigned int d = 0; d < dim; d++)
              {
                kx_mapping_shape_grad_matrix_table_for_regular(
                  TableIndices<2>(0, q))(s, d) =
                  kx_mapping_data.derivative(
                    q, kx_mapping_poly_space_inverse_numbering[s])[d];
              }
          }

        for (unsigned int s = 0; s < ky_mapping_n_shape_functions; s++)
          {
            ky_mapping_shape_value_table_for_regular(TableIndices<3>(s, 0, q)) =
              ky_mapping_data.shape(q,
                                    ky_mapping_poly_space_inverse_numbering[s]);

            for (unsigned int d = 0; d < dim; d++)
              {
                ky_mapping_shape_grad_matrix_table_for_regular(
                  TableIndices<2>(0, q))(s, d) =
                  ky_mapping_data.derivative(
                    q, ky_mapping_poly_space_inverse_numbering[s])[d];
              }
          }
      }
  }


  /**
   * Structure holding cell-wise local matrix data and DoF indices, which is
   * used for SMP parallel computation of the scaled FEM mass matrix.
   */
  template <int dim, int spacedim = dim, typename RangeNumberType = double>
  struct CellWiseCopyDataForMassMatrix
  {
    LAPACKFullMatrixExt<RangeNumberType> local_matrix;
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
    CellWiseCopyDataForMassMatrix(
      const FiniteElement<dim, spacedim> &fe_for_test_space,
      const FiniteElement<dim, spacedim> &fe_for_trial_space)
      : local_matrix(fe_for_test_space.dofs_per_cell,
                     fe_for_trial_space.dofs_per_cell)
      , local_dof_indices_for_test_space(fe_for_test_space.dofs_per_cell)
      , local_dof_indices_for_trial_space(fe_for_trial_space.dofs_per_cell)
    {}

    /**
     * Copy constructor
     *
     * @param copy_data
     */
    CellWiseCopyDataForMassMatrix(
      const CellWiseCopyDataForMassMatrix<dim, spacedim, RangeNumberType>
        &copy_data)
      : local_matrix(copy_data.local_matrix)
      , local_dof_indices_for_test_space(
          copy_data.local_dof_indices_for_test_space)
      , local_dof_indices_for_trial_space(
          copy_data.local_dof_indices_for_trial_space)
    {}
  };


  template <int dim, int spacedim = dim, typename RangeNumberType = double>
  struct CellWiseCopyDataForMassMatrixVmult
  {
    LAPACKFullMatrixExt<RangeNumberType> local_matrix;
    Vector<RangeNumberType>              local_u, local_v;

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
    CellWiseCopyDataForMassMatrixVmult(
      const FiniteElement<dim, spacedim> &fe_for_test_space,
      const FiniteElement<dim, spacedim> &fe_for_trial_space)
      : local_matrix(fe_for_test_space.dofs_per_cell,
                     fe_for_trial_space.dofs_per_cell)
      , local_u(fe_for_test_space.dofs_per_cell)
      , local_v(fe_for_trial_space.dofs_per_cell)
      , local_dof_indices_for_test_space(fe_for_test_space.dofs_per_cell)
      , local_dof_indices_for_trial_space(fe_for_trial_space.dofs_per_cell)
    {}

    /**
     * Copy constructor
     *
     * @param copy_data
     */
    CellWiseCopyDataForMassMatrixVmult(
      const CellWiseCopyDataForMassMatrix<dim, spacedim, RangeNumberType>
        &copy_data)
      : local_matrix(copy_data.local_matrix)
      , local_u(copy_data.local_u)
      , local_v(copy_data.local_v)
      , local_dof_indices_for_test_space(
          copy_data.local_dof_indices_for_test_space)
      , local_dof_indices_for_trial_space(
          copy_data.local_dof_indices_for_trial_space)
    {}
  };


  /**
   * Structure holding temporary data which are needed for cell-wise
   * integration, such as for the scaled mass matrix term \f$(v, \alpha \cdot
   * u)\f$.
   */
  template <int dim, int spacedim = dim>
  struct CellWiseScratchDataForMassMatrix
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
    CellWiseScratchDataForMassMatrix(
      const FiniteElement<dim, spacedim> &fe_for_test_space,
      const FiniteElement<dim, spacedim> &fe_for_trial_space,
      const Quadrature<dim>              &quadrature,
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
    CellWiseScratchDataForMassMatrix(
      const CellWiseScratchDataForMassMatrix<dim, spacedim> &scratch_data)
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
      const Quadrature<dim>              &quadrature,
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
  class PairCellWiseScratchData
  {
  public:
    using FE_Poly_short = FE_Poly<dim, spacedim>;

    /**
     * ID for the working thread associated with this scratch data.
     */
    std::thread::id thread_id;

#if ENABLE_DEBUG == 1 && ENABLE_TIMER == 1
    /**
     * Timer object associated with the scratch data, which is bound to a
     * working thread.
     */
    Timer timer;

    /**
     * Output stream for the working thread associated with this scratch data,
     * which will record log messages, such as timing.
     */
    std::ofstream log_stream;
#endif

    /**
     * CUDA stream associated with this CPU work stream.
     */
    cudaStream_t cuda_stream_handle;

    /**
     * The intersection set of the vertex local indices for the two cells
     * \f$K_x\f$ and \f$K_y\f$.
     */
    std::vector<std::pair<unsigned int, unsigned int>>
      common_vertex_pair_local_indices;

    /**
     * List of mapping support points in the real cell \f$K_x\f$ in the
     * tensor product order.
     */
    std::vector<Point<spacedim, RangeNumberType>>
      kx_mapping_support_points_in_default_order;
    /**
     * List of mapping support points in the real cell \f$K_y\f$ in the
     * tensor product order.
     */
    std::vector<Point<spacedim, RangeNumberType>>
      ky_mapping_support_points_in_default_order;

    /**
     * Permuted list of mapping support points in the real cell \f$K_x\f$.
     */
    std::vector<Point<spacedim, RangeNumberType>>
      kx_mapping_support_points_permuted;
    /**
     * Permuted list of mapping support points in the real cell \f$K_y\f$.
     */
    std::vector<Point<spacedim, RangeNumberType>>
      ky_mapping_support_points_permuted;

    std::vector<Point<2, RangeNumberType>>
      kx_mapping_support_points_permuted_xy_components;
    std::vector<Point<2, RangeNumberType>>
      kx_mapping_support_points_permuted_yz_components;
    std::vector<Point<2, RangeNumberType>>
      kx_mapping_support_points_permuted_zx_components;

    std::vector<Point<2, RangeNumberType>>
      ky_mapping_support_points_permuted_xy_components;
    std::vector<Point<2, RangeNumberType>>
      ky_mapping_support_points_permuted_yz_components;
    std::vector<Point<2, RangeNumberType>>
      ky_mapping_support_points_permuted_zx_components;

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
     * The numbering used for accessing the list of support points in the
     * mapping object for \f$K_x\f$ in the lexicographic order, where the list
     * of support points are stored in the hierarchic order.
     */
    std::vector<unsigned int> kx_mapping_poly_space_numbering_inverse;
    /**
     * The numbering used for accessing the list of support points in the
     * mapping object for \f$K_y\f$ in the lexicographic order, where the list
     * of support points are stored in the hierarchic order.
     */
    std::vector<unsigned int> ky_mapping_poly_space_numbering_inverse;
    /**
     * The numbering used for accessing the list of support points in the
     * mapping object for \f$K_y\f$ in the reversed lexicographic order, where
     * the list of support points are stored in the hierarchical order.
     *
     * \mynote{This numbering occurs only when \f$K_x\f$ and \f$K_y\f$ share a
     * common edge.}
     */
    std::vector<unsigned int> ky_mapping_reversed_poly_space_numbering_inverse;

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

    std::vector<unsigned int> kx_mapping_support_point_permutation;
    std::vector<unsigned int> ky_mapping_support_point_permutation;

    // The first dimension of the following data tables is the \f$k_3\f$ index.
    /**
     * Jacobian from the unit cell to the real cell \f$K_x\f$ for each
     * \f$k_3\f$ term and at each quadrature point for the same panel case.
     */
    Table<2, RangeNumberType> kx_jacobians_same_panel;
    /**
     * Jacobian from the unit cell to the real cell \f$K_x\f$ for each
     * \f$k_3\f$ term and at each quadrature point for the common edge case.
     */
    Table<2, RangeNumberType> kx_jacobians_common_edge;
    /**
     * Jacobian from the unit cell to the real cell \f$K_x\f$ for each
     * \f$k_3\f$ term and at each quadrature point for the common vertex case.
     */
    Table<2, RangeNumberType> kx_jacobians_common_vertex;
    /**
     * Jacobian from the unit cell to the real cell \f$K_x\f$ for each
     * \f$k_3\f$ term and at each quadrature point for the regular case.
     */
    Table<2, RangeNumberType> kx_jacobians_regular;

    /**
     * Normal vector at each quadrature point in the real cell \f$K_x\f$ for
     * the same panel case.
     */
    Table<2, Tensor<1, spacedim, RangeNumberType>> kx_normals_same_panel;
    /**
     * Normal vector at each quadrature point in the real cell \f$K_x\f$ for
     * the common edge case.
     */
    Table<2, Tensor<1, spacedim, RangeNumberType>> kx_normals_common_edge;
    /**
     * Normal vector at each quadrature point in the real cell \f$K_x\f$ for
     * the common vertex case.
     */
    Table<2, Tensor<1, spacedim, RangeNumberType>> kx_normals_common_vertex;
    /**
     * Normal vector at each quadrature point in the real cell \f$K_x\f$ for
     * the regular case.
     */
    Table<2, Tensor<1, spacedim, RangeNumberType>> kx_normals_regular;

    /**
     * Covariant transformation matrix for each \f$k_3\f$ term and each
     * quadrature point in the real cell \f$K_x\f$ for the same panel case.
     */
    Table<2, LAPACKFullMatrixExt<RangeNumberType>> kx_covariants_same_panel;
    /**
     * Covariant transformation matrix for each \f$k_3\f$ term and each
     * quadrature point in the real cell \f$K_x\f$ for the common edge case.
     */
    Table<2, LAPACKFullMatrixExt<RangeNumberType>> kx_covariants_common_edge;
    /**
     * Covariant transformation matrix for each \f$k_3\f$ term and each
     * quadrature point in the real cell \f$K_x\f$ for the common vertex case.
     */
    Table<2, LAPACKFullMatrixExt<RangeNumberType>> kx_covariants_common_vertex;
    /**
     * Covariant transformation matrix for each \f$k_3\f$ term and each
     * quadrature point in the real cell \f$K_x\f$ for the regular case.
     */
    Table<2, LAPACKFullMatrixExt<RangeNumberType>> kx_covariants_regular;

    /**
     * Coordinates in the real cell \f$K_x\f$ for each \f$k_3\f$ term and each
     * quadrature point for the same panel case.
     */
    Table<2, Point<spacedim, RangeNumberType>> kx_quad_points_same_panel;
    /**
     * Coordinates in the real cell \f$K_x\f$ for each \f$k_3\f$ term and each
     * quadrature point for the common edge case.
     */
    Table<2, Point<spacedim, RangeNumberType>> kx_quad_points_common_edge;
    /**
     * Coordinates in the real cell \f$K_x\f$ for each \f$k_3\f$ term and each
     * quadrature point for the common vertex case.
     */
    Table<2, Point<spacedim, RangeNumberType>> kx_quad_points_common_vertex;
    /**
     * Coordinates in the real cell \f$K_x\f$ for each \f$k_3\f$ term and each
     * quadrature point for the regular case.
     */
    Table<2, Point<spacedim, RangeNumberType>> kx_quad_points_regular;


    /**
     * Jacobian from the unit cell to the real cell \f$K_y\f$ for each
     * \f$k_3\f$ term and at each quadrature point for the same panel case.
     */
    Table<2, RangeNumberType> ky_jacobians_same_panel;
    /**
     * Jacobian from the unit cell to the real cell \f$K_y\f$ for each
     * \f$k_3\f$ term and at each quadrature point for the common edge case.
     */
    Table<2, RangeNumberType> ky_jacobians_common_edge;
    /**
     * Jacobian from the unit cell to the real cell \f$K_y\f$ for each
     * \f$k_3\f$ term and at each quadrature point for the common vertex case.
     */
    Table<2, RangeNumberType> ky_jacobians_common_vertex;
    /**
     * Jacobian from the unit cell to the real cell \f$K_y\f$ for each
     * \f$k_3\f$ term and at each quadrature point for the regular case.
     */
    Table<2, RangeNumberType> ky_jacobians_regular;

    /**
     * Normal vector at each quadrature point in the real cell \f$K_y\f$ for
     * the same panel case.
     */
    Table<2, Tensor<1, spacedim, RangeNumberType>> ky_normals_same_panel;
    /**
     * Normal vector at each quadrature point in the real cell \f$K_y\f$ for
     * the common edge case.
     */
    Table<2, Tensor<1, spacedim, RangeNumberType>> ky_normals_common_edge;
    /**
     * Normal vector at each quadrature point in the real cell \f$K_y\f$ for
     * the common vertex case.
     */
    Table<2, Tensor<1, spacedim, RangeNumberType>> ky_normals_common_vertex;
    /**
     * Normal vector at each quadrature point in the real cell \f$K_y\f$ for
     * the regular case.
     */
    Table<2, Tensor<1, spacedim, RangeNumberType>> ky_normals_regular;

    /**
     * Covariant transformation matrix for each \f$k_3\f$ term and each
     * quadrature point in the real cell \f$K_x\f$ for the same panel case.
     */
    Table<2, LAPACKFullMatrixExt<RangeNumberType>> ky_covariants_same_panel;
    /**
     * Covariant transformation matrix for each \f$k_3\f$ term and each
     * quadrature point in the real cell \f$K_x\f$ for the common edge case.
     */
    Table<2, LAPACKFullMatrixExt<RangeNumberType>> ky_covariants_common_edge;
    /**
     * Covariant transformation matrix for each \f$k_3\f$ term and each
     * quadrature point in the real cell \f$K_x\f$ for the common vertex case.
     */
    Table<2, LAPACKFullMatrixExt<RangeNumberType>> ky_covariants_common_vertex;
    /**
     * Covariant transformation matrix for each \f$k_3\f$ term and each
     * quadrature point in the real cell \f$K_x\f$ for the regular case.
     */
    Table<2, LAPACKFullMatrixExt<RangeNumberType>> ky_covariants_regular;

    /**
     * Coordinates in the real cell \f$K_y\f$ for each \f$k_3\f$ term and each
     * quadrature point for the same panel case.
     */
    Table<2, Point<spacedim, RangeNumberType>> ky_quad_points_same_panel;
    /**
     * Coordinates in the real cell \f$K_y\f$ for each \f$k_3\f$ term and each
     * quadrature point for the common edge case.
     */
    Table<2, Point<spacedim, RangeNumberType>> ky_quad_points_common_edge;
    /**
     * Coordinates in the real cell \f$K_y\f$ for each \f$k_3\f$ term and each
     * quadrature point for the common vertex case.
     */
    Table<2, Point<spacedim, RangeNumberType>> ky_quad_points_common_vertex;
    /**
     * Coordinates in the real cell \f$K_y\f$ for each \f$k_3\f$ term and each
     * quadrature point for the regular case.
     */
    Table<2, Point<spacedim, RangeNumberType>> ky_quad_points_regular;
    /**
     * Buffer holding quadrature results accumulated from all thread blocks.
     */
    RangeNumberType *quad_values_in_thread_blocks;

    /**
     * Constructor to be used for assembling full BEM matrix, which does not
     * involve @p thread_id.
     *
     * @param kx_fe
     * @param ky_fe
     * @param kx_mapping
     * @param ky_mapping
     * @param bem_values
     */
    PairCellWiseScratchData(const FiniteElement<dim, spacedim>      &kx_fe,
                            const FiniteElement<dim, spacedim>      &ky_fe,
                            const MappingQGenericExt<dim, spacedim> &kx_mapping,
                            const MappingQGenericExt<dim, spacedim> &ky_mapping,
                            const BEMValues<dim, spacedim>          &bem_values)
      : cuda_stream_handle(0)
      , common_vertex_pair_local_indices(0)
      , kx_mapping_support_points_in_default_order(
          bem_values.kx_mapping_data.n_shape_functions)
      , ky_mapping_support_points_in_default_order(
          bem_values.ky_mapping_data.n_shape_functions)
      , kx_mapping_support_points_permuted(
          bem_values.kx_mapping_data.n_shape_functions)
      , ky_mapping_support_points_permuted(
          bem_values.ky_mapping_data.n_shape_functions)
      , kx_mapping_support_points_permuted_xy_components(
          bem_values.kx_mapping_data.n_shape_functions)
      , kx_mapping_support_points_permuted_yz_components(
          bem_values.kx_mapping_data.n_shape_functions)
      , kx_mapping_support_points_permuted_zx_components(
          bem_values.kx_mapping_data.n_shape_functions)
      , ky_mapping_support_points_permuted_xy_components(
          bem_values.ky_mapping_data.n_shape_functions)
      , ky_mapping_support_points_permuted_yz_components(
          bem_values.ky_mapping_data.n_shape_functions)
      , ky_mapping_support_points_permuted_zx_components(
          bem_values.ky_mapping_data.n_shape_functions)
      , kx_local_dof_indices_in_default_dof_order(kx_fe.dofs_per_cell)
      , ky_local_dof_indices_in_default_dof_order(ky_fe.dofs_per_cell)
      , kx_fe_poly_space_numbering_inverse(kx_fe.dofs_per_cell)
      , ky_fe_poly_space_numbering_inverse(ky_fe.dofs_per_cell)
      , kx_mapping_poly_space_numbering_inverse(
          bem_values.kx_mapping_data.n_shape_functions)
      , ky_mapping_poly_space_numbering_inverse(
          bem_values.ky_mapping_data.n_shape_functions)
      , ky_mapping_reversed_poly_space_numbering_inverse(
          bem_values.ky_mapping_data.n_shape_functions)
      , kx_local_dof_permutation(kx_fe.dofs_per_cell)
      , ky_local_dof_permutation(ky_fe.dofs_per_cell)
      , kx_mapping_support_point_permutation(
          bem_values.kx_mapping_data.n_shape_functions)
      , ky_mapping_support_point_permutation(
          bem_values.ky_mapping_data.n_shape_functions)
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
      , kx_covariants_same_panel(8, bem_values.quad_rule_for_same_panel.size())
      , kx_covariants_common_edge(6,
                                  bem_values.quad_rule_for_common_edge.size())
      , kx_covariants_common_vertex(
          4,
          bem_values.quad_rule_for_common_vertex.size())
      , kx_covariants_regular(1, bem_values.quad_rule_for_regular.size())
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
      , ky_covariants_same_panel(8, bem_values.quad_rule_for_same_panel.size())
      , ky_covariants_common_edge(6,
                                  bem_values.quad_rule_for_common_edge.size())
      , ky_covariants_common_vertex(
          4,
          bem_values.quad_rule_for_common_vertex.size())
      , ky_covariants_regular(1, bem_values.quad_rule_for_regular.size())
      , ky_quad_points_same_panel(8, bem_values.quad_rule_for_same_panel.size())
      , ky_quad_points_common_edge(6,
                                   bem_values.quad_rule_for_common_edge.size())
      , ky_quad_points_common_vertex(
          4,
          bem_values.quad_rule_for_common_vertex.size())
      , ky_quad_points_regular(1, bem_values.quad_rule_for_regular.size())
      , quad_values_in_thread_blocks(nullptr)
    {
#if ENABLE_DEBUG == 1 && ENABLE_TIMER == 1
      /**
       * @internal Stop the timer at the moment, which will be started
       * afterwards when needed.
       */
      timer.stop();
#endif

      common_vertex_pair_local_indices.reserve(
        GeometryInfo<dim>::vertices_per_cell);

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

      kx_mapping_poly_space_numbering_inverse =
        FETools::lexicographic_to_hierarchic_numbering<dim>(
          kx_mapping.get_degree());
      ky_mapping_poly_space_numbering_inverse =
        FETools::lexicographic_to_hierarchic_numbering<dim>(
          ky_mapping.get_degree());
      generate_backward_mapping_support_point_permutation(
        ky_mapping, 0, ky_mapping_reversed_poly_space_numbering_inverse);
    }

    /**
     * Constructor for assembling hierarchical BEM matrix, which involves
     * @p thread_id.
     *
     * @param thread_id
     * @param kx_fe
     * @param ky_fe
     * @param kx_mapping
     * @param ky_mapping
     * @param bem_values
     */
    PairCellWiseScratchData(std::thread::id                          thread_id,
                            const FiniteElement<dim, spacedim>      &kx_fe,
                            const FiniteElement<dim, spacedim>      &ky_fe,
                            const MappingQGenericExt<dim, spacedim> &kx_mapping,
                            const MappingQGenericExt<dim, spacedim> &ky_mapping,
                            const BEMValues<dim, spacedim>          &bem_values)
      : thread_id(thread_id)
      , cuda_stream_handle(0)
      , common_vertex_pair_local_indices(0)
      , kx_mapping_support_points_in_default_order(
          bem_values.kx_mapping_data.n_shape_functions)
      , ky_mapping_support_points_in_default_order(
          bem_values.ky_mapping_data.n_shape_functions)
      , kx_mapping_support_points_permuted(
          bem_values.kx_mapping_data.n_shape_functions)
      , ky_mapping_support_points_permuted(
          bem_values.ky_mapping_data.n_shape_functions)
      , kx_mapping_support_points_permuted_xy_components(
          bem_values.kx_mapping_data.n_shape_functions)
      , kx_mapping_support_points_permuted_yz_components(
          bem_values.kx_mapping_data.n_shape_functions)
      , kx_mapping_support_points_permuted_zx_components(
          bem_values.kx_mapping_data.n_shape_functions)
      , ky_mapping_support_points_permuted_xy_components(
          bem_values.ky_mapping_data.n_shape_functions)
      , ky_mapping_support_points_permuted_yz_components(
          bem_values.ky_mapping_data.n_shape_functions)
      , ky_mapping_support_points_permuted_zx_components(
          bem_values.ky_mapping_data.n_shape_functions)
      , kx_local_dof_indices_in_default_dof_order(kx_fe.dofs_per_cell)
      , ky_local_dof_indices_in_default_dof_order(ky_fe.dofs_per_cell)
      , kx_fe_poly_space_numbering_inverse(kx_fe.dofs_per_cell)
      , ky_fe_poly_space_numbering_inverse(ky_fe.dofs_per_cell)
      , kx_mapping_poly_space_numbering_inverse(
          bem_values.kx_mapping_data.n_shape_functions)
      , ky_mapping_poly_space_numbering_inverse(
          bem_values.ky_mapping_data.n_shape_functions)
      , ky_mapping_reversed_poly_space_numbering_inverse(
          bem_values.ky_mapping_data.n_shape_functions)
      , kx_local_dof_permutation(kx_fe.dofs_per_cell)
      , ky_local_dof_permutation(ky_fe.dofs_per_cell)
      , kx_mapping_support_point_permutation(
          bem_values.kx_mapping_data.n_shape_functions)
      , ky_mapping_support_point_permutation(
          bem_values.ky_mapping_data.n_shape_functions)
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
      , kx_covariants_same_panel(8, bem_values.quad_rule_for_same_panel.size())
      , kx_covariants_common_edge(6,
                                  bem_values.quad_rule_for_common_edge.size())
      , kx_covariants_common_vertex(
          4,
          bem_values.quad_rule_for_common_vertex.size())
      , kx_covariants_regular(1, bem_values.quad_rule_for_regular.size())
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
      , ky_covariants_same_panel(8, bem_values.quad_rule_for_same_panel.size())
      , ky_covariants_common_edge(6,
                                  bem_values.quad_rule_for_common_edge.size())
      , ky_covariants_common_vertex(
          4,
          bem_values.quad_rule_for_common_vertex.size())
      , ky_covariants_regular(1, bem_values.quad_rule_for_regular.size())
      , ky_quad_points_same_panel(8, bem_values.quad_rule_for_same_panel.size())
      , ky_quad_points_common_edge(6,
                                   bem_values.quad_rule_for_common_edge.size())
      , ky_quad_points_common_vertex(
          4,
          bem_values.quad_rule_for_common_vertex.size())
      , ky_quad_points_regular(1, bem_values.quad_rule_for_regular.size())
      , quad_values_in_thread_blocks(nullptr)
    {
#if ENABLE_DEBUG == 1 && ENABLE_TIMER == 1
      /**
       * @internal Stop the timer at the moment, which will be started
       * afterwards when needed.
       */
      timer.stop();

      /**
       * @internal Open the log stream in append mode, in case there is an
       * existing log file with the same name.
       */
      std::stringstream thread_id_strstream;
      thread_id_strstream << thread_id;
      log_stream.open(std::string("thread-") + thread_id_strstream.str(),
                      std::ios_base::app);
#endif

      AssertCuda(cudaStreamCreate(&cuda_stream_handle));

      common_vertex_pair_local_indices.reserve(
        GeometryInfo<dim>::vertices_per_cell);

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

      kx_mapping_poly_space_numbering_inverse =
        FETools::lexicographic_to_hierarchic_numbering<dim>(
          kx_mapping.get_degree());
      ky_mapping_poly_space_numbering_inverse =
        FETools::lexicographic_to_hierarchic_numbering<dim>(
          ky_mapping.get_degree());
      generate_backward_mapping_support_point_permutation(
        ky_mapping, 0, ky_mapping_reversed_poly_space_numbering_inverse);

      /**
       * @internal Register host memory for asynchronous transfer to the device.
       */
      AssertCuda(
        cudaHostRegister((void *)kx_mapping_support_points_permuted.data(),
                         kx_mapping_support_points_permuted.size() *
                           sizeof(Point<spacedim, RangeNumberType>),
                         0));

      AssertCuda(
        cudaHostRegister((void *)ky_mapping_support_points_permuted.data(),
                         ky_mapping_support_points_permuted.size() *
                           sizeof(Point<spacedim, RangeNumberType>),
                         0));

      AssertCuda(cudaHostRegister(
        (void *)kx_mapping_support_points_permuted_xy_components.data(),
        kx_mapping_support_points_permuted_xy_components.size() *
          sizeof(Point<2, RangeNumberType>),
        0));

      AssertCuda(cudaHostRegister(
        (void *)kx_mapping_support_points_permuted_yz_components.data(),
        kx_mapping_support_points_permuted_yz_components.size() *
          sizeof(Point<2, RangeNumberType>),
        0));

      AssertCuda(cudaHostRegister(
        (void *)kx_mapping_support_points_permuted_zx_components.data(),
        kx_mapping_support_points_permuted_zx_components.size() *
          sizeof(Point<2, RangeNumberType>),
        0));

      AssertCuda(cudaHostRegister(
        (void *)ky_mapping_support_points_permuted_xy_components.data(),
        ky_mapping_support_points_permuted_xy_components.size() *
          sizeof(Point<2, RangeNumberType>),
        0));

      AssertCuda(cudaHostRegister(
        (void *)ky_mapping_support_points_permuted_yz_components.data(),
        ky_mapping_support_points_permuted_yz_components.size() *
          sizeof(Point<2, RangeNumberType>),
        0));

      AssertCuda(cudaHostRegister(
        (void *)ky_mapping_support_points_permuted_zx_components.data(),
        ky_mapping_support_points_permuted_zx_components.size() *
          sizeof(Point<2, RangeNumberType>),
        0));

      AssertCuda(cudaHostRegister((void *)&(kx_quad_points_same_panel(0, 0)),
                                  kx_quad_points_same_panel.n_elements() *
                                    sizeof(Point<spacedim, RangeNumberType>),
                                  0));

      AssertCuda(cudaHostRegister((void *)&(kx_quad_points_common_edge(0, 0)),
                                  kx_quad_points_common_edge.n_elements() *
                                    sizeof(Point<spacedim, RangeNumberType>),
                                  0));

      AssertCuda(cudaHostRegister((void *)&(kx_quad_points_common_vertex(0, 0)),
                                  kx_quad_points_common_vertex.n_elements() *
                                    sizeof(Point<spacedim, RangeNumberType>),
                                  0));

      AssertCuda(cudaHostRegister((void *)&(kx_quad_points_regular(0, 0)),
                                  kx_quad_points_regular.n_elements() *
                                    sizeof(Point<spacedim, RangeNumberType>),
                                  0));

      AssertCuda(cudaHostRegister((void *)&(ky_quad_points_same_panel(0, 0)),
                                  ky_quad_points_same_panel.n_elements() *
                                    sizeof(Point<spacedim, RangeNumberType>),
                                  0));

      AssertCuda(cudaHostRegister((void *)&(ky_quad_points_common_edge(0, 0)),
                                  ky_quad_points_common_edge.n_elements() *
                                    sizeof(Point<spacedim, RangeNumberType>),
                                  0));

      AssertCuda(cudaHostRegister((void *)&(ky_quad_points_common_vertex(0, 0)),
                                  ky_quad_points_common_vertex.n_elements() *
                                    sizeof(Point<spacedim, RangeNumberType>),
                                  0));

      AssertCuda(cudaHostRegister((void *)&(ky_quad_points_regular(0, 0)),
                                  ky_quad_points_regular.n_elements() *
                                    sizeof(Point<spacedim, RangeNumberType>),
                                  0));

      AssertCuda(cudaMallocHost((void **)&quad_values_in_thread_blocks,
                                100 * sizeof(RangeNumberType)));
    }


    /**
     * Copy constructor
     *
     * @param scratch
     */
    PairCellWiseScratchData(
      const PairCellWiseScratchData<dim, spacedim, RangeNumberType> &scratch)
      : thread_id(scratch.thread_id)
      , cuda_stream_handle(scratch.cuda_stream_handle)
      , common_vertex_pair_local_indices(
          scratch.common_vertex_pair_local_indices)
      , kx_mapping_support_points_in_default_order(
          scratch.kx_mapping_support_points_in_default_order)
      , ky_mapping_support_points_in_default_order(
          scratch.ky_mapping_support_points_in_default_order)
      , kx_mapping_support_points_permuted(
          scratch.kx_mapping_support_points_permuted)
      , ky_mapping_support_points_permuted(
          scratch.ky_mapping_support_points_permuted)
      , kx_mapping_support_points_permuted_xy_components(
          scratch.kx_mapping_support_points_permuted_xy_components)
      , kx_mapping_support_points_permuted_yz_components(
          scratch.kx_mapping_support_points_permuted_yz_components)
      , kx_mapping_support_points_permuted_zx_components(
          scratch.kx_mapping_support_points_permuted_zx_components)
      , ky_mapping_support_points_permuted_xy_components(
          scratch.ky_mapping_support_points_permuted_xy_components)
      , ky_mapping_support_points_permuted_yz_components(
          scratch.ky_mapping_support_points_permuted_yz_components)
      , ky_mapping_support_points_permuted_zx_components(
          scratch.ky_mapping_support_points_permuted_zx_components)
      , kx_local_dof_indices_in_default_dof_order(
          scratch.kx_local_dof_indices_in_default_dof_order)
      , ky_local_dof_indices_in_default_dof_order(
          scratch.ky_local_dof_indices_in_default_dof_order)
      , kx_fe_poly_space_numbering_inverse(
          scratch.kx_fe_poly_space_numbering_inverse)
      , ky_fe_poly_space_numbering_inverse(
          scratch.ky_fe_poly_space_numbering_inverse)
      , kx_mapping_poly_space_numbering_inverse(
          scratch.kx_mapping_poly_space_numbering_inverse)
      , ky_mapping_poly_space_numbering_inverse(
          scratch.ky_mapping_poly_space_numbering_inverse)
      , ky_mapping_reversed_poly_space_numbering_inverse(
          scratch.ky_mapping_reversed_poly_space_numbering_inverse)
      , kx_local_dof_permutation(scratch.kx_local_dof_permutation)
      , ky_local_dof_permutation(scratch.ky_local_dof_permutation)
      , kx_mapping_support_point_permutation(
          scratch.kx_mapping_support_point_permutation)
      , ky_mapping_support_point_permutation(
          scratch.ky_mapping_support_point_permutation)
      , kx_jacobians_same_panel(scratch.kx_jacobians_same_panel)
      , kx_jacobians_common_edge(scratch.kx_jacobians_common_edge)
      , kx_jacobians_common_vertex(scratch.kx_jacobians_common_vertex)
      , kx_jacobians_regular(scratch.kx_jacobians_regular)
      , kx_normals_same_panel(scratch.kx_normals_same_panel)
      , kx_normals_common_edge(scratch.kx_normals_common_edge)
      , kx_normals_common_vertex(scratch.kx_normals_common_vertex)
      , kx_normals_regular(scratch.kx_normals_regular)
      , kx_covariants_same_panel(scratch.kx_covariants_same_panel)
      , kx_covariants_common_edge(scratch.kx_covariants_common_edge)
      , kx_covariants_common_vertex(scratch.kx_covariants_common_vertex)
      , kx_covariants_regular(scratch.kx_covariants_regular)
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
      , ky_covariants_same_panel(scratch.ky_covariants_same_panel)
      , ky_covariants_common_edge(scratch.ky_covariants_common_edge)
      , ky_covariants_common_vertex(scratch.ky_covariants_common_vertex)
      , ky_covariants_regular(scratch.ky_covariants_regular)
      , ky_quad_points_same_panel(scratch.ky_quad_points_same_panel)
      , ky_quad_points_common_edge(scratch.ky_quad_points_common_edge)
      , ky_quad_points_common_vertex(scratch.ky_quad_points_common_vertex)
      , ky_quad_points_regular(scratch.ky_quad_points_regular)
      , quad_values_in_thread_blocks(scratch.quad_values_in_thread_blocks)
    {}

    void
    release()
    {
#if ENABLE_DEBUG == 1 && ENABLE_TIMER == 1
      log_stream.close();
#endif

      AssertCuda(cudaStreamDestroy(cuda_stream_handle));

      AssertCuda(
        cudaHostUnregister((void *)kx_mapping_support_points_permuted.data()));

      AssertCuda(
        cudaHostUnregister((void *)ky_mapping_support_points_permuted.data()));

      AssertCuda(cudaHostUnregister(
        (void *)kx_mapping_support_points_permuted_xy_components.data()));

      AssertCuda(cudaHostUnregister(
        (void *)kx_mapping_support_points_permuted_yz_components.data()));

      AssertCuda(cudaHostUnregister(
        (void *)kx_mapping_support_points_permuted_zx_components.data()));

      AssertCuda(cudaHostUnregister(
        (void *)ky_mapping_support_points_permuted_xy_components.data()));

      AssertCuda(cudaHostUnregister(
        (void *)ky_mapping_support_points_permuted_yz_components.data()));

      AssertCuda(cudaHostUnregister(
        (void *)ky_mapping_support_points_permuted_zx_components.data()));

      AssertCuda(
        cudaHostUnregister((void *)&(kx_quad_points_same_panel(0, 0))));

      AssertCuda(
        cudaHostUnregister((void *)&(kx_quad_points_common_edge(0, 0))));

      AssertCuda(
        cudaHostUnregister((void *)&(kx_quad_points_common_vertex(0, 0))));

      AssertCuda(cudaHostUnregister((void *)&(kx_quad_points_regular(0, 0))));

      AssertCuda(
        cudaHostUnregister((void *)&(ky_quad_points_same_panel(0, 0))));

      AssertCuda(
        cudaHostUnregister((void *)&(ky_quad_points_common_edge(0, 0))));

      AssertCuda(
        cudaHostUnregister((void *)&(ky_quad_points_common_vertex(0, 0))));

      AssertCuda(cudaHostUnregister((void *)&(ky_quad_points_regular(0, 0))));

      if (quad_values_in_thread_blocks != nullptr)
        {
          AssertCuda(cudaFreeHost(quad_values_in_thread_blocks));
        }
    }
  };


  template <int dim, int spacedim = dim, typename RangeNumberType = double>
  class PairCellWisePerTaskData
  {
  public:
    /**
     * Local matrix for the pair of cells to be assembled into the global full
     * matrix representation of the boundary integral operator.
     *
     * \comment{Therefore, this data field is only defined for verification.}
     */
    LAPACKFullMatrixExt<RangeNumberType> local_pair_cell_matrix;

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
    {
      AssertCuda(cudaHostRegister((void *)kx_local_dof_indices_permuted.data(),
                                  kx_local_dof_indices_permuted.size() *
                                    sizeof(types::global_dof_index),
                                  0));

      AssertCuda(cudaHostRegister((void *)ky_local_dof_indices_permuted.data(),
                                  ky_local_dof_indices_permuted.size() *
                                    sizeof(types::global_dof_index),
                                  0));
    }


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

    void
    release()
    {
      AssertCuda(
        cudaHostUnregister((void *)kx_local_dof_indices_permuted.data()));

      AssertCuda(
        cudaHostUnregister((void *)ky_local_dof_indices_permuted.data()));
    }
  };
} // namespace HierBEM

#endif /* INCLUDE_BEM_VALUES_H_ */
