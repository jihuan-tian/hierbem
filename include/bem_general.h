/**
 * @file bem_general.h
 * @brief Introduction of bem_general.h
 *
 * @date 2022-03-04
 * @author Jihuan Tian
 */
#ifndef INCLUDE_BEM_GENERAL_H_
#define INCLUDE_BEM_GENERAL_H_

#include <deal.II/base/quadrature.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/dofs/dof_handler.h>

#include <boost/progress.hpp>

#include <functional>
#include <map>
#include <utility>
#include <vector>

#include "bem_kernels.h"
#include "bem_tools.h"
#include "bem_values.h"
#include "sauter_quadrature.h"

namespace IdeoBEM
{
  using namespace dealii;
  using namespace BEMTools;

  template <int dim, int spacedim, typename RangeNumberType>
  void
  assemble_fem_scaled_mass_matrix_on_one_cell(
    const RangeNumberType factor,
    const typename std::vector<std::pair<
      typename DoFHandler<dim, spacedim>::active_cell_iterator,
      typename DoFHandler<dim, spacedim>::active_cell_iterator>>::const_iterator
      &                                 iterator_for_cell_iterator_pairs,
    CellWiseScratchData<dim, spacedim> &scratch,
    CellWisePerTaskData<dim, spacedim, RangeNumberType> &data)
  {
    /**
     * Clear the local matrix in case that it is reused from another finished
     * task. N.B. Its memory has already been allocated in the constructor of
     * @p CellWisePerTaskData.
     */
    data.local_matrix.reinit(data.local_dof_indices_for_test_space.size(),
                             data.local_dof_indices_for_trial_space.size());

    /**
     * N.B. The construction of the object <code>scratch.fe_values</code> is
     * carried out in the constructor of <code>CellWiseScratchData</code>.
     *
     * \comment{2022-06-27 I added a @p const keyword at the front to protect
     * the internal data in the cell. Since the vector of cell iterator pairs
     * persists at least in this function, I create references to the two cell
     * iterators instead of making copies.}
     */
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &
      cell_iter_for_test_space_domain = iterator_for_cell_iterator_pairs->first;
    const typename DoFHandler<dim, spacedim>::active_cell_iterator
      &cell_iter_for_trial_space_domain =
        iterator_for_cell_iterator_pairs->second;

    /**
     * Reinitialize the @p FEValues objects for test space and trial space
     * for the current cell.
     */
    scratch.fe_values_for_test_space.reinit(cell_iter_for_test_space_domain);
    scratch.fe_values_for_trial_space.reinit(cell_iter_for_trial_space_domain);

    AssertDimension(scratch.fe_values_for_test_space.get_quadrature().size(),
                    scratch.fe_values_for_trial_space.get_quadrature().size());
    const unsigned int n_q_points =
      scratch.fe_values_for_test_space.get_quadrature().size();

    const unsigned int dofs_per_cell_for_test_space =
      scratch.fe_values_for_test_space.get_fe().dofs_per_cell;
    const unsigned int dofs_per_cell_for_trial_space =
      scratch.fe_values_for_trial_space.get_fe().dofs_per_cell;

    /**
     * Calculate the local mass matrix multiplied by a factor in the
     * current cell.
     */
    for (unsigned int q = 0; q < n_q_points; q++)
      {
        Assert(
          scratch.fe_values_for_test_space.JxW(q) ==
            scratch.fe_values_for_trial_space.JxW(q),
          ExcMessage(
            "The JxW values in test space domain and trial space domain should be the same!"));

        /**
         * Iterate over test function DoFs.
         */
        for (unsigned int i = 0; i < dofs_per_cell_for_test_space; i++)
          {
            /**
             * Iterate over trial function DoFs.
             */
            for (unsigned int j = 0; j < dofs_per_cell_for_trial_space; j++)
              {
                data.local_matrix(i, j) +=
                  factor * scratch.fe_values_for_test_space.shape_value(i, q) *
                  scratch.fe_values_for_trial_space.shape_value(j, q) *
                  scratch.fe_values_for_test_space.JxW(q);
              }
          }
      }

    /**
     *  Extract the DoF indices. N.B. Before calling
     * <code>get_dof_indices</code>, the memory for the argument vector should
     * have been allocated. Here, the memory for
     * <code>data.local_dof_indices</code> has been allocated in the
     * constructor of <code>CellWisePerTaskData</code>.
     */
    cell_iter_for_test_space_domain->get_dof_indices(
      data.local_dof_indices_for_test_space);
    cell_iter_for_trial_space_domain->get_dof_indices(
      data.local_dof_indices_for_trial_space);
  }

  template <int dim,
            int spacedim,
            typename RangeNumberType,
            typename MatrixType>
  void
  copy_cell_local_to_global_for_fem_matrix(
    const CellWisePerTaskData<dim, spacedim, RangeNumberType> &data,
    MatrixType &target_full_matrix)
  {
    const unsigned int dofs_per_cell_for_test_space  = data.local_matrix.m();
    const unsigned int dofs_per_cell_for_trial_space = data.local_matrix.n();

    for (unsigned int i = 0; i < dofs_per_cell_for_test_space; i++)
      {
        for (unsigned int j = 0; j < dofs_per_cell_for_trial_space; j++)
          {
            target_full_matrix.add(data.local_dof_indices_for_test_space[i],
                                   data.local_dof_indices_for_trial_space[j],
                                   data.local_matrix(i, j));
          }
      }
  }


  /**
   * Initialize the cell iterator pointer pairs for the DoFHandlers
   * associated with test space and trial space. N.B. The test space comes
   * before the trial space.
   *
   * \mynote{This function is to assist the assembly of the FEM mass matrix.
   * For Dirichlet problem, it is \f$\mathcal{I}\f$ on \f$\left(
   * H^{-\frac{1}{2}+s}(\Gamma_{\rm D}), H^{\frac{1}{2}+s}(\Gamma_{\rm D})
   * \right) \f$.
   *
   * For mixed boundary value problem, they are
   * 1. \f$\mathcal{I}_1\f$ on \f$\left( H^{-\frac{1}{2}+s}(\Gamma_{\rm D}),
   * H^{\frac{1}{2}+s}(\Gamma_{\rm D}) \right) \f$
   * 2. \f$\mathcal{I}_2\f$ on \f$\left( H^{\frac{1}{2}+s}(\Gamma_{\rm N}),
   * H^{-\frac{1}{2}+s}(\Gamma_{\rm N}) \right)\f$
   *
   * It can be seen that for the mass matrices, the test space and trial
   * space are situated on a same triangulation.}
   *
   * @param dof_handler_for_test_space
   * @param dof_handler_for_trial_space
   * @param cell_iterator_pairs_for_mass_matrix
   */
  template <int dim, int spacedim>
  void
  initialize_cell_iterator_pairs_for_mass_matrix(
    const DoFHandler<dim, spacedim> &dof_handler_for_test_space,
    const DoFHandler<dim, spacedim> &dof_handler_for_trial_space,
    std::vector<
      std::pair<typename DoFHandler<dim, spacedim>::active_cell_iterator,
                typename DoFHandler<dim, spacedim>::active_cell_iterator>>
      &cell_iterator_pairs_for_mass_matrix)
  {
    /**
     * Because the two spaces are associated with a same triangulation, the
     * number of cells inferred from the two DoFHandlers should be the same.
     */
    AssertDimension(
      dof_handler_for_test_space.get_triangulation().n_active_cells(),
      dof_handler_for_trial_space.get_triangulation().n_active_cells());

    /**
     * The memory for the result vector should be preallocated.
     */
    AssertDimension(
      dof_handler_for_test_space.get_triangulation().n_active_cells(),
      cell_iterator_pairs_for_mass_matrix.size());

    typename DoFHandler<dim, spacedim>::active_cell_iterator
      cell_iterator_for_test_space = dof_handler_for_test_space.begin_active();
    typename DoFHandler<dim, spacedim>::active_cell_iterator
      cell_iterator_for_trial_space =
        dof_handler_for_trial_space.begin_active();

    std::size_t counter = 0;
    for (; cell_iterator_for_test_space != dof_handler_for_test_space.end();
         cell_iterator_for_test_space++,
         cell_iterator_for_trial_space++,
         counter++)
      {
        /**
         * \mynote{N.B. The cell iterator for the test space appears before
         * that for the trial space in the pair.}
         */
        cell_iterator_pairs_for_mass_matrix[counter].first =
          cell_iterator_for_test_space;
        cell_iterator_pairs_for_mass_matrix[counter].second =
          cell_iterator_for_trial_space;
      }
  }


  template <int dim,
            int spacedim,
            typename RangeNumberType,
            typename MatrixType>
  void
  assemble_fem_scaled_mass_matrix(
    const DoFHandler<dim, spacedim> &dof_handler_for_test_space,
    const DoFHandler<dim, spacedim> &dof_handler_for_trial_space,
    const RangeNumberType            factor,
    const Quadrature<dim> &          quad_rule,
    MatrixType &                     target_full_matrix)
  {
    AssertDimension(
      dof_handler_for_test_space.get_triangulation().n_active_cells(),
      dof_handler_for_trial_space.get_triangulation().n_active_cells());

    std::vector<
      std::pair<typename DoFHandler<dim, spacedim>::active_cell_iterator,
                typename DoFHandler<dim, spacedim>::active_cell_iterator>>
      cell_iterator_pairs_for_mass_matrix(
        dof_handler_for_test_space.get_triangulation().n_active_cells());

    initialize_cell_iterator_pairs_for_mass_matrix(
      dof_handler_for_test_space,
      dof_handler_for_trial_space,
      cell_iterator_pairs_for_mass_matrix);

    WorkStream::run(
      cell_iterator_pairs_for_mass_matrix.begin(),
      cell_iterator_pairs_for_mass_matrix.end(),
      std::bind(&assemble_fem_scaled_mass_matrix_on_one_cell<dim,
                                                             spacedim,
                                                             RangeNumberType>,
                factor,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3),
      std::bind(&copy_cell_local_to_global_for_fem_matrix<dim,
                                                          spacedim,
                                                          RangeNumberType,
                                                          MatrixType>,

                std::placeholders::_1,
                std::ref(target_full_matrix)),
      CellWiseScratchData<dim, spacedim>(dof_handler_for_test_space.get_fe(),
                                         dof_handler_for_trial_space.get_fe(),
                                         quad_rule,
                                         update_values | update_JxW_values),
      CellWisePerTaskData<dim, spacedim, RangeNumberType>(
        dof_handler_for_test_space.get_fe(),
        dof_handler_for_trial_space.get_fe()));
  }


  template <int dim,
            int spacedim,
            typename RangeNumberType,
            typename MatrixType>
  void
  copy_pair_of_cells_local_to_global_for_bem_full_matrix(
    const PairCellWisePerTaskData<dim, spacedim, RangeNumberType> &data,
    MatrixType &target_full_matrix)
  {
    const unsigned int kx_dofs_per_cell = data.local_pair_cell_matrix.m();
    const unsigned int ky_dofs_per_cell = data.local_pair_cell_matrix.n();

    for (unsigned int i = 0; i < kx_dofs_per_cell; i++)
      {
        for (unsigned int j = 0; j < ky_dofs_per_cell; j++)
          {
            target_full_matrix.add(data.kx_local_dof_indices_permuted[i],
                                   data.ky_local_dof_indices_permuted[j],
                                   data.local_pair_cell_matrix(i, j));
          }
      }
  }

  template <int dim,
            int spacedim,
            typename RangeNumberType,
            typename MatrixType>
  void
  assemble_bem_full_matrix(
    const KernelFunction<spacedim, RangeNumberType> &kernel,
    const RangeNumberType                            factor,
    const DoFHandler<dim, spacedim> &                dof_handler_for_test_space,
    const DoFHandler<dim, spacedim> &  dof_handler_for_trial_space,
    MappingQGenericExt<dim, spacedim> &kx_mapping,
    MappingQGenericExt<dim, spacedim> &ky_mapping,
    typename MappingQGeneric<dim, spacedim>::InternalData &kx_mapping_data,
    typename MappingQGeneric<dim, spacedim>::InternalData &ky_mapping_data,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_test_space_mesh_to_volume_mesh,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_trial_space_mesh_to_volume_mesh,
    const DetectCellNeighboringTypeMethod method_for_cell_neighboring_type,
    const SauterQuadratureRule<dim> &     sauter_quad_rule,
    MatrixType &                          target_full_matrix)
  {
    /**
     * Precalculate data tables for shape values at quadrature points.
     *
     * \mynote{Precalculate shape function values and their gradient
     * values at each quadrature point. N.B.
     * 1. The data tables for shape function values and their gradient
     * values should be calculated for both function space on
     * \f$K_x\f$ and function space on \f$K_y\f$.
     * 2. Being different from the integral in FEM, the integral in
     * BEM handled by Sauter's quadrature rule has multiple parts of
     * \f$k_3\f$ (except the regular cell neighboring type), each of
     * which should be evaluated at a different set of quadrature
     * points in the unit cell after coordinate transformation from
     * the parametric space. Therefore, a dimension with respect to
     * \f$k_3\f$ term index should be added to the data table compared
     * to the usual FEValues and this brings about
     * the class @p BEMValues.
     * 3. In Galerkin BEM, finite elements for the Dirichlet domain
     * and Neumann domain are different. For SLP BEM matrix, both the
     * test function space, to which \f$K_x\f$ belongs, space and the
     * trial function space, to which \f$K_y\f$ belongs, is @p FE_DGQ.
     * For DLP BEM matrix and the mass matrix, the test function space
     * is @p FE_DGQ and the trial function space is @p FE_Q.}
     */
    BEMValues<dim, spacedim, RangeNumberType> bem_values(
      dof_handler_for_test_space.get_fe(),
      dof_handler_for_trial_space.get_fe(),
      kx_mapping_data,
      ky_mapping_data,
      sauter_quad_rule.quad_rule_for_same_panel,
      sauter_quad_rule.quad_rule_for_common_edge,
      sauter_quad_rule.quad_rule_for_common_vertex,
      sauter_quad_rule.quad_rule_for_regular);

    bem_values.fill_shape_function_value_tables();

    /**
     * Create data structure for parallel matrix assembly.
     *
     * \alert{Since @p scratch_data and @p per_task_data should be copied to
     * each thread and will further be modified in the working
     * function
     * @p assemble_on_one_pair_of_cells, they should be passed-by-value.}
     */
    PairCellWiseScratchData<dim, spacedim, RangeNumberType> scratch_data(
      dof_handler_for_test_space.get_fe(),
      dof_handler_for_trial_space.get_fe(),
      kx_mapping,
      ky_mapping,
      kx_mapping_data,
      ky_mapping_data,
      bem_values);
    PairCellWisePerTaskData<dim, spacedim, RangeNumberType> per_task_data(
      dof_handler_for_test_space.get_fe(),
      dof_handler_for_trial_space.get_fe());

    boost::progress_display pd(
      dof_handler_for_test_space.get_triangulation().n_active_cells(),
      std::cerr);

    for (const auto &e : dof_handler_for_test_space.active_cell_iterators())
      {
        /**
         * Calculate Kx related data here.
         */
        // kx_mapping.compute_mapping_support_points(e);

        /**
         * Apply parallelization to the inner loop.
         *
         * \alert{@p bem_values will not be modified inside the working
         * function, so it is safe to pass it by const reference in the call of
         * @p std::bind.}
         */
        WorkStream::run(
          dof_handler_for_trial_space.begin_active(),
          dof_handler_for_trial_space.end(),
          std::bind(&sauter_assemble_on_one_pair_of_cells<dim,
                                                          spacedim,
                                                          RangeNumberType>,
                    std::cref(kernel),
                    factor,
                    std::cref(e),
                    std::placeholders::_1,
                    std::cref(kx_mapping),
                    std::cref(ky_mapping),
                    std::cref(map_from_test_space_mesh_to_volume_mesh),
                    std::cref(map_from_trial_space_mesh_to_volume_mesh),
                    method_for_cell_neighboring_type,
                    std::cref(bem_values),
                    std::placeholders::_2,
                    std::placeholders::_3),
          std::bind(&copy_pair_of_cells_local_to_global_for_bem_full_matrix<
                      dim,
                      spacedim,
                      RangeNumberType,
                      MatrixType>,
                    std::placeholders::_1,
                    std::ref(target_full_matrix)),
          scratch_data,
          per_task_data);

        ++pd;
      }
  }


  /**
   * Evaluate the integral of the product of the given kernel function and the
   * list of basis functions on the current cell.
   *
   * @param kernel
   * @param target_point
   * @param factor
   * @param cell_iter
   * @param mapping
   * @param is_normal_vector_negated
   * @param scratch
   * @param data
   */
  template <int dim, int spacedim, typename RangeNumberType>
  void
  evaluate_potential_on_one_cell(
    const KernelFunction<spacedim, RangeNumberType> &kernel,
    const Point<spacedim, RangeNumberType> &         target_point,
    const RangeNumberType                            factor,
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell_iter,
    const bool is_normal_vector_negated,
    CellWiseScratchDataForPotentialEval<dim, spacedim, RangeNumberType>
      &scratch,
    CellWisePerTaskDataForPotentialEval<dim, spacedim, RangeNumberType> &data)
  {
    /**
     * Clear the local result vector.
     */
    data.local_vector.reinit(data.local_dof_indices_for_trial_space.size());

    /**
     * Reinitialize the @p FEValues object for the trial space on the current
     * cell.
     */
    scratch.fe_values_for_trial_space.reinit(cell_iter);

    const unsigned int n_q_points =
      scratch.fe_values_for_trial_space.get_quadrature().size();

    const unsigned int dofs_per_cell =
      scratch.fe_values_for_trial_space.get_fe().dofs_per_cell;

    /**
     * Iterate over each quadrature point.
     */
    for (unsigned int q = 0; q < n_q_points; q++)
      {
        /**
         * Evaluate the kernel function at the target point (for \f$x\f$) and
         * the current quadrature point (for \f$y\f$), i.e. only the \f$y\f$
         * coordinates are pulled back to the unit cell.
         *
         * \mynote{When calculating the potential at a target point, only the
         * normal vector on \f$K_y\f$ is needed. Therefore, the normal vector
         * on \f$K_x\f$ passed to the kernel function can be arbitrary.}
         */
        RangeNumberType scaled_kernel_value =
          kernel.value(target_point,
                       scratch.fe_values_for_trial_space.quadrature_point(q),
                       Tensor<1, spacedim>(),
                       (is_normal_vector_negated ? -1.0 : 1.0) *
                         scratch.fe_values_for_trial_space.normal_vector(q)) *
          factor;

        /**
         * Iterate over each shape function.
         */
        for (unsigned int i = 0; i < dofs_per_cell; i++)
          {
            data.local_vector(i) +=
              scaled_kernel_value *
              scratch.fe_values_for_trial_space.shape_value(i, q) *
              scratch.fe_values_for_trial_space.JxW(q);
          }
      }

    /**
     * Extract DoF indices on the cell.
     */
    cell_iter->get_dof_indices(data.local_dof_indices_for_trial_space);
  }


  template <int dim,
            int spacedim,
            typename RangeNumberType,
            typename VectorType>
  void
  copy_cell_local_to_global_for_potential_eval(
    const CellWisePerTaskDataForPotentialEval<dim, spacedim, RangeNumberType>
      &         data,
    VectorType &result_vector)
  {
    result_vector.add(data.local_dof_indices_for_trial_space,
                      data.local_vector);
  }


  /**
   * Evaluate the potential values in the spatial domain at a list of target
   * points using the representation formula derived from the direct method.
   *
   * \f[
   * u(x) = -\int_{\Gamma} \widetilde{\gamma}_{1,y} G(x,y) \gamma_0^{\rm int}
   * u(y) \intd s_y + \int_{\Gamma} G(x,y) \widetilde{\gamma}_{1,y} u(y) \intd
   * s_y
   * \]
   *
   * @param kernel
   * @param factor
   * @param dof_handler_for_trial_space DoF handler related to either the
   * Dirichlet data or Neumann data on the boundary.
   * @param dof_values_in_trial_space DoF values related to either the
   * Dirichlet data or Neumann data on the boundary.
   * @param mapping
   * @param is_normal_vector_negated When the boundary surface mesh is
   * extracted from a volume mesh and an exterior problem is solved, this flag
   * should be true.
   * @param point_list
   * @param potential_values
   */
  template <int dim,
            int spacedim,
            typename RangeNumberType,
            typename VectorType>
  void
  evaluate_potential_at_points(
    const KernelFunction<spacedim, RangeNumberType> &kernel,
    const RangeNumberType                            factor,
    const DoFHandler<dim, spacedim> &dof_handler_for_trial_space,
    const VectorType &               dof_values_in_trial_space,
    const bool                       is_normal_vector_negated,
    const std::vector<Point<spacedim, RangeNumberType>> &target_point_list,
    VectorType &                                         potential_values)
  {
    AssertDimension(target_point_list.size(), potential_values.size());

    const FiniteElement<dim, spacedim> &fe =
      dof_handler_for_trial_space.get_fe();

    const types::global_dof_index n_dofs = dof_handler_for_trial_space.n_dofs();
    VectorType                    kernel_evaluation_discretized_to_dofs(n_dofs);

    QGauss<dim> quad_rule(fe.degree + 1);

    /**
     * Iterate over each target point.
     */
    const unsigned int      n_target_points = target_point_list.size();
    boost::progress_display pd(n_target_points, std::cerr);

    for (unsigned int i = 0; i < n_target_points; i++)
      {
        kernel_evaluation_discretized_to_dofs.reinit(n_dofs);

        WorkStream::run(
          dof_handler_for_trial_space.begin_active(),
          dof_handler_for_trial_space.end(),
          std::bind(
            &evaluate_potential_on_one_cell<dim, spacedim, RangeNumberType>,
            std::cref(kernel),
            std::cref(target_point_list[i]),
            factor,
            std::placeholders::_1,
            is_normal_vector_negated,
            std::placeholders::_2,
            std::placeholders::_3),
          std::bind(
            &copy_cell_local_to_global_for_potential_eval<dim,
                                                          spacedim,
                                                          RangeNumberType,
                                                          VectorType>,
            std::placeholders::_1,
            std::ref(kernel_evaluation_discretized_to_dofs)),
          CellWiseScratchDataForPotentialEval<dim, spacedim, RangeNumberType>(
            fe,
            quad_rule,
            update_quadrature_points | update_values | update_JxW_values |
              update_normal_vectors),
          CellWisePerTaskDataForPotentialEval<dim, spacedim, RangeNumberType>(
            fe));

        /**
         * Calculate the inner product of the kernel evaluation and DoF values.
         * N.B. The implementation here appends the results to the result
         * vector, so that a single vector can accumulate the contribution from
         * both the single layer and double layer potentials.
         */
        potential_values(i) +=
          kernel_evaluation_discretized_to_dofs * dof_values_in_trial_space;

        ++pd;
      }
  }
} // namespace IdeoBEM


#endif /* INCLUDE_BEM_GENERAL_H_ */
