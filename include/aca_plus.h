/**
 * @file aca_plus.h
 * @brief Implement the ACA+ method proposed in Grasedyck, L. 2005. “Adaptive
 * Recompression of \f$\mathscr{H}\f$-Matrices for BEM.” Computing 74 (3):
 * 205–23.
 *
 * @date 2022-03-07
 * @author Jihuan Tian
 */
#ifndef INCLUDE_ACA_PLUS_H_
#define INCLUDE_ACA_PLUS_H_

#include <deal.II/base/parallel.h>

#include <deal.II/lac/vector.h>

#include <algorithm>
#include <cmath>
#include <forward_list>
#include <functional>
#include <iterator>
#include <random>
#include <vector>

#include "bem_kernels.h"
#include "generic_functors.h"
#include "hmatrix.h"
#include "lapack_full_matrix_ext.h"
#include "linalg.h"
#include "sauter_quadrature.h"
#include "unary_template_arg_containers.h"

namespace IdeoBEM
{
  using namespace dealii;

  /**
   * Global declaration of the random number device and engine.
   */
  extern std::random_device rd;
#ifdef DEAL_II_WITH_64BIT_INDICES
  extern std::mt19937_64 rand_engine;
#else
  extern std::mt19937 rand_engine;
#endif


  /**
   * Configuration for ACA+.
   */
  struct ACAConfig
  {
    ACAConfig();
    ACAConfig(unsigned int v_max_iter, double v_epsilon, double v_eta);

    /**
     * Maximum number of iteration, which is also the maximum rank \f$k\f$ for
     * the far field matrix block to be built.
     */
    unsigned int max_iter;
    /**
     * Relative error between the current cross and the approximant matrix
     * \f$S\f$, i.e. \f[ \norm{u_k}_2\norm{v_k}_2 \leq
     * \frac{\varepsilon(1-\eta)}{1+\varepsilon} \norm{S}_{\rm F}. \f]
     */
    double epsilon;
    /**
     * Admissibility constant
     */
    double eta;
  };


  /**
   * The type used for matrix row and column indices.
   */
  using size_type = std::make_unsigned<types::blas_int>::type;

  /**
   * Generate a random non-negative integer in the specified range \f$[a,b]\f$
   * using the global random number engine @p rand_engine.
   *
   * @param a
   * @param b
   * @return
   */
  size_type
  generate_random_index(const size_type a, const size_type b);

  /**
   * Assemble a row vector by evaluating the Galerkin-BEM double integral with
   * respect to the kernel.
   *
   * \mynote{The memory for @p row_vector should be preallocated, since inside
   * the function there is no reinitialization of this vector anymore.}
   *
   * @param row_vector [out]
   * @param scratch
   * @param data
   * @param kernel
   * @param row_dof_index The current row DoF index.
   * @param column_dof_indices The list of column DoF indices.
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  assemble_kernel_row(
    Vector<RangeNumberType> &                        row_vector,
    const KernelFunction<spacedim> &                 kernel,
    const RangeNumberType                            factor,
    const types::global_dof_index                    row_dof_index,
    const std::vector<types::global_dof_index> &     column_dof_indices,
    const std::vector<std::vector<unsigned int>> &   kx_dof_to_cell_topo,
    const std::vector<std::vector<unsigned int>> &   ky_dof_to_cell_topo,
    const BEMValues<dim, spacedim, RangeNumberType> &bem_values,
    const DoFHandler<dim, spacedim> &                kx_dof_handler,
    const DoFHandler<dim, spacedim> &                ky_dof_handler,
    const std::vector<types::global_dof_index> &     kx_dof_i2e_numbering,
    const std::vector<types::global_dof_index> &     ky_dof_i2e_numbering,
    const MappingQGenericExt<dim, spacedim> &        kx_mapping,
    const MappingQGenericExt<dim, spacedim> &        ky_mapping,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_kx_boundary_mesh_to_volume_mesh,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_ky_boundary_mesh_to_volume_mesh,
    const DetectCellNeighboringTypeMethod method_for_cell_neighboring_type,
    PairCellWiseScratchData<dim, spacedim, RangeNumberType> &scratch_data,
    PairCellWisePerTaskData<dim, spacedim, RangeNumberType> &copy_data)
  {
    AssertDimension(row_vector.size(), column_dof_indices.size());

    /**
     * Iterate over each column DoF index.
     */
    for (size_type j = 0; j < column_dof_indices.size(); j++)
      {
        row_vector(j) = sauter_assemble_on_one_pair_of_dofs(
          kernel,
          factor,
          kx_dof_i2e_numbering[row_dof_index],
          ky_dof_i2e_numbering[column_dof_indices[j]],
          kx_dof_to_cell_topo,
          ky_dof_to_cell_topo,
          bem_values,
          kx_dof_handler,
          ky_dof_handler,
          kx_mapping,
          ky_mapping,
          map_from_kx_boundary_mesh_to_volume_mesh,
          map_from_ky_boundary_mesh_to_volume_mesh,
          method_for_cell_neighboring_type,
          scratch_data,
          copy_data);
      }
  }


  /**
   * Assemble a column vector by evaluating the Galerkin-BEM double integral
   * with respect to the kernel.
   *
   * \mynote{The memory for @p col_vector should be preallocated, since inside
   * the function there is no reinitialization of this vector anymore.}
   *
   * @param col_vector [out]
   * @param scratch
   * @param data
   * @param kernel
   * @param row_dof_indices The list of row DoF indices.
   * @param col_dof_index The current column DoF index.
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  assemble_kernel_column(
    Vector<RangeNumberType> &                        col_vector,
    const KernelFunction<spacedim> &                 kernel,
    const RangeNumberType                            factor,
    const std::vector<types::global_dof_index> &     row_dof_indices,
    const types::global_dof_index                    col_dof_index,
    const std::vector<std::vector<unsigned int>> &   kx_dof_to_cell_topo,
    const std::vector<std::vector<unsigned int>> &   ky_dof_to_cell_topo,
    const BEMValues<dim, spacedim, RangeNumberType> &bem_values,
    const DoFHandler<dim, spacedim> &                kx_dof_handler,
    const DoFHandler<dim, spacedim> &                ky_dof_handler,
    const std::vector<types::global_dof_index> &     kx_dof_i2e_numbering,
    const std::vector<types::global_dof_index> &     ky_dof_i2e_numbering,
    const MappingQGenericExt<dim, spacedim> &        kx_mapping,
    const MappingQGenericExt<dim, spacedim> &        ky_mapping,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_kx_boundary_mesh_to_volume_mesh,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_ky_boundary_mesh_to_volume_mesh,
    const DetectCellNeighboringTypeMethod method_for_cell_neighboring_type,
    PairCellWiseScratchData<dim, spacedim, RangeNumberType> &scratch_data,
    PairCellWisePerTaskData<dim, spacedim, RangeNumberType> &copy_data)
  {
    AssertDimension(col_vector.size(), row_dof_indices.size());

    /**
     * Iterate over each row DoF index.
     */
    for (size_type i = 0; i < row_dof_indices.size(); i++)
      {
        col_vector(i) = sauter_assemble_on_one_pair_of_dofs(
          kernel,
          factor,
          kx_dof_i2e_numbering[row_dof_indices[i]],
          ky_dof_i2e_numbering[col_dof_index],
          kx_dof_to_cell_topo,
          ky_dof_to_cell_topo,
          bem_values,
          kx_dof_handler,
          ky_dof_handler,
          kx_mapping,
          ky_mapping,
          map_from_kx_boundary_mesh_to_volume_mesh,
          map_from_ky_boundary_mesh_to_volume_mesh,
          method_for_cell_neighboring_type,
          scratch_data,
          copy_data);
      }
  }


  /**
   * Randomly select a reference row index from the remaining row indices for
   * the next ACA step.
   *
   * \mynote{The data type for DoF index is @p types::global_dof_index, while
   * the data type for matrix row or column index is @p size_type, which is the
   * unsigned version of @p blas::int.}
   *
   * @param row_vector The selected reference row vector, the memory of which
   * should be preallocated in order to reduce the number of times for memory
   * allocation and releasing.
   * @param scratch
   * @param data
   * @param kernel
   * @param remaining_row_indices Remaining row indices to be checked and
   * selected from, which are stored in a @p std::forward_list.
   * @param current_ref_row_index The current reference row index
   * @param row_dof_indices The list of DoF indices corresponding to the matrix
   * rows
   * @param col_dof_indices The list of DoF indices corresponding to the
   * matrix columns
   * @param dof_to_cell_topo DoF-to-cell topology
   * @param bem_values
   * @param kx_dof_handler
   * @param ky_dof_handler
   * @param kx_mapping
   * @param ky_mapping
   * @return The selected reference row index
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  size_type
  random_select_ref_row(
    Vector<RangeNumberType> &                        row_vector,
    const KernelFunction<spacedim> &                 kernel,
    const RangeNumberType                            factor,
    std::forward_list<size_type> &                   remaining_row_indices,
    const size_type                                  current_ref_row_index,
    const std::vector<types::global_dof_index> &     row_dof_indices,
    const std::vector<types::global_dof_index> &     col_dof_indices,
    const std::vector<std::vector<unsigned int>> &   kx_dof_to_cell_topo,
    const std::vector<std::vector<unsigned int>> &   ky_dof_to_cell_topo,
    const BEMValues<dim, spacedim, RangeNumberType> &bem_values,
    const DoFHandler<dim, spacedim> &                kx_dof_handler,
    const DoFHandler<dim, spacedim> &                ky_dof_handler,
    const std::vector<types::global_dof_index> &     kx_dof_i2e_numbering,
    const std::vector<types::global_dof_index> &     ky_dof_i2e_numbering,
    const MappingQGenericExt<dim, spacedim> &        kx_mapping,
    const MappingQGenericExt<dim, spacedim> &        ky_mapping,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_kx_boundary_mesh_to_volume_mesh,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_ky_boundary_mesh_to_volume_mesh,
    const DetectCellNeighboringTypeMethod method_for_cell_neighboring_type,
    PairCellWiseScratchData<dim, spacedim, RangeNumberType> &scratch_data,
    PairCellWisePerTaskData<dim, spacedim, RangeNumberType> &copy_data)
  {
    AssertDimension(row_vector.size(), col_dof_indices.size());

    /**
     * The array index used to access the @p std::forward_list
     * @p remaining_row_indices.
     */
    size_type next_ref_row_selection_index;
    /**
     * The reference row index to be selected for the next ACA step.
     */
    size_type next_ref_row_index(row_dof_indices.size());
    /**
     * The row DoF index corresponding to the selected reference row index for
     * the next ACA step.
     */
    types::global_dof_index next_ref_row_dof_index;

    while (!remaining_row_indices.empty())
      {
        next_ref_row_selection_index =
          generate_random_index(0, size(remaining_row_indices) - 1);
        next_ref_row_index =
          value_at(remaining_row_indices, next_ref_row_selection_index);
        next_ref_row_dof_index = row_dof_indices[next_ref_row_index];

        if (next_ref_row_index != current_ref_row_index)
          {
            /**
             * Get the @p next_ref_row_index'th row from the kernel evaluation.
             */
            assemble_kernel_row(row_vector,
                                kernel,
                                factor,
                                next_ref_row_dof_index,
                                col_dof_indices,
                                kx_dof_to_cell_topo,
                                ky_dof_to_cell_topo,
                                bem_values,
                                kx_dof_handler,
                                ky_dof_handler,
                                kx_dof_i2e_numbering,
                                ky_dof_i2e_numbering,
                                kx_mapping,
                                ky_mapping,
                                map_from_kx_boundary_mesh_to_volume_mesh,
                                map_from_ky_boundary_mesh_to_volume_mesh,
                                method_for_cell_neighboring_type,
                                scratch_data,
                                copy_data);

            if (LinAlg::is_all_zero(row_vector))
              {
                /**
                 * When the extracted row vector is zero-valued, remove the
                 * corresponding row index from the @p remaining_row_indices
                 * and continue another try.
                 */
                erase_at(remaining_row_indices, next_ref_row_selection_index);

                continue;
              }
            else
              {
                /**
                 * When the extracted row vector is not zero-valued, it is a
                 * valid choice. Hence, return the selected row index.
                 */
                return next_ref_row_index;
              }
          }
        else
          {
            if (size(remaining_row_indices) == 1)
              {
                /**
                 * If there is only one row index left, throw an exception here
                 * that there is no reference row can be selected.
                 */
                throw(ExcMessage(
                  "There are no remaining row indices to select from, since the current reference row is the only one left!"));
              }
            else
              {
                /**
                 * Because the selected row for the next step is the same as the
                 * current reference row, try another valid selection.
                 */
                continue;
              }
          }
      }

    throw(ExcMessage("There are no remaining row indices to select from!"));
    return next_ref_row_index;
  }


  /**
   * For debugging purpose: randomly select a reference row index from the
   * remaining row indices for the next ACA step.
   *
   * @param row_vector The extracted row vector, the memory of which should be
   * preallocated
   * @param A The full matrix for the current block, the rows of which are
   * extracted
   * @param remaining_row_indices
   * @param current_ref_row_index
   * @param row_dof_indices
   * @param col_dof_indices
   * @return
   */
  template <typename RangeNumberType = double>
  size_type
  random_select_ref_row(Vector<RangeNumberType> &                   row_vector,
                        const LAPACKFullMatrixExt<RangeNumberType> &A,
                        std::forward_list<size_type> &remaining_row_indices,
                        const size_type               current_ref_row_index)
  {
    /**
     * The array index used to access the @p std::forward_list
     * @p remaining_row_indices.
     */
    size_type next_ref_row_selection_index;
    /**
     * The reference row index to be selected for the next ACA step.
     */
    size_type next_ref_row_index;

    while (!remaining_row_indices.empty())
      {
        next_ref_row_selection_index =
          generate_random_index(0, size(remaining_row_indices) - 1);
        next_ref_row_index =
          value_at(remaining_row_indices, next_ref_row_selection_index);

        if (next_ref_row_index != current_ref_row_index)
          {
            A.get_row(next_ref_row_index, row_vector);

            if (LinAlg::is_all_zero(row_vector))
              {
                /**
                 * When the extracted row vector is zero-valued, remove the
                 * corresponding row index from the @p remaining_row_indices
                 * and continue another try.
                 */
                erase_at(remaining_row_indices, next_ref_row_selection_index);

                continue;
              }
            else
              {
                /**
                 * When the extracted row vector is not zero-valued, it is a
                 * valid choice. Hence, return the selected row index.
                 */
                return next_ref_row_index;
              }
          }
        else
          {
            if (size(remaining_row_indices) == 1)
              {
                /**
                 * If there is only one row index left, throw an exception here
                 * that there is no reference row can be selected.
                 */
                throw(ExcMessage(
                  "There are no remaining row indices to select from, since the current reference row is the only one left!"));
              }
            else
              {
                /**
                 * Because the selected row for the next step is the same as the
                 * current reference row, try another valid selection.
                 */
                continue;
              }
          }
      }

    throw(ExcMessage("There are no remaining row indices to select from!"));
    return next_ref_row_index;
  }


  /**
   * Randomly select a reference column index from the remaining column indices
   * for the next ACA step.
   *
   * \mynote{The data type for DoF index is @p types::global_dof_index, while
   * the data type for matrix row or column index is @p size_type, which is the
   * unsigned version of @p blas::int.}
   *
   * @param col_vector The selected reference column vector, the memory of which
   * should be preallocated in order to reduce the number of times for memory
   * allocation and releasing.
   * @param scratch
   * @param data
   * @param kernel
   * @param remaining_col_indices Remaining column indices to be checked and
   * selected from, which are stored in a @p std::forward_list.
   * @param current_ref_col_index The current reference column index
   * @param row_dof_indices The list of DoF indices corresponding to the matrix
   * rows
   * @param col_dof_indices The list of DoF indices corresponding to the
   * matrix columns
   * @param dof_to_cell_topo DoF-to-cell topology
   * @param bem_values
   * @param kx_dof_handler
   * @param ky_dof_handler
   * @param kx_mapping
   * @param ky_mapping
   * @return The selected reference column index
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  size_type
  random_select_ref_column(
    Vector<RangeNumberType> &                        col_vector,
    const KernelFunction<spacedim> &                 kernel,
    const RangeNumberType                            factor,
    std::forward_list<size_type> &                   remaining_col_indices,
    const size_type                                  current_ref_col_index,
    const std::vector<types::global_dof_index> &     row_dof_indices,
    const std::vector<types::global_dof_index> &     col_dof_indices,
    const std::vector<std::vector<unsigned int>> &   kx_dof_to_cell_topo,
    const std::vector<std::vector<unsigned int>> &   ky_dof_to_cell_topo,
    const BEMValues<dim, spacedim, RangeNumberType> &bem_values,
    const DoFHandler<dim, spacedim> &                kx_dof_handler,
    const DoFHandler<dim, spacedim> &                ky_dof_handler,
    const std::vector<types::global_dof_index> &     kx_dof_i2e_numbering,
    const std::vector<types::global_dof_index> &     ky_dof_i2e_numbering,
    const MappingQGenericExt<dim, spacedim> &        kx_mapping,
    const MappingQGenericExt<dim, spacedim> &        ky_mapping,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_kx_boundary_mesh_to_volume_mesh,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_ky_boundary_mesh_to_volume_mesh,
    const DetectCellNeighboringTypeMethod method_for_cell_neighboring_type,
    PairCellWiseScratchData<dim, spacedim, RangeNumberType> &scratch_data,
    PairCellWisePerTaskData<dim, spacedim, RangeNumberType> &copy_data)
  {
    AssertDimension(col_vector.size(), row_dof_indices.size());

    /**
     * The array index used to access the @p std::forward_list
     * @p remaining_col_indices.
     */
    size_type next_ref_col_selection_index;
    /**
     * The reference column index to be selected for the next ACA step.
     */
    size_type next_ref_col_index(col_dof_indices.size());
    /**
     * The column DoF index corresponding to the selected reference column index
     * for the next ACA step.
     */
    types::global_dof_index next_ref_col_dof_index;

    while (!remaining_col_indices.empty())
      {
        next_ref_col_selection_index =
          generate_random_index(0, size(remaining_col_indices) - 1);
        next_ref_col_index =
          value_at(remaining_col_indices, next_ref_col_selection_index);
        next_ref_col_dof_index = col_dof_indices[next_ref_col_index];

        if (next_ref_col_index != current_ref_col_index)
          {
            /**
             * Get the @p next_ref_col_index'th column from the kernel evaluation.
             */
            assemble_kernel_column(col_vector,
                                   kernel,
                                   factor,
                                   row_dof_indices,
                                   next_ref_col_dof_index,
                                   kx_dof_to_cell_topo,
                                   ky_dof_to_cell_topo,
                                   bem_values,
                                   kx_dof_handler,
                                   ky_dof_handler,
                                   kx_dof_i2e_numbering,
                                   ky_dof_i2e_numbering,
                                   kx_mapping,
                                   ky_mapping,
                                   map_from_kx_boundary_mesh_to_volume_mesh,
                                   map_from_ky_boundary_mesh_to_volume_mesh,
                                   method_for_cell_neighboring_type,
                                   scratch_data,
                                   copy_data);

            if (LinAlg::is_all_zero(col_vector))
              {
                /**
                 * When the extracted column vector is zero-valued, remove the
                 * corresponding column index from the @p remaining_col_indices
                 * and continue another try.
                 */
                erase_at(remaining_col_indices, next_ref_col_selection_index);

                continue;
              }
            else
              {
                /**
                 * When the extracted column vector is not zero-valued, it is a
                 * valid choice. Hence, return the selected column index.
                 */
                return next_ref_col_index;
              }
          }
        else
          {
            if (size(remaining_col_indices) == 1)
              {
                /**
                 * If there is only one column index left, throw an exception
                 * here that there is no reference column can be selected.
                 */
                throw(ExcMessage(
                  "There are no remaining column indices to select from, since the current reference column is the only one left!"));
              }
            else
              {
                /**
                 * Because the selected column for the next step is the same as
                 * the current reference column, try another valid selection.
                 */
                continue;
              }
          }
      }

    throw(ExcMessage("There are no remaining column indices to select from!"));
    return next_ref_col_index;
  }


  /**
   * For debugging purpose: randomly select a reference column index from the
   * remaining column indices for the next ACA step.
   *
   * @param col_vector The extracted column vector, the memory of which should be
   * preallocated
   * @param A The full matrix for the current block, the rows of which are
   * extracted
   * @param remaining_col_indices
   * @param current_ref_col_index
   * @param row_dof_indices
   * @param col_dof_indices
   * @return
   */
  template <typename RangeNumberType = double>
  size_type
  random_select_ref_column(Vector<RangeNumberType> &col_vector,
                           const LAPACKFullMatrixExt<RangeNumberType> &A,
                           std::forward_list<size_type> &remaining_col_indices,
                           const size_type               current_ref_col_index)
  {
    /**
     * The array index used to access the @p std::forward_list
     * @p remaining_col_indices.
     */
    size_type next_ref_col_selection_index;
    /**
     * The reference column index to be selected for the next ACA step.
     */
    size_type next_ref_col_index;

    while (!remaining_col_indices.empty())
      {
        next_ref_col_selection_index =
          generate_random_index(0, size(remaining_col_indices) - 1);
        next_ref_col_index =
          value_at(remaining_col_indices, next_ref_col_selection_index);

        if (next_ref_col_index != current_ref_col_index)
          {
            A.get_column(next_ref_col_index, col_vector);

            if (LinAlg::is_all_zero(col_vector))
              {
                /**
                 * When the extracted column vector is zero-valued, remove the
                 * corresponding column index from the @p remaining_col_indices
                 * and continue another try.
                 */
                erase_at(remaining_col_indices, next_ref_col_selection_index);
              }
            else
              {
                /**
                 * When the extracted column vector is not zero-valued, it is a
                 * valid choice. Hence, return the selected column index.
                 */
                return next_ref_col_index;
              }
          }
        else
          {
            if (size(remaining_col_indices) == 1)
              {
                /**
                 * If there is only one column index left, throw an exception
                 * here that there is no reference column can be selected.
                 */
                throw(ExcMessage(
                  "There are no remaining column indices to select from, since the current reference column is the only one left!"));
              }
            else
              {
                /**
                 * Because the selected column for the next step is the same as
                 * the current reference column, try another valid selection.
                 */
                continue;
              }
          }
      }

    throw(ExcMessage("There are no remaining column indices to select from!"));
    return next_ref_col_index;
  }



  /**
   * ACA+ algorithm
   *
   * \ref{Grasedyck, L. 2005. “Adaptive Recompression of
   * \f$\mathcal{H}\f$-Matrices for BEM.” Computing 74 (3): 205–23.}
   *
   * \mynote{At present, the simple convergence condition in this paper is
   * adopted instead of that in Bebendorf's book.}
   *
   * @param rkmat The rank-k matrix to be constructed for the current block,
   * the memory of which should be preallocated and the formal rank of which
   * should be the same as the maximum iteration number in @p aca_config.
   * @param aca_config
   * @param kernel
   * @param factor
   * @param row_dof_indices
   * @param col_dof_indices
   * @param kx_dof_to_cell_topo
   * @param ky_dof_to_cell_topo
   * @param bem_values
   * @param kx_dof_handler
   * @param ky_dof_handler
   * @param kx_mapping
   * @param ky_mapping
   * @param map_from_kx_boundary_mesh_to_volume_mesh
   * @param map_from_ky_boundary_mesh_to_volume_mesh
   * @param method_for_cell_neighboring_type
   * @param scratch_data
   * @param copy_data
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  aca_plus(
    RkMatrix<RangeNumberType> &                      rkmat,
    const ACAConfig &                                aca_config,
    const KernelFunction<spacedim> &                 kernel,
    const RangeNumberType                            factor,
    const std::array<types::global_dof_index, 2> &   row_dof_index_range,
    const std::array<types::global_dof_index, 2> &   col_dof_index_range,
    const std::vector<std::vector<unsigned int>> &   kx_dof_to_cell_topo,
    const std::vector<std::vector<unsigned int>> &   ky_dof_to_cell_topo,
    const BEMValues<dim, spacedim, RangeNumberType> &bem_values,
    const DoFHandler<dim, spacedim> &                kx_dof_handler,
    const DoFHandler<dim, spacedim> &                ky_dof_handler,
    const std::vector<types::global_dof_index> &     kx_dof_i2e_numbering,
    const std::vector<types::global_dof_index> &     ky_dof_i2e_numbering,
    const MappingQGenericExt<dim, spacedim> &        kx_mapping,
    const MappingQGenericExt<dim, spacedim> &        ky_mapping,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_kx_boundary_mesh_to_volume_mesh,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_ky_boundary_mesh_to_volume_mesh,
    const DetectCellNeighboringTypeMethod method_for_cell_neighboring_type,
    PairCellWiseScratchData<dim, spacedim, RangeNumberType> &scratch_data,
    PairCellWisePerTaskData<dim, spacedim, RangeNumberType> &copy_data)
  {
    /**
     * Get the size of each dimension of the matrix block to be built.
     */
    const size_type m = row_dof_index_range[1] - row_dof_index_range[0];
    const size_type n = col_dof_index_range[1] - col_dof_index_range[0];

    AssertDimension(rkmat.get_m(), m);
    AssertDimension(rkmat.get_n(), n);

    /**
     * Adjust the maximum ACA+ iteration if is larger than the matrix dimension.
     */
    unsigned int effective_max_iter = aca_config.max_iter;
    if (effective_max_iter > std::min(m, n))
      {
        effective_max_iter = std::min(m, n);
      }

    /**
     * Generate lists of internal DoF indices (internal DoF numbering) from
     * corresponding index ranges.
     */
    std::vector<types::global_dof_index> row_dof_indices(m);
    std::vector<types::global_dof_index> col_dof_indices(n);
    gen_linear_indices<vector_uta, types::global_dof_index>(
      row_dof_indices, row_dof_index_range[0]);
    gen_linear_indices<vector_uta, types::global_dof_index>(
      col_dof_indices, col_dof_index_range[0]);

    /**
     * Create matrix references associated with the component matrices in the
     * rank-k matrix to be returned, which hold the row and column vectors in
     * the selected crosses during the ACA iteration. The number of columns
     * should be the same as the maximum iteration number in @p aca_config.
     */
    AssertDimension(rkmat.get_formal_rank(), aca_config.max_iter);
    LAPACKFullMatrixExt<RangeNumberType> &u_mat = rkmat.get_A();
    LAPACKFullMatrixExt<RangeNumberType> &v_mat = rkmat.get_B();
    u_mat                                       = 0;
    v_mat                                       = 0;

    /**
     * Generate two lists for storing remaining row indices and column indices
     * of the matrix block.
     */
    std::forward_list<size_type> remaining_row_indices(m);
    std::forward_list<size_type> remaining_col_indices(n);

    gen_linear_indices<forward_list_uta, size_type>(remaining_row_indices);
    gen_linear_indices<forward_list_uta, size_type>(remaining_col_indices);

    /**
     * Select the initial reference row and column vectors and return their
     * matrix indices.
     */
    Vector<RangeNumberType> vr(n);
    Vector<RangeNumberType> uc(m);
    size_type               r =
      random_select_ref_row(vr,
                            kernel,
                            factor,
                            remaining_row_indices,
                            m + 1,
                            row_dof_indices,
                            col_dof_indices,
                            kx_dof_to_cell_topo,
                            ky_dof_to_cell_topo,
                            bem_values,
                            kx_dof_handler,
                            ky_dof_handler,
                            kx_dof_i2e_numbering,
                            ky_dof_i2e_numbering,
                            kx_mapping,
                            ky_mapping,
                            map_from_kx_boundary_mesh_to_volume_mesh,
                            map_from_ky_boundary_mesh_to_volume_mesh,
                            method_for_cell_neighboring_type,
                            scratch_data,
                            copy_data);
    size_type c =
      random_select_ref_column(uc,
                               kernel,
                               factor,
                               remaining_col_indices,
                               n + 1,
                               row_dof_indices,
                               col_dof_indices,
                               kx_dof_to_cell_topo,
                               ky_dof_to_cell_topo,
                               bem_values,
                               kx_dof_handler,
                               ky_dof_handler,
                               kx_dof_i2e_numbering,
                               ky_dof_i2e_numbering,
                               kx_mapping,
                               ky_mapping,
                               map_from_kx_boundary_mesh_to_volume_mesh,
                               map_from_ky_boundary_mesh_to_volume_mesh,
                               method_for_cell_neighboring_type,
                               scratch_data,
                               copy_data);

    /**
     * The absolute values for the reference row and column vectors.
     */
    Vector<RangeNumberType> vr_abs(n);
    Vector<RangeNumberType> uc_abs(m);

    /**
     * Calculate the absolute values for the reference row and column
     * vectors.
     *
     * \mynote{In the following, we use @p static_cast to explicitly
     * specify which version of @p std::fabs to call.}
     */
    std::transform(vr.begin(),
                   vr.end(),
                   vr_abs.begin(),
                   static_cast<RangeNumberType (*)(RangeNumberType)>(
                     std::fabs));
    std::transform(uc.begin(),
                   uc.end(),
                   uc_abs.begin(),
                   static_cast<RangeNumberType (*)(RangeNumberType)>(
                     std::fabs));

    /**
     * Row and column indices for the \f$k\f$'th step. The corresponding row
     * and column vectors comprise the cross.
     */
    size_type ik, jk;

    /**
     * Temporary row and column vectors for the current step \f$k\f$.
     */
    Vector<RangeNumberType> vk(n);
    Vector<RangeNumberType> uk(m);

    /**
     * The absolute values for the temporary row and column vectors for the
     * current step \f$k\f$.
     */
    Vector<RangeNumberType> vk_abs(n);
    Vector<RangeNumberType> uk_abs(m);

    /**
     * Temporary row and column vectors for the previous step \f$l\f$.
     */
    Vector<RangeNumberType> vl(n);
    Vector<RangeNumberType> ul(m);

    /**
     * The error threshold as the stopping condition.
     */
    RangeNumberType error_threshold = 0.0;

    /**
     * Start the ACA+ iteration from \f$k=1\f$.
     */
    for (unsigned int k = 1; k <= effective_max_iter; k++)
      {
        /**
         * Select the row index from the maximizer of the reference column.
         */
        // N.B. The function @p std::max_element return the iterator pointing
        // to the maximum element in the list.
        size_type i_star =
          std::distance(uc_abs.begin(),
                        std::max_element(uc_abs.begin(), uc_abs.end()));

        /**
         * Select the column index from the maximizer of the reference row.
         */
        size_type j_star =
          std::distance(vr_abs.begin(),
                        std::max_element(vr_abs.begin(), vr_abs.end()));

        if (uc_abs(i_star) > vr_abs(j_star))
          {
            /**
             * Select \f$i^*\f$ as the current row index \f$i_k\f$.
             */
            ik = i_star;

            /**
             * Extract the \f$i_k\f$'th row of \f$A\f$.
             */
            assemble_kernel_row(vk,
                                kernel,
                                factor,
                                row_dof_indices[ik],
                                col_dof_indices,
                                kx_dof_to_cell_topo,
                                ky_dof_to_cell_topo,
                                bem_values,
                                kx_dof_handler,
                                ky_dof_handler,
                                kx_dof_i2e_numbering,
                                ky_dof_i2e_numbering,
                                kx_mapping,
                                ky_mapping,
                                map_from_kx_boundary_mesh_to_volume_mesh,
                                map_from_ky_boundary_mesh_to_volume_mesh,
                                method_for_cell_neighboring_type,
                                scratch_data,
                                copy_data);

            /**
             * \mynote{Here the counter \f$l\f$ iterates over all the previous
             * steps before \f$k\f$. For each step, the row and column vectors
             * comprising the cross have been stored as column vectors into
             * @p v_mat and @p u_mat respectively. Hence, the index for
             * extracting vectors from @p v_mat or @p u_mat should be
             * \f$l-1\f$.}
             */
            for (size_type l = 1; l < k; l++)
              {
                /**
                 * Extract the previous cross.
                 */
                u_mat.get_column(l - 1, ul);
                v_mat.get_column(l - 1, vl);

                vk.add(-ul(ik), vl);
              }

            /**
             * Select the column index from the maximizer of the absolute values
             * of \f$\widetilde{v}_k\f$.
             */
            std::transform(vk.begin(),
                           vk.end(),
                           vk_abs.begin(),
                           static_cast<RangeNumberType (*)(RangeNumberType)>(
                             std::fabs));
            jk = std::distance(vk_abs.begin(),
                               std::max_element(vk_abs.begin(), vk_abs.end()));

            /**
             * Scale the vector \f$\widetilde{v}_k\f$.
             */
            RangeNumberType value_at_cross = vk(jk);
            vk /= value_at_cross;

            /**
             * Extract the \f$j_k\f$'th column from \f$A\f$.
             */
            assemble_kernel_column(uk,
                                   kernel,
                                   factor,
                                   row_dof_indices,
                                   col_dof_indices[jk],
                                   kx_dof_to_cell_topo,
                                   ky_dof_to_cell_topo,
                                   bem_values,
                                   kx_dof_handler,
                                   ky_dof_handler,
                                   kx_dof_i2e_numbering,
                                   ky_dof_i2e_numbering,
                                   kx_mapping,
                                   ky_mapping,
                                   map_from_kx_boundary_mesh_to_volume_mesh,
                                   map_from_ky_boundary_mesh_to_volume_mesh,
                                   method_for_cell_neighboring_type,
                                   scratch_data,
                                   copy_data);

            for (size_type l = 1; l < k; l++)
              {
                /**
                 * Extract the previous cross.
                 */
                u_mat.get_column(l - 1, ul);
                v_mat.get_column(l - 1, vl);

                uk.add(-vl(jk), ul);
              }
          }
        else
          {
            /**
             * Select \f$j^*\f$ as the current column index \f$j_k\f$.
             */
            jk = j_star;

            /**
             * Extract the \f$j_k\f$'th column from the matrix \f$A\f$.
             */
            assemble_kernel_column(uk,
                                   kernel,
                                   factor,
                                   row_dof_indices,
                                   col_dof_indices[jk],
                                   kx_dof_to_cell_topo,
                                   ky_dof_to_cell_topo,
                                   bem_values,
                                   kx_dof_handler,
                                   ky_dof_handler,
                                   kx_dof_i2e_numbering,
                                   ky_dof_i2e_numbering,
                                   kx_mapping,
                                   ky_mapping,
                                   map_from_kx_boundary_mesh_to_volume_mesh,
                                   map_from_ky_boundary_mesh_to_volume_mesh,
                                   method_for_cell_neighboring_type,
                                   scratch_data,
                                   copy_data);

            for (size_type l = 1; l < k; l++)
              {
                /**
                 * Extract the previous cross.
                 */
                u_mat.get_column(l - 1, ul);
                v_mat.get_column(l - 1, vl);

                uk.add(-vl(jk), ul);
              }

            /**
             * Select the row index as the maximizer of the absolute values of
             * \f$u_k\f$.
             */
            std::transform(uk.begin(),
                           uk.end(),
                           uk_abs.begin(),
                           static_cast<RangeNumberType (*)(RangeNumberType)>(
                             std::fabs));
            ik = std::distance(uk_abs.begin(),
                               std::max_element(uk_abs.begin(), uk_abs.end()));

            /**
             * Extract the \f$i_k\f$'th row from \f$A\f$.
             */
            assemble_kernel_row(vk,
                                kernel,
                                factor,
                                row_dof_indices[ik],
                                col_dof_indices,
                                kx_dof_to_cell_topo,
                                ky_dof_to_cell_topo,
                                bem_values,
                                kx_dof_handler,
                                ky_dof_handler,
                                kx_dof_i2e_numbering,
                                ky_dof_i2e_numbering,
                                kx_mapping,
                                ky_mapping,
                                map_from_kx_boundary_mesh_to_volume_mesh,
                                map_from_ky_boundary_mesh_to_volume_mesh,
                                method_for_cell_neighboring_type,
                                scratch_data,
                                copy_data);

            for (size_type l = 1; l < k; l++)
              {
                /**
                 * Extract the previous cross.
                 */
                u_mat.get_column(l - 1, ul);
                v_mat.get_column(l - 1, vl);

                vk.add(-ul(ik), vl);
              }

            /**
             * Scale the vector \f$\widetilde{v}_k\f$.
             */
            RangeNumberType value_at_cross = vk(jk);
            vk /= value_at_cross;
          }

        /**
         * Update the matrices storing row and column vectors comprising the
         * cross.
         */
        u_mat.fill_col(k - 1, uk, false);
        v_mat.fill_col(k - 1, vk, false);

        if (k == 1)
          {
            /**
             * Calculate the error threshold only in the first step.
             */
            error_threshold = aca_config.epsilon * rkmat.frobenius_norm(1);
          }

        remaining_row_indices.remove(ik);
        remaining_col_indices.remove(jk);

        if (ik == r)
          {
            /**
             * When the index of the selected row is the same as that of the
             * reference row, reselect a reference row.
             */
            r = random_select_ref_row(vr,
                                      kernel,
                                      factor,
                                      remaining_row_indices,
                                      r,
                                      row_dof_indices,
                                      col_dof_indices,
                                      kx_dof_to_cell_topo,
                                      ky_dof_to_cell_topo,
                                      bem_values,
                                      kx_dof_handler,
                                      ky_dof_handler,
                                      kx_dof_i2e_numbering,
                                      ky_dof_i2e_numbering,
                                      kx_mapping,
                                      ky_mapping,
                                      map_from_kx_boundary_mesh_to_volume_mesh,
                                      map_from_ky_boundary_mesh_to_volume_mesh,
                                      method_for_cell_neighboring_type,
                                      scratch_data,
                                      copy_data);

            for (unsigned int l = 1; l <= k; l++)
              {
                /**
                 * Extract the previous cross.
                 */
                u_mat.get_column(l - 1, ul);
                v_mat.get_column(l - 1, vl);

                vr.add(-ul(r), vl);
              }
          }
        else
          {
            /**
             * Otherwise, update the values of the reference row vector.
             */
            vr.add(-uk(r), vk);
          }

        /**
         * Update the vector of absolute values for @p vr.
         */
        std::transform(vr.begin(),
                       vr.end(),
                       vr_abs.begin(),
                       static_cast<RangeNumberType (*)(RangeNumberType)>(
                         std::fabs));

        if (jk == c)
          {
            /**
             * When the index of the selected column is the same as that of the
             * reference column, reselect a reference column.
             */
            c =
              random_select_ref_column(uc,
                                       kernel,
                                       factor,
                                       remaining_col_indices,
                                       c,
                                       row_dof_indices,
                                       col_dof_indices,
                                       kx_dof_to_cell_topo,
                                       ky_dof_to_cell_topo,
                                       bem_values,
                                       kx_dof_handler,
                                       ky_dof_handler,
                                       kx_dof_i2e_numbering,
                                       ky_dof_i2e_numbering,
                                       kx_mapping,
                                       ky_mapping,
                                       map_from_kx_boundary_mesh_to_volume_mesh,
                                       map_from_ky_boundary_mesh_to_volume_mesh,
                                       method_for_cell_neighboring_type,
                                       scratch_data,
                                       copy_data);

            for (unsigned int l = 1; l <= k; l++)
              {
                /**
                 * Extract the previous cross.
                 */
                u_mat.get_column(l - 1, ul);
                v_mat.get_column(l - 1, vl);

                uc.add(-vl(c), ul);
              }
          }
        else
          {
            /**
             * Otherwise, update the values of the reference column vector.
             */
            uc.add(-vk(c), uk);
          }

        /**
         * Update the vector of absolute values for @p uc.
         */
        std::transform(uc.begin(),
                       uc.end(),
                       uc_abs.begin(),
                       static_cast<RangeNumberType (*)(RangeNumberType)>(
                         std::fabs));

        /**
         * Check the convergence condition.
         */
        if ((k > 1) && (uk.l2_norm() * vk.l2_norm() <= error_threshold))
          {
            if (k < aca_config.max_iter)
              {
                /**
                 * If the number of ACA+ iterations is less than the allowed
                 * maximum value, the @p rkmat should be truncated to its
                 * actual rank.
                 */
                rkmat.truncate_to_rank(k);
              }

            return;
          }
        else
          {
            if (size(remaining_row_indices) == 0)
              {
                throw(ExcMessage(
                  "All rows of the block matrix have been tried for selection!"));

                break;
              }
          }
      }
  }


  /**
   * ACA+ algorithm for verification purpose. The matrix for the current block
   * is passed in as a full matrix without kernel function evaluation.
   *
   * \ref{Grasedyck, L. 2005. “Adaptive Recompression of
   * \f$\mathcal{H}\f$-Matrices for BEM.” Computing 74 (3): 205–23.}
   *
   * @param rkmat The rank-k matrix to be constructed for the current block,
   * the memory of which should be preallocated and the formal rank of which
   * should be the same as the maximum iteration number in @p aca_config.
   * @param aca_config
   * @param A The full matrix to be approximated
   */
  template <typename RangeNumberType = double>
  void
  aca_plus(RkMatrix<RangeNumberType> &                 rkmat,
           const ACAConfig &                           aca_config,
           const LAPACKFullMatrixExt<RangeNumberType> &A)
  {
    AssertDimension(rkmat.get_m(), A.m());
    AssertDimension(rkmat.get_n(), A.n());

    /**
     * Get the size of each dimension of the matrix block to be built.
     */
    const size_type m = A.m();
    const size_type n = A.n();

    /**
     * Create matrix references associated with the component matrices in the
     * rank-k matrix to be returned, which hold the row and column vectors in
     * the selected crosses during the ACA iteration. The number of columns
     * should be the same as the maximum iteration number in @p aca_config.
     */
    AssertDimension(rkmat.get_formal_rank(), aca_config.max_iter);
    LAPACKFullMatrixExt<RangeNumberType> &u_mat = rkmat.get_A();
    LAPACKFullMatrixExt<RangeNumberType> &v_mat = rkmat.get_B();
    u_mat                                       = 0.;
    v_mat                                       = 0.;

    /**
     * Generate two lists for storing remaining row indices and column indices
     * of the matrix block.
     */
    std::forward_list<size_type> remaining_row_indices(m);
    std::forward_list<size_type> remaining_col_indices(n);

    gen_linear_indices<forward_list_uta, size_type>(remaining_row_indices);
    gen_linear_indices<forward_list_uta, size_type>(remaining_col_indices);

    /**
     * Select the initial reference row and column vectors and return their
     * matrix indices.
     */
    Vector<RangeNumberType> vr(n);
    Vector<RangeNumberType> uc(m);
    size_type r = random_select_ref_row(vr, A, remaining_row_indices, m + 1);
    size_type c = random_select_ref_column(uc, A, remaining_col_indices, n + 1);

    /**
     * The absolute values for the reference row and column vectors.
     */
    Vector<RangeNumberType> vr_abs(n);
    Vector<RangeNumberType> uc_abs(m);

    /**
     * Calculate the absolute values for the reference row and column
     * vectors.
     *
     * \mynote{In the following, we use @p static_cast to explicitly
     * specify which version of @p std::fabs to call.}
     */
    std::transform(vr.begin(),
                   vr.end(),
                   vr_abs.begin(),
                   static_cast<RangeNumberType (*)(RangeNumberType)>(
                     std::fabs));
    std::transform(uc.begin(),
                   uc.end(),
                   uc_abs.begin(),
                   static_cast<RangeNumberType (*)(RangeNumberType)>(
                     std::fabs));

    /**
     * Row and column indices for the \f$k\f$'th step. The corresponding row
     * and column vectors comprise the cross.
     */
    size_type ik, jk;

    /**
     * Temporary row and column vectors for the current step \f$k\f$.
     */
    Vector<RangeNumberType> vk(n);
    Vector<RangeNumberType> uk(m);

    /**
     * The absolute values for the temporary row and column vectors for the
     * current step \f$k\f$.
     */
    Vector<RangeNumberType> vk_abs(n);
    Vector<RangeNumberType> uk_abs(m);

    /**
     * Temporary row and column vectors for the previous step \f$l\f$.
     */
    Vector<RangeNumberType> vl(n);
    Vector<RangeNumberType> ul(m);

    /**
     * Start the ACA+ iteration from \f$k=1\f$.
     */
    for (unsigned int k = 1; k <= aca_config.max_iter; k++)
      {
        /**
         * Select the row index from the maximizer of the reference column.
         */
        // N.B. The function @p std::max_element return the iterator pointing
        // to the maximum element in the list.
        size_type i_star =
          std::distance(uc_abs.begin(),
                        std::max_element(uc_abs.begin(), uc_abs.end()));

        /**
         * Select the column index from the maximizer of the reference row.
         */
        size_type j_star =
          std::distance(vr_abs.begin(),
                        std::max_element(vr_abs.begin(), vr_abs.end()));

        if (uc_abs(i_star) > vr_abs(j_star))
          {
            /**
             * Select \f$i^*\f$ as the current row index \f$i_k\f$.
             */
            ik = i_star;

            /**
             * Extract the \f$i_k\f$'th row of \f$A\f$.
             */
            A.get_row(ik, vk);

            /**
             * \mynote{Here the counter \f$l\f$ iterates over all the previous
             * steps before \f$k\f$. For each step, the row and column vectors
             * comprising the cross have been stored as column vectors into
             * @p v_mat and @p u_mat respectively. Hence, the index for
             * extracting vectors from @p v_mat or @p u_mat should be
             * \f$l-1\f$.}
             */
            for (size_type l = 1; l < k; l++)
              {
                /**
                 * Extract the previous cross.
                 */
                u_mat.get_column(l - 1, ul);
                v_mat.get_column(l - 1, vl);

                vk.add(-ul(ik), vl);
              }

            /**
             * Select the column index from the maximizer of the absolute values
             * of \f$\widetilde{v}_k\f$.
             */
            std::transform(vk.begin(),
                           vk.end(),
                           vk_abs.begin(),
                           static_cast<RangeNumberType (*)(RangeNumberType)>(
                             std::fabs));
            jk = std::distance(vk_abs.begin(),
                               std::max_element(vk_abs.begin(), vk_abs.end()));

            /**
             * Scale the vector \f$\widetilde{v}_k\f$.
             */
            RangeNumberType value_at_cross = vk(jk);
            vk /= value_at_cross;

            /**
             * Extract the \f$j_k\f$'th column from \f$A\f$.
             */
            A.get_column(jk, uk);

            for (size_type l = 1; l < k; l++)
              {
                /**
                 * Extract the previous cross.
                 */
                u_mat.get_column(l - 1, ul);
                v_mat.get_column(l - 1, vl);

                uk.add(-vl(jk), ul);
              }
          }
        else
          {
            /**
             * Select \f$j^*\f$ as the current column index \f$j_k\f$.
             */
            jk = j_star;

            /**
             * Extract the \f$j_k\f$'th column from the matrix \f$A\f$.
             */
            A.get_column(jk, uk);

            for (size_type l = 1; l < k; l++)
              {
                /**
                 * Extract the previous cross.
                 */
                u_mat.get_column(l - 1, ul);
                v_mat.get_column(l - 1, vl);

                uk.add(-vl(jk), ul);
              }

            /**
             * Select the row index as the maximizer of the absolute values of
             * \f$u_k\f$.
             */
            std::transform(uk.begin(),
                           uk.end(),
                           uk_abs.begin(),
                           static_cast<RangeNumberType (*)(RangeNumberType)>(
                             std::fabs));
            ik = std::distance(uk_abs.begin(),
                               std::max_element(uk_abs.begin(), uk_abs.end()));

            /**
             * Extract the \f$i_k\f$'th row from \f$A\f$.
             */
            A.get_row(ik, vk);

            for (size_type l = 1; l < k; l++)
              {
                /**
                 * Extract the previous cross.
                 */
                u_mat.get_column(l - 1, ul);
                v_mat.get_column(l - 1, vl);

                vk.add(-ul(ik), vl);
              }

            /**
             * Scale the vector \f$\widetilde{v}_k\f$.
             */
            RangeNumberType value_at_cross = vk(jk);
            vk /= value_at_cross;
          }

        /**
         * Update the matrices storing row and column vectors comprising the
         * cross.
         */
        u_mat.fill_col(k - 1, uk, false);
        v_mat.fill_col(k - 1, vk, false);

        remaining_row_indices.remove(ik);
        remaining_col_indices.remove(jk);

        if (ik == r)
          {
            /**
             * When the index of the selected row is the same as that of the
             * reference row, reselect a reference row.
             */
            r = random_select_ref_row(vr, A, remaining_row_indices, r);

            for (unsigned int l = 1; l <= k; l++)
              {
                /**
                 * Extract the previous cross.
                 */
                u_mat.get_column(l - 1, ul);
                v_mat.get_column(l - 1, vl);

                vr.add(-ul(r), vl);
              }
          }
        else
          {
            /**
             * Otherwise, update the values of the reference row vector.
             */
            vr.add(-uk(r), vk);
          }

        /**
         * Update the vector of absolute values for @p vr.
         */
        std::transform(vr.begin(),
                       vr.end(),
                       vr_abs.begin(),
                       static_cast<RangeNumberType (*)(RangeNumberType)>(
                         std::fabs));

        if (jk == c)
          {
            /**
             * When the index of the selected column is the same as that of the
             * reference column, reselect a reference column.
             */
            c = random_select_ref_column(uc, A, remaining_col_indices, c);

            for (unsigned int l = 1; l <= k; l++)
              {
                /**
                 * Extract the previous cross.
                 */
                u_mat.get_column(l - 1, ul);
                v_mat.get_column(l - 1, vl);

                uc.add(-vl(c), ul);
              }
          }
        else
          {
            /**
             * Otherwise, update the values of the reference column vector.
             */
            uc.add(-vk(c), uk);
          }

        /**
         * Update the vector of absolute values for @p uc.
         */
        std::transform(uc.begin(),
                       uc.end(),
                       uc_abs.begin(),
                       static_cast<RangeNumberType (*)(RangeNumberType)>(
                         std::fabs));

        /**
         * Check the convergence condition.
         */
        RangeNumberType error_threshold = 0.0;
        // DEBUG
        if (k > 1)
          {
            if (aca_config.eta >= 1.0)
              {
                error_threshold = aca_config.epsilon * rkmat.frobenius_norm(1);
              }
            else
              {
                error_threshold = aca_config.epsilon * (1.0 - aca_config.eta) /
                                  (1.0 + aca_config.epsilon) *
                                  rkmat.frobenius_norm(k - 1);
              }

            std::cerr << "uk.l2_norm() * vk.l2_norm(): "
                      << uk.l2_norm() * vk.l2_norm() << "\n"
                      << "error_threshold: " << error_threshold << std::endl;
          }
        else
          {
            std::cerr << "uk.l2_norm() * vk.l2_norm(): "
                      << uk.l2_norm() * vk.l2_norm() << std::endl;
          }

        if ((k > 1) && (uk.l2_norm() * vk.l2_norm() <= error_threshold))
          {
            if (k < aca_config.max_iter)
              {
                /**
                 * If the number of ACA+ iterations is less than the allowed
                 * maximum value, the @p rkmat should be truncated to its
                 * actual rank.
                 */
                rkmat.truncate_to_rank(k);
                // DEBUG
                std::cerr << "Early successful return from ACA+!" << std::endl;
              }

            return;
          }
        else
          {
            if (size(remaining_row_indices) == 0)
              {
                Assert(
                  false,
                  ExcMessage(
                    "All rows of the block matrix have been tried for selection!"));

                break;
              }
          }
      }

    /**
     * If the code runs here, all the @p max_iter number of columns in the
     * component matrices of @p rkmat have been filled, which means its formal
     * rank is equal to its real rank.
     */
    // DEBUG
    std::cerr << "All columns in the rank-k matrix have been filled!"
              << std::endl;
  }


  /**
   * Fill a single leaf node of the \hmatrix using ACA+. If the matrix type is
   * @p RkMatrixType, the memory for the full or rank-k matrix in the leaf node
   * has been allocated.
   *
   * For the near field matrix, full matrices will be built whose elements will
   * be obtained from the evaluation of the double integral in Galerkin-BEM. For
   * the far field admissible matrix, rank-k matrices will be built using ACA+.
   *
   * \mynote{This is used as the work function for parallel \hmatrix
   * construction using ACA+.}
   *
   * @param leaf_mat
   * @param aca_config
   * @param kernel
   * @param kernel_factor
   * @param kx_dof_to_cell_topo
   * @param ky_dof_to_cell_topo
   * @param bem_values
   * @param kx_dof_handler
   * @param ky_dof_handler
   * @param kx_mapping
   * @param ky_mapping
   * @param map_from_kx_boundary_mesh_to_volume_mesh
   * @param map_from_ky_boundary_mesh_to_volume_mesh
   * @param method_for_cell_neighboring_type
   * @param scratch_data
   * @param copy_data
   * @param enable_build_symmetric_hmat
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  fill_hmatrix_leaf_node_with_aca_plus(
    HMatrix<spacedim, RangeNumberType> *             leaf_mat,
    const ACAConfig &                                aca_config,
    const KernelFunction<spacedim> &                 kernel,
    const RangeNumberType                            kernel_factor,
    const std::vector<std::vector<unsigned int>> &   kx_dof_to_cell_topo,
    const std::vector<std::vector<unsigned int>> &   ky_dof_to_cell_topo,
    const BEMValues<dim, spacedim, RangeNumberType> &bem_values,
    const DoFHandler<dim, spacedim> &                kx_dof_handler,
    const DoFHandler<dim, spacedim> &                ky_dof_handler,
    const std::vector<types::global_dof_index> &     kx_dof_i2e_numbering,
    const std::vector<types::global_dof_index> &     ky_dof_i2e_numbering,
    const MappingQGenericExt<dim, spacedim> &        kx_mapping,
    const MappingQGenericExt<dim, spacedim> &        ky_mapping,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_kx_boundary_mesh_to_volume_mesh,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_ky_boundary_mesh_to_volume_mesh,
    const DetectCellNeighboringTypeMethod method_for_cell_neighboring_type,
    PairCellWiseScratchData<dim, spacedim, RangeNumberType> &scratch_data,
    PairCellWisePerTaskData<dim, spacedim, RangeNumberType> &copy_data,
    const bool enable_build_symmetric_hmat = false)
  {
    const std::array<types::global_dof_index, 2> *row_indices =
      leaf_mat->get_row_indices();
    const std::array<types::global_dof_index, 2> *col_indices =
      leaf_mat->get_col_indices();

    switch (leaf_mat->get_type())
      {
        case FullMatrixType:
          {
            LAPACKFullMatrixExt<RangeNumberType> *fullmat =
              leaf_mat->get_fullmatrix();

            if (enable_build_symmetric_hmat && kernel.is_symmetric())
              {
                /**
                 * When the flag @p enable_build_symmetric_hmat is true and the
                 * kernel function is symmetric, try to build a symmetric
                 * \hmatrix. Otherwise, the whole full matrix will be built.
                 */
                switch (leaf_mat->get_block_type())
                  {
                    case HMatrixSupport::diagonal_block:
                      {
                        /**
                         * A diagonal \hmatrix block as well as its associated
                         * full matrix should be symmetric.
                         */
                        Assert(
                          leaf_mat->get_property() == HMatrixSupport::symmetric,
                          ExcInvalidHMatrixProperty(leaf_mat->get_property()));
                        Assert(fullmat->get_property() ==
                                 LAPACKSupport::symmetric,
                               ExcInvalidLAPACKFullMatrixProperty(
                                 fullmat->get_property()));

                        /**
                         * Only evaluate the diagonal and lower triangular
                         * elements in the full matrix.
                         */
                        for (size_t i = 0; i < fullmat->m(); i++)
                          {
                            for (size_t j = 0; j <= i; j++)
                              {
                                (*fullmat)(i, j) =
                                  sauter_assemble_on_one_pair_of_dofs(
                                    kernel,
                                    kernel_factor,
                                    kx_dof_i2e_numbering[(*row_indices)[0] + i],
                                    ky_dof_i2e_numbering[(*col_indices)[0] + j],
                                    kx_dof_to_cell_topo,
                                    ky_dof_to_cell_topo,
                                    bem_values,
                                    kx_dof_handler,
                                    ky_dof_handler,
                                    kx_mapping,
                                    ky_mapping,
                                    map_from_kx_boundary_mesh_to_volume_mesh,
                                    map_from_ky_boundary_mesh_to_volume_mesh,
                                    method_for_cell_neighboring_type,
                                    scratch_data,
                                    copy_data);
                              }
                          }

                        break;
                      }
                    case HMatrixSupport::upper_triangular_block:
                      {
                        /**
                         * Do not build \hmatrix block belonging to the upper
                         * triangular part.
                         */

                        break;
                      }
                    case HMatrixSupport::lower_triangular_block:
                      {
                        /**
                         * When the current \hmatrix block belongs to the lower
                         * triangular part, evaluate all of its elements as
                         * usual.
                         */
                        for (size_t i = 0; i < fullmat->m(); i++)
                          {
                            for (size_t j = 0; j < fullmat->n(); j++)
                              {
                                (*fullmat)(i, j) =
                                  sauter_assemble_on_one_pair_of_dofs(
                                    kernel,
                                    kernel_factor,
                                    kx_dof_i2e_numbering[(*row_indices)[0] + i],
                                    ky_dof_i2e_numbering[(*col_indices)[0] + j],
                                    kx_dof_to_cell_topo,
                                    ky_dof_to_cell_topo,
                                    bem_values,
                                    kx_dof_handler,
                                    ky_dof_handler,
                                    kx_mapping,
                                    ky_mapping,
                                    map_from_kx_boundary_mesh_to_volume_mesh,
                                    map_from_ky_boundary_mesh_to_volume_mesh,
                                    method_for_cell_neighboring_type,
                                    scratch_data,
                                    copy_data);
                              }
                          }

                        break;
                      }
                    case HMatrixSupport::undefined_block:
                      {
                        Assert(false,
                               ExcInvalidHMatrixBlockType(
                                 leaf_mat->get_block_type()));

                        break;
                      }
                  }
              }
            else
              {
                /**
                 * Evaluate the whole full matrix.
                 */
                for (size_t i = 0; i < fullmat->m(); i++)
                  {
                    for (size_t j = 0; j < fullmat->n(); j++)
                      {
                        (*fullmat)(i, j) = sauter_assemble_on_one_pair_of_dofs(
                          kernel,
                          kernel_factor,
                          kx_dof_i2e_numbering[(*row_indices)[0] + i],
                          ky_dof_i2e_numbering[(*col_indices)[0] + j],
                          kx_dof_to_cell_topo,
                          ky_dof_to_cell_topo,
                          bem_values,
                          kx_dof_handler,
                          ky_dof_handler,
                          kx_mapping,
                          ky_mapping,
                          map_from_kx_boundary_mesh_to_volume_mesh,
                          map_from_ky_boundary_mesh_to_volume_mesh,
                          method_for_cell_neighboring_type,
                          scratch_data,
                          copy_data);
                      }
                  }
              }

            break;
          }
        case RkMatrixType:
          {
            /**
             * When the \hmatrix block type is rank-k matrix, when the top
             * level \hmatrix is symmetric and the flag
             * @p enable_build_symmetric_hmat is true, only those matrix blocks
             * belong to the lower triangular part will be computed. Otherwise,
             * the rank-k matrix block will always be computed. ACA+ will be
             * used for building the rank-k matrix.
             */
            RkMatrix<RangeNumberType> *rkmat = leaf_mat->get_rkmatrix();

            if (enable_build_symmetric_hmat && kernel.is_symmetric())
              {
                switch (leaf_mat->get_block_type())
                  {
                    case HMatrixSupport::lower_triangular_block:
                      {
                        /**
                         * Build the \hmatrix block when it belongs to the lower
                         * triangular part using ACA+.
                         */
                        aca_plus((*rkmat),
                                 aca_config,
                                 kernel,
                                 kernel_factor,
                                 *row_indices,
                                 *col_indices,
                                 kx_dof_to_cell_topo,
                                 ky_dof_to_cell_topo,
                                 bem_values,
                                 kx_dof_handler,
                                 ky_dof_handler,
                                 kx_dof_i2e_numbering,
                                 ky_dof_i2e_numbering,
                                 kx_mapping,
                                 ky_mapping,
                                 map_from_kx_boundary_mesh_to_volume_mesh,
                                 map_from_ky_boundary_mesh_to_volume_mesh,
                                 method_for_cell_neighboring_type,
                                 scratch_data,
                                 copy_data);

                        break;
                      }
                    case HMatrixSupport::upper_triangular_block:
                      {
                        /**
                         * Do not build \hmatrix block belonging to the upper
                         * triangular part.
                         */

                        break;
                      }
                    case HMatrixSupport::diagonal_block:
                      /**
                       * An rank-k matrix cannot belong to the diagonal part.
                       */
                    case HMatrixSupport::undefined_block:
                      {
                        Assert(false,
                               ExcInvalidHMatrixBlockType(
                                 leaf_mat->get_block_type()));

                        break;
                      }
                  }
              }
            else
              {
                aca_plus((*rkmat),
                         aca_config,
                         kernel,
                         kernel_factor,
                         (*row_indices),
                         (*col_indices),
                         kx_dof_to_cell_topo,
                         ky_dof_to_cell_topo,
                         bem_values,
                         kx_dof_handler,
                         ky_dof_handler,
                         kx_dof_i2e_numbering,
                         ky_dof_i2e_numbering,
                         kx_mapping,
                         ky_mapping,
                         map_from_kx_boundary_mesh_to_volume_mesh,
                         map_from_ky_boundary_mesh_to_volume_mesh,
                         method_for_cell_neighboring_type,
                         scratch_data,
                         copy_data);
              }

            break;
          }
        default:
          {
            Assert(false, ExcInvalidHMatrixType(leaf_mat->get_type()));
          }
      }
  }


  /**
   * Fill a single leaf node of the \hmatrix using ACA+.
   *
   * In the mean time, the FEM mass matrix multiplied by a factor will be added
   * to the near field matrix block.
   *
   * If the matrix type is @p RkMatrixType, the memory for the full or rank-k
   * matrix in the leaf node has been allocated.
   *
   * For the near field matrix, full matrices will be built whose elements will
   * be obtained from the evaluation of the double integral in Galerkin-BEM. For
   * the far field admissible matrix, rank-k matrices will be built using ACA+.
   *
   * \mynote{This is used as the work function for parallel \hmatrix
   * construction using ACA+.}
   *
   * @param leaf_mat
   * @param aca_config
   * @param kernel
   * @param kernel_factor
   * @param mass_matrix_factor
   * @param kx_dof_to_cell_topo
   * @param ky_dof_to_cell_topo
   * @param bem_values
   * @param mass_matrix_quadrature_formula
   * @param kx_dof_handler
   * @param ky_dof_handler
   * @param kx_mapping
   * @param ky_mapping
   * @param map_from_kx_boundary_mesh_to_volume_mesh
   * @param map_from_ky_boundary_mesh_to_volume_mesh
   * @param method_for_cell_neighboring_type
   * @param mass_matrix_scratch_data
   * @param scratch_data
   * @param copy_data
   * @param enable_build_symmetric_hmat
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  fill_hmatrix_leaf_node_with_aca_plus(
    HMatrix<spacedim, RangeNumberType> *             leaf_mat,
    const ACAConfig &                                aca_config,
    const KernelFunction<spacedim> &                 kernel,
    const RangeNumberType                            kernel_factor,
    const RangeNumberType                            mass_matrix_factor,
    const std::vector<std::vector<unsigned int>> &   kx_dof_to_cell_topo,
    const std::vector<std::vector<unsigned int>> &   ky_dof_to_cell_topo,
    const BEMValues<dim, spacedim, RangeNumberType> &bem_values,
    const DoFHandler<dim, spacedim> &                kx_dof_handler,
    const DoFHandler<dim, spacedim> &                ky_dof_handler,
    const std::vector<types::global_dof_index> &     kx_dof_i2e_numbering,
    const std::vector<types::global_dof_index> &     ky_dof_i2e_numbering,
    const MappingQGenericExt<dim, spacedim> &        kx_mapping,
    const MappingQGenericExt<dim, spacedim> &        ky_mapping,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_kx_boundary_mesh_to_volume_mesh,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_ky_boundary_mesh_to_volume_mesh,
    const DetectCellNeighboringTypeMethod method_for_cell_neighboring_type,
    CellWiseScratchDataForMassMatrix<dim, spacedim> &mass_matrix_scratch_data,
    PairCellWiseScratchData<dim, spacedim, RangeNumberType> &scratch_data,
    PairCellWisePerTaskData<dim, spacedim, RangeNumberType> &copy_data,
    const bool enable_build_symmetric_hmat = false)
  {
    const std::array<types::global_dof_index, 2> *row_indices =
      leaf_mat->get_row_indices();
    const std::array<types::global_dof_index, 2> *col_indices =
      leaf_mat->get_col_indices();

    switch (leaf_mat->get_type())
      {
        case FullMatrixType:
          {
            LAPACKFullMatrixExt<RangeNumberType> *fullmat =
              leaf_mat->get_fullmatrix();

            if (enable_build_symmetric_hmat && kernel.is_symmetric())
              {
                /**
                 * When the flag @p enable_build_symmetric_hmat is true and the
                 * kernel function is symmetric, try to build a symmetric
                 * \hmatrix. Otherwise, the whole full matrix will be built.
                 */
                switch (leaf_mat->get_block_type())
                  {
                    case HMatrixSupport::diagonal_block:
                      {
                        /**
                         * A diagonal \hmatrix block as well as its associated
                         * full matrix should be symmetric.
                         */
                        Assert(
                          leaf_mat->get_property() == HMatrixSupport::symmetric,
                          ExcInvalidHMatrixProperty(leaf_mat->get_property()));
                        Assert(fullmat->get_property() ==
                                 LAPACKSupport::symmetric,
                               ExcInvalidLAPACKFullMatrixProperty(
                                 fullmat->get_property()));

                        /**
                         * Only evaluate the diagonal and lower triangular
                         * elements in the full matrix.
                         */
                        for (size_t i = 0; i < fullmat->m(); i++)
                          {
                            for (size_t j = 0; j <= i; j++)
                              {
                                (*fullmat)(i, j) =
                                  sauter_assemble_on_one_pair_of_dofs(
                                    kernel,
                                    kernel_factor,
                                    mass_matrix_factor,
                                    kx_dof_i2e_numbering[(*row_indices)[0] + i],
                                    ky_dof_i2e_numbering[(*col_indices)[0] + j],
                                    kx_dof_to_cell_topo,
                                    ky_dof_to_cell_topo,
                                    bem_values,
                                    kx_dof_handler,
                                    ky_dof_handler,
                                    kx_mapping,
                                    ky_mapping,
                                    map_from_kx_boundary_mesh_to_volume_mesh,
                                    map_from_ky_boundary_mesh_to_volume_mesh,
                                    method_for_cell_neighboring_type,
                                    mass_matrix_scratch_data,
                                    scratch_data,
                                    copy_data);
                              }
                          }

                        break;
                      }
                    case HMatrixSupport::upper_triangular_block:
                      {
                        /**
                         * Do not build \hmatrix block belonging to the upper
                         * triangular part.
                         */

                        break;
                      }
                    case HMatrixSupport::lower_triangular_block:
                      {
                        /**
                         * When the current \hmatrix block belongs to the lower
                         * triangular part, evaluate all of its elements as
                         * usual.
                         */
                        for (size_t i = 0; i < fullmat->m(); i++)
                          {
                            for (size_t j = 0; j < fullmat->n(); j++)
                              {
                                (*fullmat)(i, j) =
                                  sauter_assemble_on_one_pair_of_dofs(
                                    kernel,
                                    kernel_factor,
                                    mass_matrix_factor,
                                    kx_dof_i2e_numbering[(*row_indices)[0] + i],
                                    ky_dof_i2e_numbering[(*col_indices)[0] + j],
                                    kx_dof_to_cell_topo,
                                    ky_dof_to_cell_topo,
                                    bem_values,
                                    kx_dof_handler,
                                    ky_dof_handler,
                                    kx_mapping,
                                    ky_mapping,
                                    map_from_kx_boundary_mesh_to_volume_mesh,
                                    map_from_ky_boundary_mesh_to_volume_mesh,
                                    method_for_cell_neighboring_type,
                                    mass_matrix_scratch_data,
                                    scratch_data,
                                    copy_data);
                              }
                          }

                        break;
                      }
                    case HMatrixSupport::undefined_block:
                      {
                        Assert(false,
                               ExcInvalidHMatrixBlockType(
                                 leaf_mat->get_block_type()));

                        break;
                      }
                  }
              }
            else
              {
                for (size_t i = 0; i < fullmat->m(); i++)
                  {
                    for (size_t j = 0; j < fullmat->n(); j++)
                      {
                        (*fullmat)(i, j) = sauter_assemble_on_one_pair_of_dofs(
                          kernel,
                          kernel_factor,
                          mass_matrix_factor,
                          kx_dof_i2e_numbering[(*row_indices)[0] + i],
                          ky_dof_i2e_numbering[(*col_indices)[0] + j],
                          kx_dof_to_cell_topo,
                          ky_dof_to_cell_topo,
                          bem_values,
                          kx_dof_handler,
                          ky_dof_handler,
                          kx_mapping,
                          ky_mapping,
                          map_from_kx_boundary_mesh_to_volume_mesh,
                          map_from_ky_boundary_mesh_to_volume_mesh,
                          method_for_cell_neighboring_type,
                          mass_matrix_scratch_data,
                          scratch_data,
                          copy_data);
                      }
                  }
              }

            break;
          }
        case RkMatrixType:
          {
            RkMatrix<RangeNumberType> *rkmat = leaf_mat->get_rkmatrix();

            if (enable_build_symmetric_hmat && kernel.is_symmetric())
              {
                switch (leaf_mat->get_block_type())
                  {
                    case HMatrixSupport::lower_triangular_block:
                      {
                        /**
                         * Build the \hmatrix block when it belongs to the lower
                         * triangular part using ACA+.
                         */
                        aca_plus((*rkmat),
                                 aca_config,
                                 kernel,
                                 kernel_factor,
                                 *row_indices,
                                 *col_indices,
                                 kx_dof_to_cell_topo,
                                 ky_dof_to_cell_topo,
                                 bem_values,
                                 kx_dof_handler,
                                 ky_dof_handler,
                                 kx_dof_i2e_numbering,
                                 ky_dof_i2e_numbering,
                                 kx_mapping,
                                 ky_mapping,
                                 map_from_kx_boundary_mesh_to_volume_mesh,
                                 map_from_ky_boundary_mesh_to_volume_mesh,
                                 method_for_cell_neighboring_type,
                                 scratch_data,
                                 copy_data);

                        break;
                      }
                    case HMatrixSupport::upper_triangular_block:
                      {
                        /**
                         * Do not build \hmatrix block belonging to the upper
                         * triangular part.
                         */

                        break;
                      }
                    case HMatrixSupport::diagonal_block:
                      /**
                       * An rank-k matrix cannot belong to the diagonal part.
                       */
                    case HMatrixSupport::undefined_block:
                      {
                        Assert(false,
                               ExcInvalidHMatrixBlockType(
                                 leaf_mat->get_block_type()));

                        break;
                      }
                  }
              }
            else
              {
                aca_plus((*rkmat),
                         aca_config,
                         kernel,
                         kernel_factor,
                         *row_indices,
                         *col_indices,
                         kx_dof_to_cell_topo,
                         ky_dof_to_cell_topo,
                         bem_values,
                         kx_dof_handler,
                         ky_dof_handler,
                         kx_dof_i2e_numbering,
                         ky_dof_i2e_numbering,
                         kx_mapping,
                         ky_mapping,
                         map_from_kx_boundary_mesh_to_volume_mesh,
                         map_from_ky_boundary_mesh_to_volume_mesh,
                         method_for_cell_neighboring_type,
                         scratch_data,
                         copy_data);
              }

            break;
          }
        default:
          {
            Assert(false, ExcInvalidHMatrixType(leaf_mat->get_type()));
          }
      }
  }


  /**
   * Fill a vector of leaf \hmatrices corresponding to the given vector of
   * kernel functions using ACA+.
   *
   * If the matrix type is @p RkMatrixType, the memory for the full or rank-k
   * matrix in the leaf node has been allocated. This version is applied to a
   * list of kernels.
   *
   * For the near field matrix, full matrices will be built whose elements will
   * be obtained from the evaluation of the double integral in Galerkin-BEM. For
   * the far field admissible matrix, rank-k matrices will be built using ACA+.
   *
   * \mynote{This is used as the working function for parallel \hmatrix
   * construction using ACA+.
   *
   * The list of bilinear forms related to the kernels have the same test and
   * trial spaces. Therefore, the list \hmatrices have the same \bct structure.}
   *
   * @param leaf_mat_for_kernels A vector of leaf \hmatrix pointers
   * @param aca_config
   * @param kernels A vector of kernel function pointers
   * @param kernel_factors
   * @param kx_dof_to_cell_topo
   * @param ky_dof_to_cell_topo
   * @param bem_values
   * @param kx_dof_handler
   * @param ky_dof_handler
   * @param kx_mapping
   * @param ky_mapping
   * @param map_from_kx_boundary_mesh_to_volume_mesh
   * @param map_from_ky_boundary_mesh_to_volume_mesh
   * @param method_for_cell_neighboring_type
   * @param scratch_data
   * @param copy_data
   * @param enable_build_symmetric_hmat Flag indicating whether symmetric
   * \hmatrix will be built when the kernel function is symmetric.
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  fill_hmatrix_leaf_node_with_aca_plus(
    const std::vector<HMatrix<spacedim, RangeNumberType> *>
      &                                              leaf_mat_for_kernels,
    const ACAConfig &                                aca_config,
    const std::vector<KernelFunction<spacedim> *> &  kernels,
    const std::vector<RangeNumberType> &             kernel_factors,
    const std::vector<std::vector<unsigned int>> &   kx_dof_to_cell_topo,
    const std::vector<std::vector<unsigned int>> &   ky_dof_to_cell_topo,
    const BEMValues<dim, spacedim, RangeNumberType> &bem_values,
    const DoFHandler<dim, spacedim> &                kx_dof_handler,
    const DoFHandler<dim, spacedim> &                ky_dof_handler,
    const std::vector<types::global_dof_index> &     kx_dof_i2e_numbering,
    const std::vector<types::global_dof_index> &     ky_dof_i2e_numbering,
    const MappingQGenericExt<dim, spacedim> &        kx_mapping,
    const MappingQGenericExt<dim, spacedim> &        ky_mapping,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_kx_boundary_mesh_to_volume_mesh,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_ky_boundary_mesh_to_volume_mesh,
    const DetectCellNeighboringTypeMethod method_for_cell_neighboring_type,
    PairCellWiseScratchData<dim, spacedim, RangeNumberType> &scratch_data,
    PairCellWisePerTaskData<dim, spacedim, RangeNumberType> &copy_data,
    const bool enable_build_symmetric_hmat = false)
  {
    AssertDimension(leaf_mat_for_kernels.size(), kernels.size());

    const unsigned int kernel_num = kernels.size();

    const std::array<types::global_dof_index, 2> *row_indices =
      leaf_mat_for_kernels[0]->get_row_indices();
    const std::array<types::global_dof_index, 2> *col_indices =
      leaf_mat_for_kernels[0]->get_col_indices();

    /**
     * Flags indicating whether the kernels are to be evaluated at the pair of
     * DoFs.
     */
    std::vector<bool> enable_kernel_evaluations(kernel_num);

    switch (leaf_mat_for_kernels[0]->get_type())
      {
        case FullMatrixType:
          {
            Vector<RangeNumberType> fullmat_coeffs(kernel_num);

            /**
             * Iterate over each DoF index \f$i\f$ in the cluster \$\tau\f$.
             */
            for (size_t i = 0; i < ((*row_indices)[1] - (*row_indices)[0]); i++)
              {
                /**
                 * Iterate over each DoF index \f$j\f$ in the cluster
                 * \$\sigma\f$.
                 */
                for (size_t j = 0; j < ((*col_indices)[1] - (*col_indices)[0]);
                     j++)
                  {
                    /**
                     * Determine if each kernel is to be evaluated at the
                     * current pair of DoFs.
                     */
                    for (unsigned int k = 0; k < kernel_num; k++)
                      {
                        if (enable_build_symmetric_hmat &&
                            kernels[k]->is_symmetric())
                          {
                            /**
                             * When the flag @p enable_build_symmetric_hmat is true and the
                             * kernel function is symmetric, try to build a
                             * symmetric \hmatrix. Otherwise, the whole full
                             * matrix will be always built.
                             */
                            switch (leaf_mat_for_kernels[k]->get_block_type())
                              {
                                case HMatrixSupport::diagonal_block:
                                  {
                                    /**
                                     * A diagonal \hmatrix block as well as its
                                     * associated full matrix should be
                                     * symmetric.
                                     */
                                    Assert(
                                      leaf_mat_for_kernels[k]->get_property() ==
                                        HMatrixSupport::symmetric,
                                      ExcInvalidHMatrixProperty(
                                        leaf_mat_for_kernels[k]
                                          ->get_property()));
                                    Assert(leaf_mat_for_kernels[k]
                                               ->get_fullmatrix()
                                               ->get_property() ==
                                             LAPACKSupport::symmetric,
                                           ExcInvalidLAPACKFullMatrixProperty(
                                             leaf_mat_for_kernels[k]
                                               ->get_fullmatrix()
                                               ->get_property()));

                                    if (j <= i)
                                      {
                                        /**
                                         * Only the diagonal and lower
                                         * triangular elements in the full
                                         * matrix will be evaluated.
                                         */
                                        enable_kernel_evaluations[k] = true;
                                      }
                                    else
                                      {
                                        /**
                                         * Upper triangular elements in the full
                                         * matrix are ignored.
                                         */
                                        enable_kernel_evaluations[k] = false;
                                      }

                                    break;
                                  }
                                case HMatrixSupport::upper_triangular_block:
                                  {
                                    /**
                                     * Do not build \hmatrix block belonging to
                                     * the upper triangular part.
                                     */
                                    enable_kernel_evaluations[k] = false;

                                    break;
                                  }
                                case HMatrixSupport::lower_triangular_block:
                                  {
                                    /**
                                     * When the current \hmatrix block belongs
                                     * to the lower triangular part, evaluate
                                     * all of its elements as usual.
                                     */
                                    enable_kernel_evaluations[k] = true;

                                    break;
                                  }
                                case HMatrixSupport::undefined_block:
                                  {
                                    Assert(false,
                                           ExcInvalidHMatrixBlockType(
                                             leaf_mat_for_kernels[k]
                                               ->get_block_type()));
                                    enable_kernel_evaluations[k] = true;

                                    break;
                                  }
                              }
                          }
                        else
                          {
                            enable_kernel_evaluations[k] = true;
                          }
                      }

                    /**
                     * Perform Sauter quadrature on the pair of DoF indices
                     * \f$(i,j)\f$ for the vector kernel functions. The list of
                     * results are collected into the vector @p fullmat_coeffs.
                     */
                    sauter_assemble_on_one_pair_of_dofs(
                      kernels,
                      kernel_factors,
                      enable_kernel_evaluations,
                      fullmat_coeffs,
                      kx_dof_i2e_numbering[(*row_indices)[0] + i],
                      ky_dof_i2e_numbering[(*col_indices)[0] + j],
                      kx_dof_to_cell_topo,
                      ky_dof_to_cell_topo,
                      bem_values,
                      kx_dof_handler,
                      ky_dof_handler,
                      kx_mapping,
                      ky_mapping,
                      map_from_kx_boundary_mesh_to_volume_mesh,
                      map_from_ky_boundary_mesh_to_volume_mesh,
                      method_for_cell_neighboring_type,
                      scratch_data,
                      copy_data);

                    /**
                     * Assign the vector of returned values to each full matrix
                     * corresponding to the kernel function.
                     */
                    for (unsigned int k = 0; k < kernel_num; k++)
                      {
                        if (enable_kernel_evaluations[k])
                          {
                            LAPACKFullMatrixExt<RangeNumberType> *fullmat =
                              leaf_mat_for_kernels[k]->get_fullmatrix();
                            (*fullmat)(i, j) = fullmat_coeffs(k);
                          }
                      }
                  }
              }

            break;
          }
        case RkMatrixType:
          {
            /**
             * Determine if each kernel is to be evaluated at the current pair
             * of DoFs.
             */
            for (unsigned int k = 0; k < kernel_num; k++)
              {
                if (enable_build_symmetric_hmat && kernels[k]->is_symmetric())
                  {
                    switch (leaf_mat_for_kernels[k]->get_block_type())
                      {
                        case HMatrixSupport::lower_triangular_block:
                          {
                            /**
                             * Build the \hmatrix block when it belongs to the
                             * lower triangular part using ACA+.
                             */
                            enable_kernel_evaluations[k] = true;

                            break;
                          }
                        case HMatrixSupport::upper_triangular_block:
                          {
                            /**
                             * Do not build \hmatrix block belonging to the
                             * upper triangular part.
                             */
                            enable_kernel_evaluations[k] = false;

                            break;
                          }
                        case HMatrixSupport::diagonal_block:
                          /**
                           * An rank-k matrix cannot belong to the diagonal
                           * part.
                           */
                        case HMatrixSupport::undefined_block:
                          {
                            Assert(
                              false,
                              ExcInvalidHMatrixBlockType(
                                leaf_mat_for_kernels[k]->get_block_type()));
                            enable_kernel_evaluations[k] = true;

                            break;
                          }
                      }
                  }
                else
                  {
                    enable_kernel_evaluations[k] = true;
                  }
              }

            /**
             * Iterate over each kernel and build the far field matrix block in
             * the rank-k format using ACA+.
             */
            unsigned int counter = 0;
            for (const KernelFunction<spacedim> *kernel : kernels)
              {
                if (enable_kernel_evaluations[counter])
                  {
                    RkMatrix<RangeNumberType> *rkmat =
                      leaf_mat_for_kernels[counter]->get_rkmatrix();

                    aca_plus((*rkmat),
                             aca_config,
                             *kernel,
                             kernel_factors[counter],
                             *row_indices,
                             *col_indices,
                             kx_dof_to_cell_topo,
                             ky_dof_to_cell_topo,
                             bem_values,
                             kx_dof_handler,
                             ky_dof_handler,
                             kx_dof_i2e_numbering,
                             ky_dof_i2e_numbering,
                             kx_mapping,
                             ky_mapping,
                             map_from_kx_boundary_mesh_to_volume_mesh,
                             map_from_ky_boundary_mesh_to_volume_mesh,
                             method_for_cell_neighboring_type,
                             scratch_data,
                             copy_data);
                  }

                counter++;
              }

            break;
          }
        default:
          {
            Assert(false,
                   ExcInvalidHMatrixType(leaf_mat_for_kernels[0]->get_type()));
          }
      }
  }


  /**
   * Fill a vector of leaf \hmatrices corresponding to the given vector of
   * kernel functions using ACA+.
   *
   * In the meantime, the FEM mass matrix multiplied by a factor will be added
   * to the near field matrix block.
   *
   * If the matrix type is @p RkMatrixType, the memory for the full or rank-k
   * matrix in the leaf node has been allocated. This version is applied to a
   * list of kernels.
   *
   * For the near field matrix, full matrices will be built whose elements will
   * be obtained from the evaluation of the double integral in Galerkin-BEM. For
   * the far field admissible matrix, rank-k matrices will be built using ACA+.
   *
   * \mynote{This is used as the work function for parallel \hmatrix
   * construction using ACA+.}
   *
   * @param leaf_mat_for_kernels
   * @param aca_config
   * @param kernels
   * @param kernel_factors
   * @param mass_matrix_factors
   * @param kx_dof_to_cell_topo
   * @param ky_dof_to_cell_topo
   * @param bem_values
   * @param kx_dof_handler
   * @param ky_dof_handler
   * @param kx_mapping
   * @param ky_mapping
   * @param map_from_kx_boundary_mesh_to_volume_mesh
   * @param map_from_ky_boundary_mesh_to_volume_mesh
   * @param method_for_cell_neighboring_type
   * @param mass_matrix__scratch_data
   * @param scratch_data
   * @param copy_data
   * @param enable_build_symmetric_hmat Flag indicating whether symmetric
   * \hmatrix will be built when the kernel function is symmetric.
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  fill_hmatrix_leaf_node_with_aca_plus(
    const std::vector<HMatrix<spacedim, RangeNumberType> *>
      &                                              leaf_mat_for_kernels,
    const ACAConfig &                                aca_config,
    const std::vector<KernelFunction<spacedim> *> &  kernels,
    const std::vector<RangeNumberType> &             kernel_factors,
    const std::vector<RangeNumberType> &             mass_matrix_factors,
    const std::vector<std::vector<unsigned int>> &   kx_dof_to_cell_topo,
    const std::vector<std::vector<unsigned int>> &   ky_dof_to_cell_topo,
    const BEMValues<dim, spacedim, RangeNumberType> &bem_values,
    const DoFHandler<dim, spacedim> &                kx_dof_handler,
    const DoFHandler<dim, spacedim> &                ky_dof_handler,
    const std::vector<types::global_dof_index> &     kx_dof_i2e_numbering,
    const std::vector<types::global_dof_index> &     ky_dof_i2e_numbering,
    const MappingQGenericExt<dim, spacedim> &        kx_mapping,
    const MappingQGenericExt<dim, spacedim> &        ky_mapping,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_kx_boundary_mesh_to_volume_mesh,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_ky_boundary_mesh_to_volume_mesh,
    const DetectCellNeighboringTypeMethod method_for_cell_neighboring_type,
    CellWiseScratchDataForMassMatrix<dim, spacedim> &mass_matrix_scratch_data,
    PairCellWiseScratchData<dim, spacedim, RangeNumberType> &scratch_data,
    PairCellWisePerTaskData<dim, spacedim, RangeNumberType> &copy_data,
    const bool enable_build_symmetric_hmat = false)
  {
    AssertDimension(leaf_mat_for_kernels.size(), kernels.size());

    const unsigned int kernel_num = kernels.size();

    const std::array<types::global_dof_index, 2> *row_indices =
      leaf_mat_for_kernels[0]->get_row_indices();
    const std::array<types::global_dof_index, 2> *col_indices =
      leaf_mat_for_kernels[0]->get_col_indices();

    /**
     * Flags indicating whether the kernels are to be evaluated at the pair of
     * DoFs.
     */
    std::vector<bool> enable_kernel_evaluations(kernel_num);

    switch (leaf_mat_for_kernels[0]->get_type())
      {
        case FullMatrixType:
          {
            Vector<RangeNumberType> fullmat_coeffs(kernel_num);

            /**
             * Iterate over each DoF index \f$i\f$ in the cluster \$\tau\f$.
             */
            for (size_t i = 0; i < ((*row_indices)[1] - (*row_indices)[0]); i++)
              {
                /**
                 * Iterate over each DoF index \f$j\f$ in the cluster
                 * \$\sigma\f$.
                 */
                for (size_t j = 0; j < ((*col_indices)[1] - (*col_indices)[0]);
                     j++)
                  {
                    /**
                     * Determine if each kernel is to be evaluated at the
                     * current pair of DoFs.
                     */
                    for (unsigned int k = 0; k < kernel_num; k++)
                      {
                        if (enable_build_symmetric_hmat &&
                            kernels[k]->is_symmetric())
                          {
                            /**
                             * When the flag @p enable_build_symmetric_hmat is true and the
                             * kernel function is symmetric, try to build a
                             * symmetric \hmatrix. Otherwise, the whole full
                             * matrix will be always built.
                             */
                            switch (leaf_mat_for_kernels[k]->get_block_type())
                              {
                                case HMatrixSupport::diagonal_block:
                                  {
                                    /**
                                     * A diagonal \hmatrix block as well as its
                                     * associated full matrix should be
                                     * symmetric.
                                     */
                                    Assert(
                                      leaf_mat_for_kernels[k]->get_property() ==
                                        HMatrixSupport::symmetric,
                                      ExcInvalidHMatrixProperty(
                                        leaf_mat_for_kernels[k]
                                          ->get_property()));
                                    Assert(leaf_mat_for_kernels[k]
                                               ->get_fullmatrix()
                                               ->get_property() ==
                                             LAPACKSupport::symmetric,
                                           ExcInvalidLAPACKFullMatrixProperty(
                                             leaf_mat_for_kernels[k]
                                               ->get_fullmatrix()
                                               ->get_property()));

                                    if (j <= i)
                                      {
                                        /**
                                         * Only the diagonal and lower
                                         * triangular elements in the full
                                         * matrix will be evaluated.
                                         */
                                        enable_kernel_evaluations[k] = true;
                                      }
                                    else
                                      {
                                        /**
                                         * Upper triangular elements in the full
                                         * matrix are ignored.
                                         */
                                        enable_kernel_evaluations[k] = false;
                                      }

                                    break;
                                  }
                                case HMatrixSupport::upper_triangular_block:
                                  {
                                    /**
                                     * Do not build \hmatrix block belonging to
                                     * the upper triangular part.
                                     */
                                    enable_kernel_evaluations[k] = false;

                                    break;
                                  }
                                case HMatrixSupport::lower_triangular_block:
                                  {
                                    /**
                                     * When the current \hmatrix block belongs
                                     * to the lower triangular part, evaluate
                                     * all of its elements as usual.
                                     */
                                    enable_kernel_evaluations[k] = true;

                                    break;
                                  }
                                case HMatrixSupport::undefined_block:
                                  {
                                    Assert(false,
                                           ExcInvalidHMatrixBlockType(
                                             leaf_mat_for_kernels[k]
                                               ->get_block_type()));
                                    enable_kernel_evaluations[k] = true;

                                    break;
                                  }
                              }
                          }
                        else
                          {
                            enable_kernel_evaluations[k] = true;
                          }
                      }

                    /**
                     * Perform Sauter quadrature on the pair of DoF indices
                     * \f$(i,j)\f$ for the vector kernel functions. The list of
                     * results are collected into the vector @p fullmat_coeffs.
                     */
                    sauter_assemble_on_one_pair_of_dofs(
                      kernels,
                      kernel_factors,
                      mass_matrix_factors,
                      enable_kernel_evaluations,
                      fullmat_coeffs,
                      kx_dof_i2e_numbering[(*row_indices)[0] + i],
                      ky_dof_i2e_numbering[(*col_indices)[0] + j],
                      kx_dof_to_cell_topo,
                      ky_dof_to_cell_topo,
                      bem_values,
                      kx_dof_handler,
                      ky_dof_handler,
                      kx_mapping,
                      ky_mapping,
                      map_from_kx_boundary_mesh_to_volume_mesh,
                      map_from_ky_boundary_mesh_to_volume_mesh,
                      method_for_cell_neighboring_type,
                      mass_matrix_scratch_data,
                      scratch_data,
                      copy_data);

                    /**
                     * Assign the vector of returned values to each full matrix
                     * corresponding to the kernel function.
                     */
                    for (unsigned int k = 0; k < kernel_num; k++)
                      {
                        if (enable_kernel_evaluations[k])
                          {
                            LAPACKFullMatrixExt<RangeNumberType> *fullmat =
                              leaf_mat_for_kernels[k]->get_fullmatrix();
                            (*fullmat)(i, j) = fullmat_coeffs(k);
                          }
                      }
                  }
              }

            break;
          }
        case RkMatrixType:
          {
            /**
             * Determine if each kernel is to be evaluated at the current pair
             * of DoFs.
             */
            for (unsigned int k = 0; k < kernel_num; k++)
              {
                if (enable_build_symmetric_hmat && kernels[k]->is_symmetric())
                  {
                    switch (leaf_mat_for_kernels[k]->get_block_type())
                      {
                        case HMatrixSupport::lower_triangular_block:
                          {
                            /**
                             * Build the \hmatrix block when it belongs to the
                             * lower triangular part using ACA+.
                             */
                            enable_kernel_evaluations[k] = true;

                            break;
                          }
                        case HMatrixSupport::upper_triangular_block:
                          {
                            /**
                             * Do not build \hmatrix block belonging to the
                             * upper triangular part.
                             */
                            enable_kernel_evaluations[k] = false;

                            break;
                          }
                        case HMatrixSupport::diagonal_block:
                          /**
                           * An rank-k matrix cannot belong to the diagonal
                           * part.
                           */
                        case HMatrixSupport::undefined_block:
                          {
                            Assert(
                              false,
                              ExcInvalidHMatrixBlockType(
                                leaf_mat_for_kernels[k]->get_block_type()));
                            enable_kernel_evaluations[k] = true;

                            break;
                          }
                      }
                  }
                else
                  {
                    enable_kernel_evaluations[k] = true;
                  }
              }

            /**
             * Iterate over each kernel and build the far field matrix block in
             * the rank-k format using ACA+.
             */
            unsigned int counter = 0;
            for (const KernelFunction<spacedim> *kernel : kernels)
              {
                if (enable_kernel_evaluations[counter])
                  {
                    RkMatrix<RangeNumberType> *rkmat =
                      leaf_mat_for_kernels[counter]->get_rkmatrix();

                    aca_plus((*rkmat),
                             aca_config,
                             *kernel,
                             kernel_factors[counter],
                             *row_indices,
                             *col_indices,
                             kx_dof_to_cell_topo,
                             ky_dof_to_cell_topo,
                             bem_values,
                             kx_dof_handler,
                             ky_dof_handler,
                             kx_dof_i2e_numbering,
                             ky_dof_i2e_numbering,
                             kx_mapping,
                             ky_mapping,
                             map_from_kx_boundary_mesh_to_volume_mesh,
                             map_from_ky_boundary_mesh_to_volume_mesh,
                             method_for_cell_neighboring_type,
                             scratch_data,
                             copy_data);
                  }

                counter++;
              }

            break;
          }
        default:
          {
            Assert(false,
                   ExcInvalidHMatrixType(leaf_mat_for_kernels[0]->get_type()));
          }
      }
  }


  /**
   * Fill the leaf nodes in a subrange of an \hmatrix using ACA+.
   *
   * \mynote{This function is to be used for TBB parallelization.}
   *
   * @param range
   * @param aca_config
   * @param kernel
   * @param kernel_factor
   * @param kx_dof_to_cell_topo
   * @param ky_dof_to_cell_topo
   * @param bem_values
   * @param kx_dof_handler
   * @param ky_dof_handler
   * @param kx_mapping
   * @param ky_mapping
   * @param map_from_kx_boundary_mesh_to_volume_mesh
   * @param map_from_ky_boundary_mesh_to_volume_mesh
   * @param method_for_cell_neighboring_type
   * @param enable_build_symmetric_hmat Flag indicating whether symmetric
   * \hmatrix will be built when the kernel function is symmetric.
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  fill_hmatrix_leaf_node_subrange_with_aca_plus(
    const tbb::blocked_range<
      typename std::vector<HMatrix<spacedim, RangeNumberType> *>::iterator>
      &                                              range,
    const ACAConfig &                                aca_config,
    const KernelFunction<spacedim> &                 kernel,
    const RangeNumberType                            kernel_factor,
    const std::vector<std::vector<unsigned int>> &   kx_dof_to_cell_topo,
    const std::vector<std::vector<unsigned int>> &   ky_dof_to_cell_topo,
    const BEMValues<dim, spacedim, RangeNumberType> &bem_values,
    const DoFHandler<dim, spacedim> &                kx_dof_handler,
    const DoFHandler<dim, spacedim> &                ky_dof_handler,
    const std::vector<types::global_dof_index> &     kx_dof_i2e_numbering,
    const std::vector<types::global_dof_index> &     ky_dof_i2e_numbering,
    const MappingQGenericExt<dim, spacedim> &        kx_mapping,
    const MappingQGenericExt<dim, spacedim> &        ky_mapping,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_kx_boundary_mesh_to_volume_mesh,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_ky_boundary_mesh_to_volume_mesh,
    const DetectCellNeighboringTypeMethod method_for_cell_neighboring_type,
    const bool                            enable_build_symmetric_hmat = false)
  {
    /**
     * Define @p PairCellWiseScratchData and @p PairCellWisePerTaskData which
     * are local to the current working thread. This is mandatory because
     * each current working thread should have its own copy of these data.
     */
    PairCellWiseScratchData<dim, spacedim, RangeNumberType> scratch_data(
      kx_dof_handler.get_fe(),
      ky_dof_handler.get_fe(),
      kx_mapping,
      ky_mapping,
      bem_values);
    PairCellWisePerTaskData<dim, spacedim, RangeNumberType> copy_data(
      kx_dof_handler.get_fe(), ky_dof_handler.get_fe());

    for (typename std::vector<HMatrix<spacedim, RangeNumberType> *>::iterator
           iter = range.begin();
         iter != range.end();
         iter++)
      {
        fill_hmatrix_leaf_node_with_aca_plus(
          (*iter),
          aca_config,
          kernel,
          kernel_factor,
          kx_dof_to_cell_topo,
          ky_dof_to_cell_topo,
          bem_values,
          kx_dof_handler,
          ky_dof_handler,
          kx_dof_i2e_numbering,
          ky_dof_i2e_numbering,
          kx_mapping,
          ky_mapping,
          map_from_kx_boundary_mesh_to_volume_mesh,
          map_from_ky_boundary_mesh_to_volume_mesh,
          method_for_cell_neighboring_type,
          scratch_data,
          copy_data,
          enable_build_symmetric_hmat);
      }
  }


  /**
   * Fill the leaf nodes in a subrange of an \hmatrix using ACA+.
   *
   * In the mean time, the FEM mass matrix multiplied by a factor will be added
   * to the near field matrix block.
   *
   * \mynote{This function is to be used for TBB parallelization.}
   *
   * @param range
   * @param aca_config
   * @param kernel
   * @param kernel_factor
   * @param mass_matrix_factor
   * @param kx_dof_to_cell_topo
   * @param ky_dof_to_cell_topo
   * @param bem_values
   * @param mass_matrix_quadrature_formula
   * @param kx_dof_handler
   * @param ky_dof_handler
   * @param kx_mapping
   * @param ky_mapping
   * @param map_from_kx_boundary_mesh_to_volume_mesh
   * @param map_from_ky_boundary_mesh_to_volume_mesh
   * @param method_for_cell_neighboring_type
   * @param enable_build_symmetric_hmat
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  fill_hmatrix_leaf_node_subrange_with_aca_plus(
    const tbb::blocked_range<
      typename std::vector<HMatrix<spacedim, RangeNumberType> *>::iterator>
      &                                              range,
    const ACAConfig &                                aca_config,
    const KernelFunction<spacedim> &                 kernel,
    const RangeNumberType                            kernel_factor,
    const RangeNumberType                            mass_matrix_factor,
    const std::vector<std::vector<unsigned int>> &   kx_dof_to_cell_topo,
    const std::vector<std::vector<unsigned int>> &   ky_dof_to_cell_topo,
    const BEMValues<dim, spacedim, RangeNumberType> &bem_values,
    const QGauss<dim> &                         mass_matrix_quadrature_formula,
    const DoFHandler<dim, spacedim> &           kx_dof_handler,
    const DoFHandler<dim, spacedim> &           ky_dof_handler,
    const std::vector<types::global_dof_index> &kx_dof_i2e_numbering,
    const std::vector<types::global_dof_index> &ky_dof_i2e_numbering,
    const MappingQGenericExt<dim, spacedim> &   kx_mapping,
    const MappingQGenericExt<dim, spacedim> &   ky_mapping,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_kx_boundary_mesh_to_volume_mesh,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_ky_boundary_mesh_to_volume_mesh,
    const DetectCellNeighboringTypeMethod method_for_cell_neighboring_type,
    const bool                            enable_build_symmetric_hmat = false)
  {
    /**
     * Define @p CellWiseScratchData which is local to the current working thread.
     */
    CellWiseScratchDataForMassMatrix<dim, spacedim> mass_matrix_scratch_data(
      kx_dof_handler.get_fe(),
      ky_dof_handler.get_fe(),
      mass_matrix_quadrature_formula,
      update_values | update_JxW_values);

    /**
     * Define @p PairCellWiseScratchData and @p PairCellWisePerTaskData which
     * are local to the current working thread. This is mandatory because
     * each current working thread should have its own copy of these data.
     */
    PairCellWiseScratchData<dim, spacedim, RangeNumberType> scratch_data(
      kx_dof_handler.get_fe(),
      ky_dof_handler.get_fe(),
      kx_mapping,
      ky_mapping,
      bem_values);
    PairCellWisePerTaskData<dim, spacedim, RangeNumberType> copy_data(
      kx_dof_handler.get_fe(), ky_dof_handler.get_fe());

    for (typename std::vector<HMatrix<spacedim, RangeNumberType> *>::iterator
           iter = range.begin();
         iter != range.end();
         iter++)
      {
        fill_hmatrix_leaf_node_with_aca_plus(
          (*iter),
          aca_config,
          kernel,
          kernel_factor,
          mass_matrix_factor,
          kx_dof_to_cell_topo,
          ky_dof_to_cell_topo,
          bem_values,
          kx_dof_handler,
          ky_dof_handler,
          kx_dof_i2e_numbering,
          ky_dof_i2e_numbering,
          kx_mapping,
          ky_mapping,
          map_from_kx_boundary_mesh_to_volume_mesh,
          map_from_ky_boundary_mesh_to_volume_mesh,
          method_for_cell_neighboring_type,
          mass_matrix_scratch_data,
          scratch_data,
          copy_data,
          enable_build_symmetric_hmat);
      }
  }


  /**
   * Fill the leaf nodes in a subrange for a list of \hmatrices, each of which
   * corresponds to a kernel function in the list @p kernels using ACA+.
   *
   * \mynote{This function is to be used for TBB parallelization.
   *
   * As regards the leaf sets for the collection of \hmatrices, they are stored
   * in a vector of vectors, i.e.
   * @p std::vector<std::vector<HMatrix<spacedim, RangeNumberType> *> *>. For
   * the outer vector, its size is equal to the number of kernels or \hmatrices
   * to be built. For the inner vector, it is the leaf set related to each
   * \hmatrix.}
   *
   * @param range
   * @param collection_of_leaf_sets
   * @param aca_config
   * @param kernels
   * @param kernel_factors
   * @param kx_dof_to_cell_topo
   * @param ky_dof_to_cell_topo
   * @param bem_values
   * @param kx_dof_handler
   * @param ky_dof_handler
   * @param kx_mapping
   * @param ky_mapping
   * @param map_from_kx_boundary_mesh_to_volume_mesh
   * @param map_from_ky_boundary_mesh_to_volume_mesh
   * @param method_for_cell_neighboring_type
   * @param enable_build_symmetric_hmat
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  fill_hmatrix_leaf_node_subrange_with_aca_plus(
    const tbb::blocked_range<size_t> &range,
    const std::vector<std::vector<HMatrix<spacedim, RangeNumberType> *> *>
      &                                              collection_of_leaf_sets,
    const ACAConfig &                                aca_config,
    const std::vector<KernelFunction<spacedim> *> &  kernels,
    const std::vector<RangeNumberType> &             kernel_factors,
    const std::vector<std::vector<unsigned int>> &   kx_dof_to_cell_topo,
    const std::vector<std::vector<unsigned int>> &   ky_dof_to_cell_topo,
    const BEMValues<dim, spacedim, RangeNumberType> &bem_values,
    const DoFHandler<dim, spacedim> &                kx_dof_handler,
    const DoFHandler<dim, spacedim> &                ky_dof_handler,
    const std::vector<types::global_dof_index> &     kx_dof_i2e_numbering,
    const std::vector<types::global_dof_index> &     ky_dof_i2e_numbering,
    const MappingQGenericExt<dim, spacedim> &        kx_mapping,
    const MappingQGenericExt<dim, spacedim> &        ky_mapping,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_kx_boundary_mesh_to_volume_mesh,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_ky_boundary_mesh_to_volume_mesh,
    const DetectCellNeighboringTypeMethod method_for_cell_neighboring_type,
    const bool                            enable_build_symmetric_hmat = false)
  {
    AssertDimension(collection_of_leaf_sets.size(), kernels.size());

    const unsigned int kernel_num = kernels.size();

    /**
     * Define @p PairCellWiseScratchData and @p PairCellWisePerTaskData which
     * are local to the current working thread. This is mandatory because
     * each current working thread should have its own copy of these data.
     */
    PairCellWiseScratchData<dim, spacedim, RangeNumberType> scratch_data(
      kx_dof_handler.get_fe(),
      ky_dof_handler.get_fe(),
      kx_mapping,
      ky_mapping,
      bem_values);
    PairCellWisePerTaskData<dim, spacedim, RangeNumberType> copy_data(
      kx_dof_handler.get_fe(), ky_dof_handler.get_fe());

    /**
     * Generate a list of \hmatrix pointers at a given index in the subrange,
     * each of which corresponds to a kernel function in the vector @p kernels.
     */
    std::vector<HMatrix<spacedim, RangeNumberType> *> leaf_mat_list(kernel_num);

    /**
     * Iterate over each index in the subrange.
     */
    for (size_t i = range.begin(); i != range.end(); i++)
      {
        /**
         * Iterate over each kernel function and thus each leaf set in
         * @p collection_of_leaf_sets.
         */
        for (unsigned int k = 0; k < kernel_num; k++)
          {
            leaf_mat_list[k] = (*collection_of_leaf_sets[k])[i];
          }

        fill_hmatrix_leaf_node_with_aca_plus(
          leaf_mat_list,
          aca_config,
          kernels,
          kernel_factors,
          kx_dof_to_cell_topo,
          ky_dof_to_cell_topo,
          bem_values,
          kx_dof_handler,
          ky_dof_handler,
          kx_dof_i2e_numbering,
          ky_dof_i2e_numbering,
          kx_mapping,
          ky_mapping,
          map_from_kx_boundary_mesh_to_volume_mesh,
          map_from_ky_boundary_mesh_to_volume_mesh,
          method_for_cell_neighboring_type,
          scratch_data,
          copy_data,
          enable_build_symmetric_hmat);
      }
  }


  /**
   * Fill the leaf nodes in a subrange for a list of \hmatrices, each of which
   * corresponds to a kernel function in the list @p kernels using ACA+.
   *
   * In the meantime, the FEM mass matrix multiplied by a factor will be added
   * to the near field matrix block.
   *
   * \mynote{This function is to be used for TBB parallelization.}
   *
   * @param range
   * @param collection_of_leaf_sets
   * @param aca_config
   * @param kernels
   * @param kernel_factors
   * @param mass_matrix_factors
   * @param kx_dof_to_cell_topo
   * @param ky_dof_to_cell_topo
   * @param bem_values
   * @param mass_matrix_quadrature_formula
   * @param kx_dof_handler
   * @param ky_dof_handler
   * @param kx_mapping
   * @param ky_mapping
   * @param map_from_kx_boundary_mesh_to_volume_mesh
   * @param map_from_ky_boundary_mesh_to_volume_mesh
   * @param method_for_cell_neighboring_type
   * @param enable_build_symmetric_hmat
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  fill_hmatrix_leaf_node_subrange_with_aca_plus(
    const tbb::blocked_range<size_t> &range,
    const std::vector<std::vector<HMatrix<spacedim, RangeNumberType> *> *>
      &                                              collection_of_leaf_sets,
    const ACAConfig &                                aca_config,
    const std::vector<KernelFunction<spacedim> *> &  kernels,
    const std::vector<RangeNumberType> &             kernel_factors,
    const std::vector<RangeNumberType> &             mass_matrix_factors,
    const std::vector<std::vector<unsigned int>> &   kx_dof_to_cell_topo,
    const std::vector<std::vector<unsigned int>> &   ky_dof_to_cell_topo,
    const BEMValues<dim, spacedim, RangeNumberType> &bem_values,
    const QGauss<dim> &                         mass_matrix_quadrature_formula,
    const DoFHandler<dim, spacedim> &           kx_dof_handler,
    const DoFHandler<dim, spacedim> &           ky_dof_handler,
    const std::vector<types::global_dof_index> &kx_dof_i2e_numbering,
    const std::vector<types::global_dof_index> &ky_dof_i2e_numbering,
    const MappingQGenericExt<dim, spacedim> &   kx_mapping,
    const MappingQGenericExt<dim, spacedim> &   ky_mapping,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_kx_boundary_mesh_to_volume_mesh,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_ky_boundary_mesh_to_volume_mesh,
    const DetectCellNeighboringTypeMethod method_for_cell_neighboring_type,
    const bool                            enable_build_symmetric_hmat = false)
  {
    AssertDimension(collection_of_leaf_sets.size(), kernels.size());

    const unsigned int kernel_num = kernels.size();

    /**
     * Define @p CellWiseScratchData which is local to the current working thread.
     */
    CellWiseScratchDataForMassMatrix<dim, spacedim> mass_matrix_scratch_data(
      kx_dof_handler.get_fe(),
      ky_dof_handler.get_fe(),
      mass_matrix_quadrature_formula,
      update_values | update_JxW_values);

    /**
     * Define @p PairCellWiseScratchData and @p PairCellWisePerTaskData which
     * are local to the current working thread. This is mandatory because
     * each current working thread should have its own copy of these data.
     */
    PairCellWiseScratchData<dim, spacedim, RangeNumberType> scratch_data(
      kx_dof_handler.get_fe(),
      ky_dof_handler.get_fe(),
      kx_mapping,
      ky_mapping,
      bem_values);
    PairCellWisePerTaskData<dim, spacedim, RangeNumberType> copy_data(
      kx_dof_handler.get_fe(), ky_dof_handler.get_fe());

    /**
     * Generate a list of \hmatrix pointers at a given index in the subrange,
     * each of which corresponds to a kernel function in the vector @p kernels.
     */
    std::vector<HMatrix<spacedim, RangeNumberType> *> leaf_mat_list(kernel_num);

    /**
     * Iterate over each index in the subrange.
     */
    for (size_t i = range.begin(); i != range.end(); i++)
      {
        /**
         * Iterate over each kernel function and thus each leaf set in
         * @p collection_of_leaf_sets.
         */
        for (unsigned int k = 0; k < kernel_num; k++)
          {
            leaf_mat_list[k] = (*collection_of_leaf_sets[k])[i];
          }

        fill_hmatrix_leaf_node_with_aca_plus(
          leaf_mat_list,
          aca_config,
          kernels,
          kernel_factors,
          mass_matrix_factors,
          kx_dof_to_cell_topo,
          ky_dof_to_cell_topo,
          bem_values,
          kx_dof_handler,
          ky_dof_handler,
          kx_dof_i2e_numbering,
          ky_dof_i2e_numbering,
          kx_mapping,
          ky_mapping,
          map_from_kx_boundary_mesh_to_volume_mesh,
          map_from_ky_boundary_mesh_to_volume_mesh,
          mass_matrix_scratch_data,
          scratch_data,
          copy_data,
          enable_build_symmetric_hmat);
      }
  }


  /**
   * Fill the leaf set of the \hmatrix using ACA+, where the hierarchical
   * structure of the \hmatrix has been built with respect to a block cluster
   * tree and the memory for the matrices in the leaf set has been allocated.
   *
   * For the near field matrices in the leaf set, full matrices will be built
   * whose elements will be obtained from the evaluation of the double integral
   * in Galerkin-BEM. For the far field admissible matrices in the leaf set,
   * rank-k matrices will be built using ACA+.
   *
   * This version serially processes each \hmatnode in the leaf set one-by-one.
   *
   * @param hmat
   * @param aca_config
   * @param kernel
   * @param factor
   * @param kx_dof_to_cell_topo
   * @param ky_dof_to_cell_topo
   * @param bem_values
   * @param kx_dof_handler
   * @param ky_dof_handler
   * @param kx_mapping
   * @param ky_mapping
   * @param map_from_kx_boundary_mesh_to_volume_mesh
   * @param map_from_ky_boundary_mesh_to_volume_mesh
   * @param method_for_cell_neighboring_type
   * @param enable_build_symmetric_hmat Flag indicating whether symmetric
   * \hmatrix will be built when the kernel function is symmetric.
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  fill_hmatrix_with_aca_plus(
    HMatrix<spacedim, RangeNumberType> &                   hmat,
    const ACAConfig &                                      aca_config,
    const KernelFunction<spacedim> &                       kernel,
    const RangeNumberType                                  kernel_factor,
    const std::vector<std::vector<unsigned int>> &         kx_dof_to_cell_topo,
    const std::vector<std::vector<unsigned int>> &         ky_dof_to_cell_topo,
    const SauterQuadratureRule<dim> &                      sauter_quad_rule,
    const DoFHandler<dim, spacedim> &                      kx_dof_handler,
    const DoFHandler<dim, spacedim> &                      ky_dof_handler,
    const MappingQGenericExt<dim, spacedim> &              kx_mapping,
    const MappingQGenericExt<dim, spacedim> &              ky_mapping,
    typename MappingQGeneric<dim, spacedim>::InternalData &kx_mapping_data,
    typename MappingQGeneric<dim, spacedim>::InternalData &ky_mapping_data,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_kx_boundary_mesh_to_volume_mesh,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_ky_boundary_mesh_to_volume_mesh,
    const DetectCellNeighboringTypeMethod method_for_cell_neighboring_type,
    const bool                            enable_build_symmetric_hmat = false)
  {
    /**
     * Precalculate data tables for shape values at quadrature points.
     *
     * \mynote{Precalculate shape function values and their gradient values
     * at each quadrature point. N.B.
     * 1. The data tables for shape function values and their gradient
     * values should be calculated for both function space on \f$K_x\f$ and
     * function space on \f$K_y\f$.
     * 2. Being different from the integral in FEM, the integral in BEM
     * handled by Sauter's quadrature rule has multiple parts of \f$k_3\f$
     * (except the regular cell neighboring type), each of which should be
     * evaluated at a different set of quadrature points in the unit cell
     * after coordinate transformation from the parametric space. Therefore,
     * a dimension with respect to \f$k_3\f$ term index should be added to
     * the data table compared to the usual FEValues and this brings about
     * the class @p BEMValues.}
     */
    BEMValues<dim, spacedim, RangeNumberType> bem_values(
      kx_dof_handler.get_fe(),
      ky_dof_handler.get_fe(),
      kx_mapping_data,
      ky_mapping_data,
      sauter_quad_rule.quad_rule_for_same_panel,
      sauter_quad_rule.quad_rule_for_common_edge,
      sauter_quad_rule.quad_rule_for_common_vertex,
      sauter_quad_rule.quad_rule_for_regular);

    bem_values.fill_shape_function_value_tables();

    /**
     * Define @p PairCellWiseScratchData and @p PairCellWisePerTaskData which
     * are local to the current working thread. This is mandatory because
     * each current working thread should have its own copy of these data.
     */
    PairCellWiseScratchData<dim, spacedim, RangeNumberType> scratch_data(
      kx_dof_handler.get_fe(), ky_dof_handler.get_fe(), bem_values);
    PairCellWisePerTaskData<dim, spacedim, RangeNumberType> copy_data(
      kx_dof_handler.get_fe(), ky_dof_handler.get_fe());

    for (HMatrix<spacedim, RangeNumberType> *leaf_mat : hmat.get_leaf_set())
      {
        fill_hmatrix_leaf_node_with_aca_plus(
          leaf_mat,
          aca_config,
          kernel,
          kernel_factor,
          kx_dof_to_cell_topo,
          ky_dof_to_cell_topo,
          bem_values,
          kx_dof_handler,
          ky_dof_handler,
          kx_mapping,
          ky_mapping,
          map_from_kx_boundary_mesh_to_volume_mesh,
          map_from_ky_boundary_mesh_to_volume_mesh,
          method_for_cell_neighboring_type,
          scratch_data,
          copy_data,
          enable_build_symmetric_hmat);
      }
  }


  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  fill_hmatrix_with_aca_plus(
    HMatrix<spacedim, RangeNumberType> &          hmat,
    const ACAConfig &                             aca_config,
    const KernelFunction<spacedim> &              kernel,
    const RangeNumberType                         kernel_factor,
    const RangeNumberType                         mass_matrix_factor,
    const std::vector<std::vector<unsigned int>> &kx_dof_to_cell_topo,
    const std::vector<std::vector<unsigned int>> &ky_dof_to_cell_topo,
    const SauterQuadratureRule<dim> &             sauter_quad_rule,
    const QGauss<dim> &                      mass_matrix_quadrature_formula,
    const DoFHandler<dim, spacedim> &        kx_dof_handler,
    const DoFHandler<dim, spacedim> &        ky_dof_handler,
    const MappingQGenericExt<dim, spacedim> &kx_mapping,
    const MappingQGenericExt<dim, spacedim> &ky_mapping,
    typename MappingQGeneric<dim, spacedim>::InternalData &kx_mapping_data,
    typename MappingQGeneric<dim, spacedim>::InternalData &ky_mapping_data,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_kx_boundary_mesh_to_volume_mesh,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_ky_boundary_mesh_to_volume_mesh,
    const DetectCellNeighboringTypeMethod method_for_cell_neighboring_type,
    const bool                            enable_build_symmetric_hmat = false)
  {
    /**
     * Precalculate data tables for shape values at quadrature points.
     *
     * \mynote{Precalculate shape function values and their gradient values
     * at each quadrature point. N.B.
     * 1. The data tables for shape function values and their gradient
     * values should be calculated for both function space on \f$K_x\f$ and
     * function space on \f$K_y\f$.
     * 2. Being different from the integral in FEM, the integral in BEM
     * handled by Sauter's quadrature rule has multiple parts of \f$k_3\f$
     * (except the regular cell neighboring type), each of which should be
     * evaluated at a different set of quadrature points in the unit cell
     * after coordinate transformation from the parametric space. Therefore,
     * a dimension with respect to \f$k_3\f$ term index should be added to
     * the data table compared to the usual FEValues and this brings about
     * the class @p BEMValues.}
     */
    BEMValues<dim, spacedim, RangeNumberType> bem_values(
      kx_dof_handler.get_fe(),
      ky_dof_handler.get_fe(),
      kx_mapping_data,
      ky_mapping_data,
      sauter_quad_rule.quad_rule_for_same_panel,
      sauter_quad_rule.quad_rule_for_common_edge,
      sauter_quad_rule.quad_rule_for_common_vertex,
      sauter_quad_rule.quad_rule_for_regular);

    bem_values.fill_shape_function_value_tables();

    /**
     * Define @p CellWiseScratchDataForMassMatrix which is local to current
     * working thread.
     */
    CellWiseScratchDataForMassMatrix<dim, spacedim> mass_matrix_scratch_data(
      kx_dof_handler.get_fe(),
      ky_dof_handler.get_fe(),
      mass_matrix_quadrature_formula,
      update_values | update_JxW_values);

    /**
     * Define @p PairCellWiseScratchData and @p PairCellWisePerTaskData which
     * are local to the current working thread. This is mandatory because
     * each current working thread should have its own copy of these data.
     */
    PairCellWiseScratchData<dim, spacedim, RangeNumberType> scratch_data(
      kx_dof_handler.get_fe(),
      ky_dof_handler.get_fe(),
      kx_mapping,
      ky_mapping,
      bem_values);
    PairCellWisePerTaskData<dim, spacedim, RangeNumberType> copy_data(
      kx_dof_handler.get_fe(), ky_dof_handler.get_fe());

    for (HMatrix<spacedim, RangeNumberType> *leaf_mat : hmat.get_leaf_set())
      {
        fill_hmatrix_leaf_node_with_aca_plus(
          leaf_mat,
          aca_config,
          kernel,
          kernel_factor,
          mass_matrix_factor,
          kx_dof_to_cell_topo,
          ky_dof_to_cell_topo,
          bem_values,
          kx_dof_handler,
          ky_dof_handler,
          kx_mapping,
          ky_mapping,
          map_from_kx_boundary_mesh_to_volume_mesh,
          map_from_ky_boundary_mesh_to_volume_mesh,
          method_for_cell_neighboring_type,
          mass_matrix_scratch_data,
          scratch_data,
          copy_data,
          enable_build_symmetric_hmat);
      }
  }


  /**
   * Fill the leaf set of the \hmatrix using ACA+, where the hierarchical
   * structure of the \hmatrix has been built with respect to a block cluster
   * tree and the memory for the matrices in the leaf set has been allocated.
   *
   * For the near field matrices in the leaf set, full matrices will be built
   * whose elements will be obtained from the evaluation of the double integral
   * in Galerkin-BEM. For the far field admissible matrices in the leaf set,
   * rank-k matrices will be built using ACA+.
   *
   * \mynote{In this version, the computation of the list of \hmatnodes in the
   * leaf set is parallelized.}
   *
   * @param thread_num
   * @param hmat
   * @param aca_config
   * @param kernel
   * @param kernel_factor
   * @param kx_dof_to_cell_topo
   * @param ky_dof_to_cell_topo
   * @param bem_values
   * @param kx_dof_handler
   * @param ky_dof_handler
   * @param kx_mapping
   * @param ky_mapping
   * @param map_from_kx_boundary_mesh_to_volume_mesh
   * @param map_from_ky_boundary_mesh_to_volume_mesh
   * @param method_for_cell_neighboring_type
   * @param enable_build_symmetric_hmat
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  fill_hmatrix_with_aca_plus_smp(
    const unsigned int                                     thread_num,
    HMatrix<spacedim, RangeNumberType> &                   hmat,
    const ACAConfig &                                      aca_config,
    const KernelFunction<spacedim> &                       kernel,
    const RangeNumberType                                  kernel_factor,
    const std::vector<std::vector<unsigned int>> &         kx_dof_to_cell_topo,
    const std::vector<std::vector<unsigned int>> &         ky_dof_to_cell_topo,
    const SauterQuadratureRule<dim> &                      sauter_quad_rule,
    const DoFHandler<dim, spacedim> &                      kx_dof_handler,
    const DoFHandler<dim, spacedim> &                      ky_dof_handler,
    const std::vector<types::global_dof_index> &           kx_dof_i2e_numbering,
    const std::vector<types::global_dof_index> &           ky_dof_i2e_numbering,
    const MappingQGenericExt<dim, spacedim> &              kx_mapping,
    const MappingQGenericExt<dim, spacedim> &              ky_mapping,
    typename MappingQGeneric<dim, spacedim>::InternalData &kx_mapping_data,
    typename MappingQGeneric<dim, spacedim>::InternalData &ky_mapping_data,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_kx_boundary_mesh_to_volume_mesh,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_ky_boundary_mesh_to_volume_mesh,
    const DetectCellNeighboringTypeMethod method_for_cell_neighboring_type,
    const bool                            enable_build_symmetric_hmat = false)
  {
    /**
     * Precalculate data tables for shape values at quadrature points.
     *
     * \mynote{Precalculate shape function values and their gradient values
     * at each quadrature point. N.B.
     * 1. The data tables for shape function values and their gradient
     * values should be calculated for both function space on \f$K_x\f$ and
     * function space on \f$K_y\f$.
     * 2. Being different from the integral in FEM, the integral in BEM
     * handled by Sauter's quadrature rule has multiple parts of \f$k_3\f$
     * (except the regular cell neighboring type), each of which should be
     * evaluated at a different set of quadrature points in the unit cell
     * after coordinate transformation from the parametric space. Therefore,
     * a dimension with respect to \f$k_3\f$ term index should be added to
     * the data table compared to the usual FEValues and this brings about
     * the class @p BEMValues.}
     */
    BEMValues<dim, spacedim, RangeNumberType> bem_values(
      kx_dof_handler.get_fe(),
      ky_dof_handler.get_fe(),
      kx_mapping_data,
      ky_mapping_data,
      sauter_quad_rule.quad_rule_for_same_panel,
      sauter_quad_rule.quad_rule_for_common_edge,
      sauter_quad_rule.quad_rule_for_common_vertex,
      sauter_quad_rule.quad_rule_for_regular);

    bem_values.fill_shape_function_value_tables();

    std::vector<HMatrix<spacedim, RangeNumberType> *> &leaf_set =
      hmat.get_leaf_set();
    /**
     * Estimate the grain size.
     */
    unsigned int grain_size = leaf_set.size() / thread_num;
    if (grain_size == 0)
      {
        grain_size = 1;
      }

    /**
     * \mynote{N.B. There are two places where argument passing for
     * parallelization is involved: 1. passing to @p std::bind; 2. passing to
     * the working function.
     *
     * Local variables captured by @p std::bind are by default pass-by-value.
     * Hence, for capturing large objects, pass-by-reference should be adopted
     * instead of pass-by-value, which is realized by adding the prefix
     * @p std::cref for const reference or @p std::ref for mutable reference.
     *
     * @p PairCellWiseScratchData and @p PairCellWisePerTaskData should be
     * captured by @p std::bind via pass-by-value, then for each working thread,
     * a copy of them will automatically be made.
     *
     * Meanwhile, in a same working thread, the working function will be
     * called for each object specified by the index or iterator in
     * the blocked range, during which @p PairCellWiseScratchData and
     * @p PairCellWisePerTaskData can be reused. Therefore, they should be
     * passed into the working function by non-const reference.
     *
     * For other variables to be captured like @p bem_values,
     * @p kx/ky_dof_handler, * @p kx/ky_mapping and
     * @p map_from_kx/ky_bvoundary_mesh_to_volume_mesh, since they will be not
     * modified at all, they are captured by @p std::bind via const reference
     * and then passed to the working function by const reference.}
     */
    parallel::internal::parallel_for(
      leaf_set.begin(),
      leaf_set.end(),
      std::bind(static_cast<void (*)(
                  const tbb::blocked_range<typename std::vector<
                    HMatrix<spacedim, RangeNumberType> *>::iterator> &,
                  const ACAConfig &,
                  const KernelFunction<spacedim> &,
                  const RangeNumberType,
                  const std::vector<std::vector<unsigned int>> &,
                  const std::vector<std::vector<unsigned int>> &,
                  const BEMValues<dim, spacedim, RangeNumberType> &,
                  const DoFHandler<dim, spacedim> &,
                  const DoFHandler<dim, spacedim> &,
                  const std::vector<types::global_dof_index> &,
                  const std::vector<types::global_dof_index> &,
                  const MappingQGenericExt<dim, spacedim> &,
                  const MappingQGenericExt<dim, spacedim> &,
                  const std::map<
                    typename Triangulation<dim, spacedim>::cell_iterator,
                    typename Triangulation<dim + 1, spacedim>::face_iterator> &,
                  const std::map<
                    typename Triangulation<dim, spacedim>::cell_iterator,
                    typename Triangulation<dim + 1, spacedim>::face_iterator> &,
                  const DetectCellNeighboringTypeMethod,
                  const bool)>(fill_hmatrix_leaf_node_subrange_with_aca_plus),
                std::placeholders::_1,
                std::cref(aca_config),
                std::cref(kernel),
                kernel_factor,
                std::cref(kx_dof_to_cell_topo),
                std::cref(ky_dof_to_cell_topo),
                std::cref(bem_values),
                std::cref(kx_dof_handler),
                std::cref(ky_dof_handler),
                std::cref(kx_dof_i2e_numbering),
                std::cref(ky_dof_i2e_numbering),
                std::cref(kx_mapping),
                std::cref(ky_mapping),
                std::cref(map_from_kx_boundary_mesh_to_volume_mesh),
                std::cref(map_from_ky_boundary_mesh_to_volume_mesh),
                method_for_cell_neighboring_type,
                enable_build_symmetric_hmat),
      grain_size);
  }


  /**
   * Fill the leaf set of the \hmatrix using ACA+, where the hierarchical
   * structure of the \hmatrix has been built with respect to a block cluster
   * tree and the memory for the matrices in the leaf set has been allocated.
   *
   * In the mean time, the FEM mass matrix multiplied by a factor will be added
   * to the near field matrix block.
   *
   * For the near field matrices in the leaf set, full matrices will be built
   * whose elements will be obtained from the evaluation of the double integral
   * in Galerkin-BEM. For the far field admissible matrices in the leaf set,
   * rank-k matrices will be built using ACA+.
   *
   * @param thread_num
   * @param hmat
   * @param aca_config
   * @param kernel
   * @param kernel_factor
   * @param mass_matrix_factor
   * @param kx_dof_to_cell_topo
   * @param ky_dof_to_cell_topo
   * @param bem_values
   * @param mass_matrix_quadrature_formula
   * @param kx_dof_handler
   * @param ky_dof_handler
   * @param kx_mapping
   * @param ky_mapping
   * @param map_from_kx_boundary_mesh_to_volume_mesh
   * @param map_from_ky_boundary_mesh_to_volume_mesh
   * @param method_for_cell_neighboring_type
   * @param enable_build_symmetric_hmat
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  fill_hmatrix_with_aca_plus_smp(
    const unsigned int                            thread_num,
    HMatrix<spacedim, RangeNumberType> &          hmat,
    const ACAConfig &                             aca_config,
    const KernelFunction<spacedim> &              kernel,
    const RangeNumberType                         kernel_factor,
    const RangeNumberType                         mass_matrix_factor,
    const std::vector<std::vector<unsigned int>> &kx_dof_to_cell_topo,
    const std::vector<std::vector<unsigned int>> &ky_dof_to_cell_topo,
    const SauterQuadratureRule<dim> &             sauter_quad_rule,
    const QGauss<dim> &                         mass_matrix_quadrature_formula,
    const DoFHandler<dim, spacedim> &           kx_dof_handler,
    const DoFHandler<dim, spacedim> &           ky_dof_handler,
    const std::vector<types::global_dof_index> &kx_dof_i2e_numbering,
    const std::vector<types::global_dof_index> &ky_dof_i2e_numbering,
    const MappingQGenericExt<dim, spacedim> &   kx_mapping,
    const MappingQGenericExt<dim, spacedim> &   ky_mapping,
    typename MappingQGeneric<dim, spacedim>::InternalData &kx_mapping_data,
    typename MappingQGeneric<dim, spacedim>::InternalData &ky_mapping_data,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_kx_boundary_mesh_to_volume_mesh,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_ky_boundary_mesh_to_volume_mesh,
    const DetectCellNeighboringTypeMethod method_for_cell_neighboring_type,
    const bool                            enable_build_symmetric_hmat = false)
  {
    /**
     * Precalculate data tables for shape values at quadrature points.
     *
     * \mynote{Precalculate shape function values and their gradient values
     * at each quadrature point. N.B.
     * 1. The data tables for shape function values and their gradient
     * values should be calculated for both function space on \f$K_x\f$ and
     * function space on \f$K_y\f$.
     * 2. Being different from the integral in FEM, the integral in BEM
     * handled by Sauter's quadrature rule has multiple parts of \f$k_3\f$
     * (except the regular cell neighboring type), each of which should be
     * evaluated at a different set of quadrature points in the unit cell
     * after coordinate transformation from the parametric space. Therefore,
     * a dimension with respect to \f$k_3\f$ term index should be added to
     * the data table compared to the usual FEValues and this brings about
     * the class @p BEMValues.}
     */
    BEMValues<dim, spacedim, RangeNumberType> bem_values(
      kx_dof_handler.get_fe(),
      ky_dof_handler.get_fe(),
      kx_mapping_data,
      ky_mapping_data,
      sauter_quad_rule.quad_rule_for_same_panel,
      sauter_quad_rule.quad_rule_for_common_edge,
      sauter_quad_rule.quad_rule_for_common_vertex,
      sauter_quad_rule.quad_rule_for_regular);

    bem_values.fill_shape_function_value_tables();

    std::vector<HMatrix<spacedim, RangeNumberType> *> &leaf_set =
      hmat.get_leaf_set();
    /**
     * Estimate the grain size.
     */
    unsigned int grain_size = leaf_set.size() / thread_num;
    if (grain_size == 0)
      {
        grain_size = 1;
      }

    /**
     * \mynote{N.B. There are two places where argument passing for
     * parallelization is involved: 1. passing to @p std::bind; 2. passing to
     * the working function.
     *
     * Local variables captured by @p std::bind are by default pass-by-value.
     * Hence, for capturing large objects, pass-by-reference should be adopted
     * instead of pass-by-value, which is realized by adding the prefix
     * @p std::cref for const reference or @p std::ref for mutable reference.
     *
     * @p PairCellWiseScratchData and @p PairCellWisePerTaskData should be
     * captured by @p std::bind via pass-by-value, then for each working thread,
     * a copy of them will automatically be made.
     *
     * Meanwhile, in a same working thread, the working function will be
     * called for each object specified by the index or iterator in
     * the blocked range, during which @p PairCellWiseScratchData and
     * @p PairCellWisePerTaskData can be reused. Therefore, they should be
     * passed into the working function by non-const reference.
     *
     * For other variables to be captured like @p bem_values,
     * @p kx/ky_dof_handler, * @p kx/ky_mapping and
     * @p map_from_kx/ky_bvoundary_mesh_to_volume_mesh, since they will be not
     * modified at all, they are captured by @p std::bind via const reference
     * and then passed to the working function by const reference.}
     */
    parallel::internal::parallel_for(
      leaf_set.begin(),
      leaf_set.end(),
      std::bind(static_cast<void (*)(
                  const tbb::blocked_range<typename std::vector<
                    HMatrix<spacedim, RangeNumberType> *>::iterator> &,
                  const ACAConfig &,
                  const KernelFunction<spacedim> &,
                  const RangeNumberType,
                  const RangeNumberType,
                  const std::vector<std::vector<unsigned int>> &,
                  const std::vector<std::vector<unsigned int>> &,
                  const BEMValues<dim, spacedim, RangeNumberType> &,
                  const QGauss<dim> &,
                  const DoFHandler<dim, spacedim> &,
                  const DoFHandler<dim, spacedim> &,
                  const std::vector<types::global_dof_index> &,
                  const std::vector<types::global_dof_index> &,
                  const MappingQGenericExt<dim, spacedim> &,
                  const MappingQGenericExt<dim, spacedim> &,
                  const std::map<
                    typename Triangulation<dim, spacedim>::cell_iterator,
                    typename Triangulation<dim + 1, spacedim>::face_iterator> &,
                  const std::map<
                    typename Triangulation<dim, spacedim>::cell_iterator,
                    typename Triangulation<dim + 1, spacedim>::face_iterator> &,
                  const DetectCellNeighboringTypeMethod,
                  const bool)>(fill_hmatrix_leaf_node_subrange_with_aca_plus),
                std::placeholders::_1,
                std::cref(aca_config),
                std::cref(kernel),
                kernel_factor,
                mass_matrix_factor,
                std::cref(kx_dof_to_cell_topo),
                std::cref(ky_dof_to_cell_topo),
                std::cref(bem_values),
                std::cref(mass_matrix_quadrature_formula),
                std::cref(kx_dof_handler),
                std::cref(ky_dof_handler),
                std::cref(kx_dof_i2e_numbering),
                std::cref(ky_dof_i2e_numbering),
                std::cref(kx_mapping),
                std::cref(ky_mapping),
                std::cref(map_from_kx_boundary_mesh_to_volume_mesh),
                std::cref(map_from_ky_boundary_mesh_to_volume_mesh),
                method_for_cell_neighboring_type,
                enable_build_symmetric_hmat),
      grain_size);
  }


  /**
   * Fill the leaf sets for a collection of \hmatrices using ACA+ with respect
   * to a list of kernel functions.
   *
   * \mynote{This requires the trial spaces as well as the ansatz spaces of
   * these \hmatrices to be the same. Therefore, the hierarchical structures for
   * all the \hmatrices should be the same, which should have been built with
   * respect to a same \bct. The memory for the matrices in the leaf set should
   * have been allocated beforehand.
   *
   * For the near field matrices in the leaf set, full matrices will be built
   * whose elements will be obtained from the evaluation of the double integral
   * in Galerkin-BEM. For the far field admissible matrices in the leaf set,
   * rank-k matrices will be built using ACA+.
   *
   * As regards the leaf sets for the collection of \hmatrices, they are stored
   * in a vector of vectors, i.e.
   * @p std::vector<std::vector<HMatrix<spacedim, RangeNumberType> *> *>. For
   * the outer vector, its size is equal to the number of kernels or \hmatrices
   * to be built. For the inner vector, it is the leaf set related to each
   * \hmatrix.}
   *
   * @param thread_num
   * @param hmats
   * @param aca_config
   * @param kernels
   * @param kernel_factors
   * @param kx_dof_to_cell_topo
   * @param ky_dof_to_cell_topo
   * @param bem_values
   * @param kx_dof_handler
   * @param ky_dof_handler
   * @param kx_mapping
   * @param ky_mapping
   * @param map_from_kx_boundary_mesh_to_volume_mesh
   * @param map_from_ky_boundary_mesh_to_volume_mesh
   * @param method_for_cell_neighboring_type
   * @param enable_build_symmetric_hmat
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  fill_hmatrix_with_aca_plus_smp(
    const unsigned int                                     thread_num,
    std::vector<HMatrix<spacedim, RangeNumberType> *> &    hmats,
    const ACAConfig &                                      aca_config,
    const std::vector<KernelFunction<spacedim> *> &        kernels,
    const std::vector<RangeNumberType> &                   kernel_factors,
    const std::vector<std::vector<unsigned int>> &         kx_dof_to_cell_topo,
    const std::vector<std::vector<unsigned int>> &         ky_dof_to_cell_topo,
    const SauterQuadratureRule<dim> &                      sauter_quad_rule,
    const DoFHandler<dim, spacedim> &                      kx_dof_handler,
    const DoFHandler<dim, spacedim> &                      ky_dof_handler,
    const std::vector<types::global_dof_index> &           kx_dof_i2e_numbering,
    const std::vector<types::global_dof_index> &           ky_dof_i2e_numbering,
    const MappingQGenericExt<dim, spacedim> &              kx_mapping,
    const MappingQGenericExt<dim, spacedim> &              ky_mapping,
    typename MappingQGeneric<dim, spacedim>::InternalData &kx_mapping_data,
    typename MappingQGeneric<dim, spacedim>::InternalData &ky_mapping_data,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_kx_boundary_mesh_to_volume_mesh,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_ky_boundary_mesh_to_volume_mesh,
    const DetectCellNeighboringTypeMethod method_for_cell_neighboring_type,
    const bool                            enable_build_symmetric_hmat = false)
  {
    AssertDimension(hmats.size(), kernels.size());

    const unsigned int kernel_num = kernels.size();

    /**
     * Precalculate data tables for shape values at quadrature points.
     *
     * \mynote{Precalculate shape function values and their gradient values
     * at each quadrature point. N.B.
     * 1. The data tables for shape function values and their gradient
     * values should be calculated for both function space on \f$K_x\f$ and
     * function space on \f$K_y\f$.
     * 2. Being different from the integral in FEM, the integral in BEM
     * handled by Sauter's quadrature rule has multiple parts of \f$k_3\f$
     * (except the regular cell neighboring type), each of which should be
     * evaluated at a different set of quadrature points in the unit cell
     * after coordinate transformation from the parametric space. Therefore,
     * a dimension with respect to \f$k_3\f$ term index should be added to
     * the data table compared to the usual FEValues and this brings about
     * the class @p BEMValues.}
     */
    BEMValues<dim, spacedim, RangeNumberType> bem_values(
      kx_dof_handler.get_fe(),
      ky_dof_handler.get_fe(),
      kx_mapping_data,
      ky_mapping_data,
      sauter_quad_rule.quad_rule_for_same_panel,
      sauter_quad_rule.quad_rule_for_common_edge,
      sauter_quad_rule.quad_rule_for_common_vertex,
      sauter_quad_rule.quad_rule_for_regular);

    bem_values.fill_shape_function_value_tables();

    /**
     * Extract the collection of leaf sets from the vector of \hmatrices.
     */
    std::vector<std::vector<HMatrix<spacedim, RangeNumberType> *> *>
      collection_of_leaf_sets(kernel_num);
    for (unsigned int k = 0; k < kernel_num; k++)
      {
        collection_of_leaf_sets[k] = &(hmats[k]->get_leaf_set());
      }

    /**
     * Estimate the grain size by evenly distributing the leaf sets to threads.
     */
    const size_t leaf_set_size = collection_of_leaf_sets[0]->size();
    unsigned int grain_size    = leaf_set_size / thread_num;
    if (grain_size == 0)
      {
        grain_size = 1;
      }

    /**
     * \mynote{N.B. There are two places where argument passing for
     * parallelization is involved: 1. passing to @p std::bind; 2. passing to
     * the working function.
     *
     * Local variables captured by @p std::bind are by default pass-by-value.
     * Hence, for capturing large objects, pass-by-reference should be adopted
     * instead of pass-by-value, which is realized by adding the prefix
     * @p std::cref for const reference or @p std::ref for mutable reference.
     *
     * @p PairCellWiseScratchData and @p PairCellWisePerTaskData should be
     * captured by @p std::bind via pass-by-value, then for each working thread,
     * a copy of them will automatically be made.
     *
     * Meanwhile, in a same working thread, the working function will be
     * called for each object specified by the index or iterator in
     * the blocked range, during which @p PairCellWiseScratchData and
     * @p PairCellWisePerTaskData can be reused. Therefore, they should be
     * passed into the working function by non-const reference.
     *
     * For other variables to be captured like @p bem_values,
     * @p kx/ky_dof_handler, * @p kx/ky_mapping and
     * @p map_from_kx/ky_bvoundary_mesh_to_volume_mesh, since they will be not
     * modified at all, they are captured by @p std::bind via const reference
     * and then passed to the working function by const reference.}
     */

    /**
     * Here we use @p size_t index to refer to the \hmatrix in the leaf set
     * and the kernel function in the provided list @p kernels.
     */
    tbb::parallel_for(
      tbb::blocked_range<size_t>(0, leaf_set_size, grain_size),
      std::bind(
        static_cast<void (*)(
          const tbb::blocked_range<size_t> &,
          const std::vector<std::vector<HMatrix<spacedim, RangeNumberType> *> *>
            &,
          const ACAConfig &,
          const std::vector<KernelFunction<spacedim> *> &,
          const std::vector<RangeNumberType> &,
          const std::vector<std::vector<unsigned int>> &,
          const std::vector<std::vector<unsigned int>> &,
          const BEMValues<dim, spacedim, RangeNumberType> &,
          const DoFHandler<dim, spacedim> &,
          const DoFHandler<dim, spacedim> &,
          const std::vector<types::global_dof_index> &,
          const std::vector<types::global_dof_index> &,
          const MappingQGenericExt<dim, spacedim> &,
          const MappingQGenericExt<dim, spacedim> &,
          const std::map<
            typename Triangulation<dim, spacedim>::cell_iterator,
            typename Triangulation<dim + 1, spacedim>::face_iterator> &,
          const std::map<
            typename Triangulation<dim, spacedim>::cell_iterator,
            typename Triangulation<dim + 1, spacedim>::face_iterator> &,
          const DetectCellNeighboringTypeMethod,
          const bool)>(fill_hmatrix_leaf_node_subrange_with_aca_plus),
        std::placeholders::_1,
        std::cref(collection_of_leaf_sets),
        std::cref(aca_config),
        std::cref(kernels),
        std::cref(kernel_factors),
        std::cref(kx_dof_to_cell_topo),
        std::cref(ky_dof_to_cell_topo),
        std::cref(bem_values),
        std::cref(kx_dof_handler),
        std::cref(ky_dof_handler),
        std::cref(kx_dof_i2e_numbering),
        std::cref(ky_dof_i2e_numbering),
        std::cref(kx_mapping),
        std::cref(ky_mapping),
        std::cref(map_from_kx_boundary_mesh_to_volume_mesh),
        std::cref(map_from_ky_boundary_mesh_to_volume_mesh),
        method_for_cell_neighboring_type,
        enable_build_symmetric_hmat),
      tbb::auto_partitioner());
  }


  /**
   * Fill the leaf sets for a collection of \hmatrices using ACA+ with respect
   * to a list of kernel functions.
   *
   * In the meantime, the FEM mass matrix multiplied by a factor will be added
   * to the near field matrix block.
   *
   * \mynote{This requires the trial spaces as well as the ansatz spaces of
   * these \hmatrices to be the same. Therefore, the hierarchical structures for
   * all the \hmatrices should be the same, which should have been built with
   * respect to a same \bct. The memory for the matrices in the leaf set should
   * have been allocated beforehand.
   *
   * For the near field matrices in the leaf set, full matrices will be built
   * whose elements will be obtained from the evaluation of the double integral
   * in Galerkin-BEM. For the far field admissible matrices in the leaf set,
   * rank-k matrices will be built using ACA+.
   *
   * As regards the leaf sets for the collection of \hmatrices, they are stored
   * in a vector of vectors, i.e.
   * @p std::vector<std::vector<HMatrix<spacedim, RangeNumberType> *> *>. For
   * the outer vector, its size is equal to the number of kernels or \hmatrices
   * to be built. For the inner vector, it is the leaf set related to each
   * \hmatrix.}
   *
   * @param thread_num
   * @param hmats
   * @param aca_config
   * @param kernels
   * @param kernel_factors
   * @param mass_matrix_factors
   * @param kx_dof_to_cell_topo
   * @param ky_dof_to_cell_topo
   * @param bem_values
   * @param mass_matrix_quadrature_formula
   * @param kx_dof_handler
   * @param ky_dof_handler
   * @param kx_mapping
   * @param ky_mapping
   * @param map_from_kx_boundary_mesh_to_volume_mesh
   * @param map_from_ky_boundary_mesh_to_volume_mesh
   * @param method_for_cell_neighboring_type
   * @param enable_build_symmetric_hmat
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  fill_hmatrix_with_aca_plus_smp(
    const unsigned int                                 thread_num,
    std::vector<HMatrix<spacedim, RangeNumberType> *> &hmats,
    const ACAConfig &                                  aca_config,
    const std::vector<KernelFunction<spacedim> *> &    kernels,
    const std::vector<RangeNumberType> &               kernel_factors,
    const std::vector<RangeNumberType> &               mass_matrix_factors,
    const std::vector<std::vector<unsigned int>> &     kx_dof_to_cell_topo,
    const std::vector<std::vector<unsigned int>> &     ky_dof_to_cell_topo,
    const SauterQuadratureRule<dim> &                  sauter_quad_rule,
    const QGauss<dim> &                         mass_matrix_quadrature_formula,
    const DoFHandler<dim, spacedim> &           kx_dof_handler,
    const DoFHandler<dim, spacedim> &           ky_dof_handler,
    const std::vector<types::global_dof_index> &kx_dof_i2e_numbering,
    const std::vector<types::global_dof_index> &ky_dof_i2e_numbering,
    const MappingQGenericExt<dim, spacedim> &   kx_mapping,
    const MappingQGenericExt<dim, spacedim> &   ky_mapping,
    typename MappingQGeneric<dim, spacedim>::InternalData &kx_mapping_data,
    typename MappingQGeneric<dim, spacedim>::InternalData &ky_mapping_data,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_kx_boundary_mesh_to_volume_mesh,
    const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                   typename Triangulation<dim + 1, spacedim>::face_iterator>
      &map_from_ky_boundary_mesh_to_volume_mesh,
    const DetectCellNeighboringTypeMethod method_for_cell_neighboring_type,
    const bool                            enable_build_symmetric_hmat = false)
  {
    AssertDimension(hmats.size(), kernels.size());

    const unsigned int kernel_num = kernels.size();

    /**
     * Precalculate data tables for shape values at quadrature points.
     *
     * \mynote{Precalculate shape function values and their gradient values
     * at each quadrature point. N.B.
     * 1. The data tables for shape function values and their gradient
     * values should be calculated for both function space on \f$K_x\f$ and
     * function space on \f$K_y\f$.
     * 2. Being different from the integral in FEM, the integral in BEM
     * handled by Sauter's quadrature rule has multiple parts of \f$k_3\f$
     * (except the regular cell neighboring type), each of which should be
     * evaluated at a different set of quadrature points in the unit cell
     * after coordinate transformation from the parametric space. Therefore,
     * a dimension with respect to \f$k_3\f$ term index should be added to
     * the data table compared to the usual FEValues and this brings about
     * the class @p BEMValues.}
     */
    BEMValues<dim, spacedim, RangeNumberType> bem_values(
      kx_dof_handler.get_fe(),
      ky_dof_handler.get_fe(),
      kx_mapping_data,
      ky_mapping_data,
      sauter_quad_rule.quad_rule_for_same_panel,
      sauter_quad_rule.quad_rule_for_common_edge,
      sauter_quad_rule.quad_rule_for_common_vertex,
      sauter_quad_rule.quad_rule_for_regular);

    bem_values.fill_shape_function_value_tables();

    /**
     * Extract the collection of leaf sets from the vector of \hmatrices.
     */
    std::vector<std::vector<HMatrix<spacedim, RangeNumberType> *> *>
      collection_of_leaf_sets(kernel_num);
    for (unsigned int k = 0; k < kernel_num; k++)
      {
        collection_of_leaf_sets[k] = &(hmats[k]->get_leaf_set());
      }

    /**
     * Estimate the grain size by evenly distributing the leaf sets to threads.
     */
    const size_t leaf_set_size = collection_of_leaf_sets[0]->size();
    unsigned int grain_size    = leaf_set_size / thread_num;
    if (grain_size == 0)
      {
        grain_size = 1;
      }

    /**
     * \mynote{N.B. There are two places where argument passing for
     * parallelization is involved: 1. passing to @p std::bind; 2. passing to
     * the working function.
     *
     * Local variables captured by @p std::bind are by default pass-by-value.
     * Hence, for capturing large objects, pass-by-reference should be adopted
     * instead of pass-by-value, which is realized by adding the prefix
     * @p std::cref for const reference or @p std::ref for mutable reference.
     *
     * @p PairCellWiseScratchData and @p PairCellWisePerTaskData should be
     * captured by @p std::bind via pass-by-value, then for each working thread,
     * a copy of them will automatically be made.
     *
     * Meanwhile, in a same working thread, the working function will be
     * called for each object specified by the index or iterator in
     * the blocked range, during which @p PairCellWiseScratchData and
     * @p PairCellWisePerTaskData can be reused. Therefore, they should be
     * passed into the working function by non-const reference.
     *
     * For other variables to be captured like @p bem_values,
     * @p kx/ky_dof_handler, * @p kx/ky_mapping and
     * @p map_from_kx/ky_bvoundary_mesh_to_volume_mesh, since they will be not
     * modified at all, they are captured by @p std::bind via const reference
     * and then passed to the working function by const reference.}
     */

    /**
     * Here we use @p size_t index to refer to the \hmatrix in the leaf set
     * and the kernel function in the provided list @p kernels.
     */
    tbb::parallel_for(
      tbb::blocked_range<size_t>(0, leaf_set_size, grain_size),
      std::bind(
        static_cast<void (*)(
          const tbb::blocked_range<size_t> &,
          const std::vector<std::vector<HMatrix<spacedim, RangeNumberType> *> *>
            &,
          const ACAConfig &,
          const std::vector<KernelFunction<spacedim> *> &,
          const std::vector<RangeNumberType> &,
          const std::vector<RangeNumberType> &,
          const std::vector<std::vector<unsigned int>> &,
          const std::vector<std::vector<unsigned int>> &,
          const BEMValues<dim, spacedim, RangeNumberType> &,
          const QGauss<dim> &,
          const DoFHandler<dim, spacedim> &,
          const DoFHandler<dim, spacedim> &,
          const std::vector<types::global_dof_index> &,
          const std::vector<types::global_dof_index> &,
          const MappingQGenericExt<dim, spacedim> &,
          const MappingQGenericExt<dim, spacedim> &,
          const std::map<
            typename Triangulation<dim, spacedim>::cell_iterator,
            typename Triangulation<dim + 1, spacedim>::face_iterator> &,
          const std::map<
            typename Triangulation<dim, spacedim>::cell_iterator,
            typename Triangulation<dim + 1, spacedim>::face_iterator> &,
          const DetectCellNeighboringTypeMethod,
          const bool)>(fill_hmatrix_leaf_node_subrange_with_aca_plus),
        std::placeholders::_1,
        std::cref(collection_of_leaf_sets),
        std::cref(aca_config),
        std::cref(kernels),
        std::cref(kernel_factors),
        std::cref(mass_matrix_factors),
        std::cref(kx_dof_to_cell_topo),
        std::cref(ky_dof_to_cell_topo),
        std::cref(bem_values),
        std::cref(mass_matrix_quadrature_formula),
        std::cref(kx_dof_handler),
        std::cref(ky_dof_handler),
        std::cref(kx_dof_i2e_numbering),
        std::cref(ky_dof_i2e_numbering),
        std::cref(kx_mapping),
        std::cref(ky_mapping),
        std::cref(map_from_kx_boundary_mesh_to_volume_mesh),
        std::cref(map_from_ky_boundary_mesh_to_volume_mesh),
        method_for_cell_neighboring_type,
        enable_build_symmetric_hmat),
      tbb::auto_partitioner());
  }
} // namespace IdeoBEM

#endif /* INCLUDE_ACA_PLUS_H_ */
