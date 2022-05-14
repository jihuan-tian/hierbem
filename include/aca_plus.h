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
    PairCellWiseScratchData &                        scratch,
    PairCellWisePerTaskData &                        data,
    const KernelFunction<spacedim> &                 kernel,
    const types::global_dof_index                    row_dof_index,
    const std::vector<types::global_dof_index> &     column_dof_indices,
    const std::vector<std::vector<unsigned int>> &   dof_to_cell_topo,
    const BEMValues<dim, spacedim, RangeNumberType> &bem_values,
    const DoFHandler<dim, spacedim> &                kx_dof_handler,
    const DoFHandler<dim, spacedim> &                ky_dof_handler,
    const MappingQGeneric<dim, spacedim> &           kx_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const MappingQGeneric<dim, spacedim> &ky_mapping =
      MappingQGeneric<dim, spacedim>(1))
  {
    AssertDimension(row_vector.size(), column_dof_indices.size());

    /**
     * Iterate over each column DoF index.
     */
    for (size_type j = 0; j < column_dof_indices.size(); j++)
      {
        row_vector(j) =
          sauter_assemble_on_one_pair_of_dofs(scratch,
                                              data,
                                              kernel,
                                              row_dof_index,
                                              column_dof_indices[j],
                                              dof_to_cell_topo,
                                              bem_values,
                                              kx_dof_handler,
                                              ky_dof_handler,
                                              kx_mapping,
                                              ky_mapping);
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
    PairCellWiseScratchData &                        scratch,
    PairCellWisePerTaskData &                        data,
    const KernelFunction<spacedim> &                 kernel,
    const std::vector<types::global_dof_index> &     row_dof_indices,
    const types::global_dof_index                    col_dof_index,
    const std::vector<std::vector<unsigned int>> &   dof_to_cell_topo,
    const BEMValues<dim, spacedim, RangeNumberType> &bem_values,
    const DoFHandler<dim, spacedim> &                kx_dof_handler,
    const DoFHandler<dim, spacedim> &                ky_dof_handler,
    const MappingQGeneric<dim, spacedim> &           kx_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const MappingQGeneric<dim, spacedim> &ky_mapping =
      MappingQGeneric<dim, spacedim>(1))
  {
    AssertDimension(col_vector.size(), row_dof_indices.size());

    /**
     * Iterate over each row DoF index.
     */
    for (size_type i = 0; i < row_dof_indices.size(); i++)
      {
        col_vector(i) = sauter_assemble_on_one_pair_of_dofs(scratch,
                                                            data,
                                                            kernel,
                                                            row_dof_indices[i],
                                                            col_dof_index,
                                                            dof_to_cell_topo,
                                                            bem_values,
                                                            kx_dof_handler,
                                                            ky_dof_handler,
                                                            kx_mapping,
                                                            ky_mapping);
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
    PairCellWiseScratchData &                        scratch,
    PairCellWisePerTaskData &                        data,
    const KernelFunction<spacedim> &                 kernel,
    std::forward_list<size_type> &                   remaining_row_indices,
    const size_type                                  current_ref_row_index,
    const std::vector<types::global_dof_index> &     row_dof_indices,
    const std::vector<types::global_dof_index> &     col_dof_indices,
    const std::vector<std::vector<unsigned int>> &   dof_to_cell_topo,
    const BEMValues<dim, spacedim, RangeNumberType> &bem_values,
    const DoFHandler<dim, spacedim> &                kx_dof_handler,
    const DoFHandler<dim, spacedim> &                ky_dof_handler,
    const MappingQGeneric<dim, spacedim> &           kx_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const MappingQGeneric<dim, spacedim> &ky_mapping =
      MappingQGeneric<dim, spacedim>(1))
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
                                scratch,
                                data,
                                kernel,
                                next_ref_row_dof_index,
                                col_dof_indices,
                                dof_to_cell_topo,
                                bem_values,
                                kx_dof_handler,
                                ky_dof_handler,
                                kx_mapping,
                                ky_mapping);

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
                Assert(
                  false,
                  ExcMessage(
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

    Assert(false,
           ExcMessage("There are no remaining row indices to select from!"));
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
                Assert(
                  false,
                  ExcMessage(
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

    Assert(false,
           ExcMessage("There are no remaining row indices to select from!"));
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
    PairCellWiseScratchData &                        scratch,
    PairCellWisePerTaskData &                        data,
    const KernelFunction<spacedim> &                 kernel,
    std::forward_list<size_type> &                   remaining_col_indices,
    const size_type                                  current_ref_col_index,
    const std::vector<types::global_dof_index> &     row_dof_indices,
    const std::vector<types::global_dof_index> &     col_dof_indices,
    const std::vector<std::vector<unsigned int>> &   dof_to_cell_topo,
    const BEMValues<dim, spacedim, RangeNumberType> &bem_values,
    const DoFHandler<dim, spacedim> &                kx_dof_handler,
    const DoFHandler<dim, spacedim> &                ky_dof_handler,
    const MappingQGeneric<dim, spacedim> &           kx_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const MappingQGeneric<dim, spacedim> &ky_mapping =
      MappingQGeneric<dim, spacedim>(1))
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
                                   scratch,
                                   data,
                                   kernel,
                                   row_dof_indices,
                                   next_ref_col_dof_index,
                                   dof_to_cell_topo,
                                   bem_values,
                                   kx_dof_handler,
                                   ky_dof_handler,
                                   kx_mapping,
                                   ky_mapping);

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
                Assert(
                  false,
                  ExcMessage(
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

    Assert(false,
           ExcMessage("There are no remaining column indices to select from!"));
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
                Assert(
                  false,
                  ExcMessage(
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

    Assert(false,
           ExcMessage("There are no remaining column indices to select from!"));
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
   * @param scratch
   * @param data
   * @param aca_config
   * @param kernel
   * @param row_dof_indices
   * @param col_dof_indices
   * @param dof_to_cell_topo
   * @param bem_values
   * @param kx_dof_handler
   * @param ky_dof_handler
   * @param kx_mapping
   * @param ky_mapping
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  aca_plus(RkMatrix<RangeNumberType> &                      rkmat,
           PairCellWiseScratchData &                        scratch,
           PairCellWisePerTaskData &                        data,
           const ACAConfig &                                aca_config,
           const KernelFunction<spacedim> &                 kernel,
           const std::vector<types::global_dof_index> &     row_dof_indices,
           const std::vector<types::global_dof_index> &     col_dof_indices,
           const std::vector<std::vector<unsigned int>> &   dof_to_cell_topo,
           const BEMValues<dim, spacedim, RangeNumberType> &bem_values,
           const DoFHandler<dim, spacedim> &                kx_dof_handler,
           const DoFHandler<dim, spacedim> &                ky_dof_handler,
           const MappingQGeneric<dim, spacedim> &           kx_mapping =
             MappingQGeneric<dim, spacedim>(1),
           const MappingQGeneric<dim, spacedim> &ky_mapping =
             MappingQGeneric<dim, spacedim>(1))
  {
    AssertDimension(rkmat.get_m(), row_dof_indices.size());
    AssertDimension(rkmat.get_n(), col_dof_indices.size());

    /**
     * Get the size of each dimension of the matrix block to be built.
     */
    const size_type m = row_dof_indices.size();
    const size_type n = col_dof_indices.size();

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
    size_type               r = random_select_ref_row(vr,
                                        scratch,
                                        data,
                                        kernel,
                                        remaining_row_indices,
                                        m + 1,
                                        row_dof_indices,
                                        col_dof_indices,
                                        dof_to_cell_topo,
                                        bem_values,
                                        kx_dof_handler,
                                        ky_dof_handler,
                                        kx_mapping,
                                        ky_mapping);
    size_type               c = random_select_ref_column(uc,
                                           scratch,
                                           data,
                                           kernel,
                                           remaining_col_indices,
                                           n + 1,
                                           row_dof_indices,
                                           col_dof_indices,
                                           dof_to_cell_topo,
                                           bem_values,
                                           kx_dof_handler,
                                           ky_dof_handler,
                                           kx_mapping,
                                           ky_mapping);

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
            assemble_kernel_row(vk,
                                scratch,
                                data,
                                kernel,
                                row_dof_indices[ik],
                                col_dof_indices,
                                dof_to_cell_topo,
                                bem_values,
                                kx_dof_handler,
                                ky_dof_handler,
                                kx_mapping,
                                ky_mapping);

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
                                   scratch,
                                   data,
                                   kernel,
                                   row_dof_indices,
                                   col_dof_indices[jk],
                                   dof_to_cell_topo,
                                   bem_values,
                                   kx_dof_handler,
                                   ky_dof_handler,
                                   kx_mapping,
                                   ky_mapping);

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
                                   scratch,
                                   data,
                                   kernel,
                                   row_dof_indices,
                                   col_dof_indices[jk],
                                   dof_to_cell_topo,
                                   bem_values,
                                   kx_dof_handler,
                                   ky_dof_handler,
                                   kx_mapping,
                                   ky_mapping);

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
                                scratch,
                                data,
                                kernel,
                                row_dof_indices[ik],
                                col_dof_indices,
                                dof_to_cell_topo,
                                bem_values,
                                kx_dof_handler,
                                ky_dof_handler,
                                kx_mapping,
                                ky_mapping);

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
                                      scratch,
                                      data,
                                      kernel,
                                      remaining_row_indices,
                                      r,
                                      row_dof_indices,
                                      col_dof_indices,
                                      dof_to_cell_topo,
                                      bem_values,
                                      kx_dof_handler,
                                      ky_dof_handler,
                                      kx_mapping,
                                      ky_mapping);

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
            c = random_select_ref_column(uc,
                                         scratch,
                                         data,
                                         kernel,
                                         remaining_col_indices,
                                         c,
                                         row_dof_indices,
                                         col_dof_indices,
                                         dof_to_cell_topo,
                                         bem_values,
                                         kx_dof_handler,
                                         ky_dof_handler,
                                         kx_mapping,
                                         ky_mapping);

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
                Assert(
                  false,
                  ExcMessage(
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
   * @param scratch
   * @param data
   * @param aca_config
   * @param kernel
   * @param dof_to_cell_topo
   * @param bem_values
   * @param kx_dof_handler
   * @param ky_dof_handler
   * @param kx_mapping
   * @param ky_mapping
   * @param enable_build_symmetric_hmat Flag indicating whether symmetric
   * \hmatrix will be built when the kernel function is symmetric.
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  fill_hmatrix_leaf_node_with_aca_plus(
    HMatrix<spacedim, RangeNumberType> *             leaf_mat,
    PairCellWiseScratchData &                        scratch,
    PairCellWisePerTaskData &                        data,
    const ACAConfig &                                aca_config,
    const KernelFunction<spacedim> &                 kernel,
    const std::vector<std::vector<unsigned int>> &   dof_to_cell_topo,
    const BEMValues<dim, spacedim, RangeNumberType> &bem_values,
    const DoFHandler<dim, spacedim> &                kx_dof_handler,
    const DoFHandler<dim, spacedim> &                ky_dof_handler,
    const MappingQGeneric<dim, spacedim> &           kx_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const MappingQGeneric<dim, spacedim> &ky_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const bool enable_build_symmetric_hmat = false)
  {
    const std::vector<types::global_dof_index> *row_dof_indices =
      leaf_mat->get_row_indices();
    const std::vector<types::global_dof_index> *col_dof_indices =
      leaf_mat->get_col_indices();
    const std::map<types::global_dof_index, size_t>
      &row_index_global_to_local_map =
        leaf_mat->get_row_index_global_to_local_map();
    const std::map<types::global_dof_index, size_t>
      &col_index_global_to_local_map =
        leaf_mat->get_col_index_global_to_local_map();

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
                        for (size_t i = 0; i < row_dof_indices->size(); i++)
                          {
                            for (size_t j = 0; j <= i; j++)
                              {
                                (*fullmat)(row_index_global_to_local_map.at(
                                             (*row_dof_indices)[i]),
                                           col_index_global_to_local_map.at(
                                             (*col_dof_indices)[j])) =
                                  sauter_assemble_on_one_pair_of_dofs(
                                    scratch,
                                    data,
                                    kernel,
                                    (*row_dof_indices)[i],
                                    (*col_dof_indices)[j],
                                    dof_to_cell_topo,
                                    bem_values,
                                    kx_dof_handler,
                                    ky_dof_handler,
                                    kx_mapping,
                                    ky_mapping);
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
                        for (types::global_dof_index row_dof_index :
                             (*row_dof_indices))
                          {
                            for (types::global_dof_index col_dof_index :
                                 (*col_dof_indices))
                              {
                                (*fullmat)(row_index_global_to_local_map.at(
                                             row_dof_index),
                                           col_index_global_to_local_map.at(
                                             col_dof_index)) =
                                  sauter_assemble_on_one_pair_of_dofs(
                                    scratch,
                                    data,
                                    kernel,
                                    row_dof_index,
                                    col_dof_index,
                                    dof_to_cell_topo,
                                    bem_values,
                                    kx_dof_handler,
                                    ky_dof_handler,
                                    kx_mapping,
                                    ky_mapping);
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
                for (types::global_dof_index row_dof_index : (*row_dof_indices))
                  {
                    for (types::global_dof_index col_dof_index :
                         (*col_dof_indices))
                      {
                        (*fullmat)(
                          row_index_global_to_local_map.at(row_dof_index),
                          col_index_global_to_local_map.at(col_dof_index)) =
                          sauter_assemble_on_one_pair_of_dofs(scratch,
                                                              data,
                                                              kernel,
                                                              row_dof_index,
                                                              col_dof_index,
                                                              dof_to_cell_topo,
                                                              bem_values,
                                                              kx_dof_handler,
                                                              ky_dof_handler,
                                                              kx_mapping,
                                                              ky_mapping);
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
                                 scratch,
                                 data,
                                 aca_config,
                                 kernel,
                                 (*row_dof_indices),
                                 (*col_dof_indices),
                                 dof_to_cell_topo,
                                 bem_values,
                                 kx_dof_handler,
                                 ky_dof_handler,
                                 kx_mapping,
                                 ky_mapping);

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
                         scratch,
                         data,
                         aca_config,
                         kernel,
                         (*row_dof_indices),
                         (*col_dof_indices),
                         dof_to_cell_topo,
                         bem_values,
                         kx_dof_handler,
                         ky_dof_handler,
                         kx_mapping,
                         ky_mapping);
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
   * kernel functions using ACA+. If the matrix type is @p RkMatrixType, the
   * memory for the full or rank-k matrix in the leaf node has been allocated.
   * This version is applied to a list of kernels.
   *
   * For the near field matrix, full matrices will be built whose elements will
   * be obtained from the evaluation of the double integral in Galerkin-BEM. For
   * the far field admissible matrix, rank-k matrices will be built using ACA+.
   *
   * \mynote{This is used as the work function for parallel \hmatrix
   * construction using ACA+.}
   *
   * @param leaf_mat_for_kernels A vector of leaf \hmatrix pointers
   * @param scratch
   * @param data
   * @param aca_config
   * @param kernels A vector of kernel function pointers
   * @param dof_to_cell_topo
   * @param bem_values
   * @param kx_dof_handler
   * @param ky_dof_handler
   * @param kx_mapping
   * @param ky_mapping
   * @param enable_build_symmetric_hmat Flag indicating whether symmetric
   * \hmatrix will be built when the kernel function is symmetric.
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  fill_hmatrix_leaf_node_with_aca_plus(
    std::vector<HMatrix<spacedim, RangeNumberType> *> leaf_mat_for_kernels,
    PairCellWiseScratchData &                         scratch,
    PairCellWisePerTaskData &                         data,
    const ACAConfig &                                 aca_config,
    const std::vector<KernelFunction<spacedim> *> &   kernels,
    const std::vector<std::vector<unsigned int>> &    dof_to_cell_topo,
    const BEMValues<dim, spacedim, RangeNumberType> & bem_values,
    const DoFHandler<dim, spacedim> &                 kx_dof_handler,
    const DoFHandler<dim, spacedim> &                 ky_dof_handler,
    const MappingQGeneric<dim, spacedim> &            kx_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const MappingQGeneric<dim, spacedim> &ky_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const bool enable_build_symmetric_hmat = false)
  {
    AssertDimension(leaf_mat_for_kernels.size(), kernels.size());

    const unsigned int kernel_num = kernels.size();

    const std::vector<types::global_dof_index> *row_dof_indices =
      leaf_mat_for_kernels[0]->get_row_indices();
    const std::vector<types::global_dof_index> *col_dof_indices =
      leaf_mat_for_kernels[0]->get_col_indices();
    const std::map<types::global_dof_index, size_t>
      &row_index_global_to_local_map =
        leaf_mat_for_kernels[0]->get_row_index_global_to_local_map();
    const std::map<types::global_dof_index, size_t>
      &col_index_global_to_local_map =
        leaf_mat_for_kernels[0]->get_col_index_global_to_local_map();

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
            for (size_t i = 0; i < row_dof_indices->size(); i++)
              {
                /**
                 * Iterate over each DoF index \f$j\f$ in the cluster
                 * \$\sigma\f$.
                 */
                for (size_t j = 0; j < col_dof_indices->size(); j++)
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
                      fullmat_coeffs,
                      scratch,
                      data,
                      kernels,
                      enable_kernel_evaluations,
                      (*row_dof_indices)[i],
                      (*col_dof_indices)[j],
                      dof_to_cell_topo,
                      bem_values,
                      kx_dof_handler,
                      ky_dof_handler,
                      kx_mapping,
                      ky_mapping);

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
                            (*fullmat)(row_index_global_to_local_map.at(
                                         (*row_dof_indices)[i]),
                                       col_index_global_to_local_map.at(
                                         (*col_dof_indices)[j])) =
                              fullmat_coeffs(k);
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
                             scratch,
                             data,
                             aca_config,
                             *kernel,
                             (*row_dof_indices),
                             (*col_dof_indices),
                             dof_to_cell_topo,
                             bem_values,
                             kx_dof_handler,
                             ky_dof_handler,
                             kx_mapping,
                             ky_mapping);
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
   * kernel functions using ACA+. In the meantime, the FEM mass matrix
   * multiplied by a factor will be added to the near field matrix block. If
   * the matrix type is @p RkMatrixType, the memory for the full or rank-k
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
   * @param scratch
   * @param data
   * @param mass_matrix_factors
   * @param aca_config
   * @param kernels
   * @param dof_to_cell_topo
   * @param bem_values
   * @param kx_dof_handler
   * @param ky_dof_handler
   * @param kx_mapping
   * @param ky_mapping
   * @param enable_build_symmetric_hmat Flag indicating whether symmetric
   * \hmatrix will be built when the kernel function is symmetric.
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  fill_hmatrix_leaf_node_with_aca_plus(
    std::vector<HMatrix<spacedim, RangeNumberType> *> leaf_mat_for_kernels,
    PairCellWiseScratchData &                         scratch,
    PairCellWisePerTaskData &                         data,
    CellWiseScratchData &                             fem_scratch,
    const std::vector<RangeNumberType> &              mass_matrix_factors,
    const ACAConfig &                                 aca_config,
    const std::vector<KernelFunction<spacedim> *> &   kernels,
    const std::vector<std::vector<unsigned int>> &    dof_to_cell_topo,
    const BEMValues<dim, spacedim, RangeNumberType> & bem_values,
    const DoFHandler<dim, spacedim> &                 kx_dof_handler,
    const DoFHandler<dim, spacedim> &                 ky_dof_handler,
    const MappingQGeneric<dim, spacedim> &            kx_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const MappingQGeneric<dim, spacedim> &ky_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const bool enable_build_symmetric_hmat = false)
  {
    AssertDimension(leaf_mat_for_kernels.size(), kernels.size());

    const unsigned int kernel_num = kernels.size();

    const std::vector<types::global_dof_index> *row_dof_indices =
      leaf_mat_for_kernels[0]->get_row_indices();
    const std::vector<types::global_dof_index> *col_dof_indices =
      leaf_mat_for_kernels[0]->get_col_indices();
    const std::map<types::global_dof_index, size_t>
      &row_index_global_to_local_map =
        leaf_mat_for_kernels[0]->get_row_index_global_to_local_map();
    const std::map<types::global_dof_index, size_t>
      &col_index_global_to_local_map =
        leaf_mat_for_kernels[0]->get_col_index_global_to_local_map();

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
            for (size_t i = 0; i < row_dof_indices->size(); i++)
              {
                /**
                 * Iterate over each DoF index \f$j\f$ in the cluster
                 * \$\sigma\f$.
                 */
                for (size_t j = 0; j < col_dof_indices->size(); j++)
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
                      fullmat_coeffs,
                      scratch,
                      data,
                      fem_scratch,
                      mass_matrix_factors,
                      kernels,
                      enable_kernel_evaluations,
                      (*row_dof_indices)[i],
                      (*col_dof_indices)[j],
                      dof_to_cell_topo,
                      bem_values,
                      kx_dof_handler,
                      ky_dof_handler,
                      kx_mapping,
                      ky_mapping);

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
                            (*fullmat)(row_index_global_to_local_map.at(
                                         (*row_dof_indices)[i]),
                                       col_index_global_to_local_map.at(
                                         (*col_dof_indices)[j])) =
                              fullmat_coeffs(k);
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
                             scratch,
                             data,
                             aca_config,
                             *kernel,
                             (*row_dof_indices),
                             (*col_dof_indices),
                             dof_to_cell_topo,
                             bem_values,
                             kx_dof_handler,
                             ky_dof_handler,
                             kx_mapping,
                             ky_mapping);
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
   * @param scratch
   * @param data
   * @param aca_config
   * @param kernel
   * @param dof_to_cell_topo
   * @param bem_values
   * @param kx_dof_handler
   * @param ky_dof_handler
   * @param kx_mapping
   * @param ky_mapping
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
    const std::vector<std::vector<unsigned int>> &   dof_to_cell_topo,
    const BEMValues<dim, spacedim, RangeNumberType> &bem_values,
    const DoFHandler<dim, spacedim> &                kx_dof_handler,
    const DoFHandler<dim, spacedim> &                ky_dof_handler,
    const MappingQGeneric<dim, spacedim> &           kx_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const MappingQGeneric<dim, spacedim> &ky_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const bool enable_build_symmetric_hmat = false)
  {
    /**
     * Define @p PairCellWiseScratchData and @p PairCellWisePerTaskData which
     * are local to the current working thread. This is mandatory because
     * each current working thread should have its own copy of these data.
     */
    PairCellWiseScratchData scratch_data(kx_dof_handler.get_fe(),
                                         ky_dof_handler.get_fe(),
                                         bem_values);
    PairCellWisePerTaskData per_task_data(kx_dof_handler.get_fe(),
                                          ky_dof_handler.get_fe());

    for (typename std::vector<HMatrix<spacedim, RangeNumberType> *>::iterator
           iter = range.begin();
         iter != range.end();
         iter++)
      {
        fill_hmatrix_leaf_node_with_aca_plus((*iter),
                                             scratch_data,
                                             per_task_data,
                                             aca_config,
                                             kernel,
                                             dof_to_cell_topo,
                                             bem_values,
                                             kx_dof_handler,
                                             ky_dof_handler,
                                             kx_mapping,
                                             ky_mapping,
                                             enable_build_symmetric_hmat);
      }
  }


  /**
   * Fill the leaf nodes in a subrange for a list of \hmatrices, each of which
   * corresponds to a kernel function in the list @p kernels using ACA+.
   *
   * \mynote{This function is to be used for TBB parallelization.}
   *
   * @param range
   * @param collection_of_leaf_sets A vector of pointers to the leaf sets of a
   * list of \hmatrices.
   * @param aca_config
   * @param kernels A vector of kernel function corresponding to the list of
   * \hmatrices.
   * @param dof_to_cell_topo
   * @param bem_values
   * @param kx_dof_handler
   * @param ky_dof_handler
   * @param kx_mapping
   * @param ky_mapping
   * @param enable_build_symmetric_hmat Flag indicating whether symmetric
   * \hmatrix will be built when the kernel function is symmetric.
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  fill_hmatrix_leaf_node_subrange_with_aca_plus_for_kernel_list(
    const tbb::blocked_range<size_t> &range,
    const std::vector<std::vector<HMatrix<spacedim, RangeNumberType> *> *>
      &                                              collection_of_leaf_sets,
    const ACAConfig &                                aca_config,
    const std::vector<KernelFunction<spacedim> *> &  kernels,
    const std::vector<std::vector<unsigned int>> &   dof_to_cell_topo,
    const BEMValues<dim, spacedim, RangeNumberType> &bem_values,
    const DoFHandler<dim, spacedim> &                kx_dof_handler,
    const DoFHandler<dim, spacedim> &                ky_dof_handler,
    const MappingQGeneric<dim, spacedim> &           kx_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const MappingQGeneric<dim, spacedim> &ky_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const bool enable_build_symmetric_hmat = false)
  {
    AssertDimension(collection_of_leaf_sets.size(), kernels.size());

    const unsigned int kernel_num = kernels.size();

    /**
     * Define @p PairCellWiseScratchData and @p PairCellWisePerTaskData which
     * are local to the current working thread. This is mandatory because
     * each current working thread should have its own copy of these data.
     */
    PairCellWiseScratchData scratch_data(kx_dof_handler.get_fe(),
                                         ky_dof_handler.get_fe(),
                                         bem_values);
    PairCellWisePerTaskData per_task_data(kx_dof_handler.get_fe(),
                                          ky_dof_handler.get_fe());

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

        fill_hmatrix_leaf_node_with_aca_plus(leaf_mat_list,
                                             scratch_data,
                                             per_task_data,
                                             aca_config,
                                             kernels,
                                             dof_to_cell_topo,
                                             bem_values,
                                             kx_dof_handler,
                                             ky_dof_handler,
                                             kx_mapping,
                                             ky_mapping,
                                             enable_build_symmetric_hmat);
      }
  }


  /**
   * Fill the leaf nodes in a subrange for a list of \hmatrices, each of which
   * corresponds to a kernel function in the list @p kernels using ACA+. In the
   * meantime, the FEM mass matrix multiplied by a factor will be added to the
   * near field matrix block.
   *
   * \mynote{This function is to be used for TBB parallelization.}
   *
   * @param range
   * @param collection_of_leaf_sets
   * @param mass_matrix_factors
   * @param aca_config
   * @param kernels
   * @param dof_to_cell_topo
   * @param bem_values
   * @param kx_dof_handler
   * @param ky_dof_handler
   * @param kx_mapping
   * @param ky_mapping
   * @param enable_build_symmetric_hmat Flag indicating whether symmetric
   * \hmatrix will be built when the kernel function is symmetric.
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  fill_hmatrix_leaf_node_subrange_with_aca_plus_for_kernel_list(
    const tbb::blocked_range<size_t> &range,
    const std::vector<std::vector<HMatrix<spacedim, RangeNumberType> *> *>
      &                                              collection_of_leaf_sets,
    const std::vector<RangeNumberType> &             mass_matrix_factors,
    const ACAConfig &                                aca_config,
    const std::vector<KernelFunction<spacedim> *> &  kernels,
    const std::vector<std::vector<unsigned int>> &   dof_to_cell_topo,
    const BEMValues<dim, spacedim, RangeNumberType> &bem_values,
    const QGauss<dim> &                              fem_quadrature_formula,
    const DoFHandler<dim, spacedim> &                kx_dof_handler,
    const DoFHandler<dim, spacedim> &                ky_dof_handler,
    const MappingQGeneric<dim, spacedim> &           kx_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const MappingQGeneric<dim, spacedim> &ky_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const bool enable_build_symmetric_hmat = false)
  {
    AssertDimension(collection_of_leaf_sets.size(), kernels.size());

    const unsigned int kernel_num = kernels.size();

    /**
     * Define @p CellWiseScratchData which is local to the current working thread.
     */
    CellWiseScratchData fem_scratch_data(kx_dof_handler.get_fe(),
                                         fem_quadrature_formula,
                                         update_values | update_JxW_values);

    /**
     * Define @p PairCellWiseScratchData and @p PairCellWisePerTaskData which
     * are local to the current working thread. This is mandatory because
     * each current working thread should have its own copy of these data.
     */
    PairCellWiseScratchData scratch_data(kx_dof_handler.get_fe(),
                                         ky_dof_handler.get_fe(),
                                         bem_values);
    PairCellWisePerTaskData per_task_data(kx_dof_handler.get_fe(),
                                          ky_dof_handler.get_fe());

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

        fill_hmatrix_leaf_node_with_aca_plus(leaf_mat_list,
                                             scratch_data,
                                             per_task_data,
                                             fem_scratch_data,
                                             mass_matrix_factors,
                                             aca_config,
                                             kernels,
                                             dof_to_cell_topo,
                                             bem_values,
                                             kx_dof_handler,
                                             ky_dof_handler,
                                             kx_mapping,
                                             ky_mapping,
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
   * @param hmat
   * @param scratch
   * @param data
   * @param aca_config
   * @param kernel
   * @param dof_to_cell_topo
   * @param bem_values
   * @param kx_dof_handler
   * @param ky_dof_handler
   * @param kx_mapping
   * @param ky_mapping
   * @param enable_build_symmetric_hmat Flag indicating whether symmetric
   * \hmatrix will be built when the kernel function is symmetric.
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  fill_hmatrix_with_aca_plus(
    HMatrix<spacedim, RangeNumberType> &             hmat,
    PairCellWiseScratchData &                        scratch,
    PairCellWisePerTaskData &                        data,
    const ACAConfig &                                aca_config,
    const KernelFunction<spacedim> &                 kernel,
    const std::vector<std::vector<unsigned int>> &   dof_to_cell_topo,
    const BEMValues<dim, spacedim, RangeNumberType> &bem_values,
    const DoFHandler<dim, spacedim> &                kx_dof_handler,
    const DoFHandler<dim, spacedim> &                ky_dof_handler,
    const MappingQGeneric<dim, spacedim> &           kx_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const MappingQGeneric<dim, spacedim> &ky_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const bool enable_build_symmetric_hmat = false)
  {
    for (HMatrix<spacedim, RangeNumberType> *leaf_mat : hmat.get_leaf_set())
      {
        fill_hmatrix_leaf_node_with_aca_plus(leaf_mat,
                                             scratch,
                                             data,
                                             aca_config,
                                             kernel,
                                             dof_to_cell_topo,
                                             bem_values,
                                             kx_dof_handler,
                                             ky_dof_handler,
                                             kx_mapping,
                                             ky_mapping,
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
   * \mynote{This version utilizes SMP parallelization.}
   *
   * @param hmat
   * @param scratch
   * @param data
   * @param aca_config
   * @param kernel
   * @param dof_to_cell_topo
   * @param bem_values
   * @param kx_dof_handler
   * @param ky_dof_handler
   * @param kx_mapping
   * @param ky_mapping
   * @param enable_build_symmetric_hmat Flag indicating whether symmetric
   * \hmatrix will be built when the kernel function is symmetric.
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  fill_hmatrix_with_aca_plus_smp(
    const unsigned int                               thread_num,
    HMatrix<spacedim, RangeNumberType> &             hmat,
    const ACAConfig &                                aca_config,
    const KernelFunction<spacedim> &                 kernel,
    const std::vector<std::vector<unsigned int>> &   dof_to_cell_topo,
    const BEMValues<dim, spacedim, RangeNumberType> &bem_values,
    const DoFHandler<dim, spacedim> &                kx_dof_handler,
    const DoFHandler<dim, spacedim> &                ky_dof_handler,
    const MappingQGeneric<dim, spacedim> &           kx_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const MappingQGeneric<dim, spacedim> &ky_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const bool enable_build_symmetric_hmat = false)
  {
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

    //    /**
    //     * Implementation of the parallelism using C++ lambda.
    //     */
    //    auto f = [&](tbb::blocked_range<typename std::vector<
    //                   HMatrix<spacedim, RangeNumberType> *>::iterator>
    //                   &range) {
    //      /**
    //       * Define @p PairCellWiseScratchData and @p PairCellWisePerTaskData which
    //       * are local to the current working thread. This is mandatory
    //       because
    //       * each current working thread should have its own copy of these
    //       data.
    //       */
    //      PairCellWiseScratchData scratch_data(kx_dof_handler.get_fe(),
    //                                           ky_dof_handler.get_fe(),
    //                                           bem_values);
    //      PairCellWisePerTaskData per_task_data(kx_dof_handler.get_fe(),
    //                                            ky_dof_handler.get_fe());
    //
    //      /**
    //       * Iterate over each \hmatrix node in the subrange.
    //       */
    //      for (typename std::vector<HMatrix<spacedim, RangeNumberType>
    //      *>::iterator
    //             iter = range.begin();
    //           iter != range.end();
    //           iter++)
    //        {
    //          fill_hmatrix_leaf_node_with_aca_plus((*iter),
    //                                               scratch_data,
    //                                               per_task_data,
    //                                               aca_config,
    //                                               kernel,
    //                                               dof_to_cell_topo,
    //                                               bem_values,
    //                                               kx_dof_handler,
    //                                               ky_dof_handler,
    //                                               kx_mapping,
    //                                               ky_mapping);
    //        }
    //    };
    //
    //    parallel::internal::parallel_for(leaf_set.begin(),
    //                                     leaf_set.end(),
    //                                     f,
    //                                     grain_size);

    /**
     * \mynote{Local variables captured by @p std::bind are by default
     * pass-by-value. Hence, for capturing large objects,
     * pass-by-reference should be adopted, which is realized by adding the
     * prefix @p std::cref or @p std::ref depending on whether const reference
     * is required. Because @p PairCellWiseScratchData and
     * @p PairCellWisePerTaskData will be modified in each call of the work
     * function passed to @p std::bind, they should be captured by @p std::bind
     * via pass-by-value.
     *
     * Meanwhile, in a same working thread, the work function will be
     * called for each object specified by the index or iterator in
     * the blocked range, during which @p PairCellWiseScratchData and
     * @p PairCellWisePerTaskData can be reused. Therefore, they should be
     * passed into the work function by non-const reference.
     *
     * For other variables to be captured like @p bem_values, @p kx_dof_handler
     * and @p kx_mapping, since they will be not modified at all, they are
     * captured by const reference and passed to the work function by const
     * reference.}
     */
    parallel::internal::parallel_for(
      leaf_set.begin(),
      leaf_set.end(),
      std::bind(static_cast<
                  void (*)(const tbb::blocked_range<typename std::vector<
                             HMatrix<spacedim, RangeNumberType> *>::iterator> &,
                           const ACAConfig &,
                           const KernelFunction<spacedim> &,
                           const std::vector<std::vector<unsigned int>> &,
                           const BEMValues<dim, spacedim, RangeNumberType> &,
                           const DoFHandler<dim, spacedim> &,
                           const DoFHandler<dim, spacedim> &,
                           const MappingQGeneric<dim, spacedim> &,
                           const MappingQGeneric<dim, spacedim> &)>(
                  fill_hmatrix_leaf_node_subrange_with_aca_plus),
                std::placeholders::_1,
                std::cref(aca_config),
                std::cref(kernel),
                std::cref(dof_to_cell_topo),
                std::cref(bem_values),
                std::cref(kx_dof_handler),
                std::cref(ky_dof_handler),
                std::cref(kx_mapping),
                std::cref(ky_mapping),
                enable_build_symmetric_hmat),
      grain_size);
  }


  /**
   * Fill the leaf sets for a collection of \hmatrices using ACA+ with respect
   * to a list of kernel functions. The hierarchical structures for all the
   * \hmatrices should be the same, which should have been built with respect to
   * a block cluster tree. The memory for the matrices in the leaf set should
   * have been allocated.
   *
   * For the near field matrices in the leaf set, full matrices will be built
   * whose elements will be obtained from the evaluation of the double integral
   * in Galerkin-BEM. For the far field admissible matrices in the leaf set,
   * rank-k matrices will be built using ACA+.
   *
   * @param thread_num
   * @param hmats A vector of \hmatrix pointers
   * @param aca_config
   * @param kernels A vector of kernel function pointers associated with the vector of
   * \hmatrix pointers stored in @p hmats
   * @param dof_to_cell_topo
   * @param bem_values
   * @param kx_dof_handler
   * @param ky_dof_handler
   * @param kx_mapping
   * @param ky_mapping
   * @param enable_build_symmetric_hmat Flag indicating whether symmetric
   * \hmatrix will be built when the kernel function is symmetric.
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  fill_hmatrix_with_aca_plus_smp(
    const unsigned int                                 thread_num,
    std::vector<HMatrix<spacedim, RangeNumberType> *> &hmats,
    const ACAConfig &                                  aca_config,
    const std::vector<KernelFunction<spacedim> *> &    kernels,
    const std::vector<std::vector<unsigned int>> &     dof_to_cell_topo,
    const BEMValues<dim, spacedim, RangeNumberType> &  bem_values,
    const DoFHandler<dim, spacedim> &                  kx_dof_handler,
    const DoFHandler<dim, spacedim> &                  ky_dof_handler,
    const MappingQGeneric<dim, spacedim> &             kx_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const MappingQGeneric<dim, spacedim> &ky_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const bool enable_build_symmetric_hmat = false)
  {
    AssertDimension(hmats.size(), kernels.size());

    const unsigned int kernel_num = kernels.size();

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
     * \mynote{Local variables captured by @p std::bind are by default
     * <strong>pass-by-value</strong>. Hence, for capturing large objects,
     * <strong>pass-by-reference</strong> should be adopted, which is realized
     * by adding the prefix @p std::cref or @p std::ref to the variables. Which
     * one should be used depends on whether the const reference is required.
     * Because @p PairCellWiseScratchData and @p PairCellWisePerTaskData will
     * be modified in each call of the work function that passed to
     * @p std::bind, they should be captured by @p std::bind via pass-by-value.
     *
     * Meanwhile, in a same working thread, the work function will be
     * called for each object specified by the index or iterator in
     * the blocked range, during which @p PairCellWiseScratchData and
     * @p PairCellWisePerTaskData can be reused. Therefore, they should be
     * passed into the work function by non-const reference.
     *
     * For other variables to be captured like @p bem_values, @p kx_dof_handler
     * and @p kx_mapping, since they will be not modified at all, they are
     * captured by const reference by @p std::bind first and then passed to the
     * work function by const reference.}
     */
    // Here we use <code>unsigned int</code> index to refer to the \hmatrix in
    // the leaf set and the kernel function in the provided list @p kernels.
    tbb::parallel_for(
      tbb::blocked_range<size_t>(0, leaf_set_size, grain_size),
      std::bind(
        static_cast<void (*)(const tbb::blocked_range<size_t> &,
                             const std::vector<std::vector<
                               HMatrix<spacedim, RangeNumberType> *> *> &,
                             const ACAConfig &,
                             const std::vector<KernelFunction<spacedim> *> &,
                             const std::vector<std::vector<unsigned int>> &,
                             const BEMValues<dim, spacedim, RangeNumberType> &,
                             const DoFHandler<dim, spacedim> &,
                             const DoFHandler<dim, spacedim> &,
                             const MappingQGeneric<dim, spacedim> &,
                             const MappingQGeneric<dim, spacedim> &,
                             const bool)>(
          fill_hmatrix_leaf_node_subrange_with_aca_plus_for_kernel_list),
        std::placeholders::_1,
        std::cref(collection_of_leaf_sets),
        std::cref(aca_config),
        std::cref(kernels),
        std::cref(dof_to_cell_topo),
        std::cref(bem_values),
        std::cref(kx_dof_handler),
        std::cref(ky_dof_handler),
        std::cref(kx_mapping),
        std::cref(ky_mapping),
        enable_build_symmetric_hmat),
      tbb::auto_partitioner());
  }


  /**
   * Fill the leaf sets for a collection of \hmatrices using ACA+ with respect
   * to a list of kernel functions. In the meantime, the FEM mass matrix
   * multiplied by a factor will be added to the near field matrix block. The
   * hierarchical structures for all the \hmatrices should be the same, which
   * should have been built with respect to a block cluster tree. The memory for
   * the matrices in the leaf set should have been allocated.
   *
   * For the near field matrices in the leaf set, full matrices will be built
   * whose elements will be obtained from the evaluation of the double integral
   * in Galerkin-BEM. For the far field admissible matrices in the leaf set,
   * rank-k matrices will be built using ACA+.
   *
   * @param thread_num
   * @param hmats A vector of \hmatrix pointers
   * @param mass_matrix_factors A vector of mass matrix factors for the
   * \hmatrices. When a factor is zero, the addition of the FEM mass matrix to
   * the \hmatrix will be disabled.
   * @param aca_config
   * @param kernels A vector of kernel function pointers associated with the vector of
   * \hmatrix pointers stored in @p hmats
   * @param dof_to_cell_topo
   * @param bem_values
   * @param kx_dof_handler
   * @param ky_dof_handler
   * @param kx_mapping
   * @param ky_mapping
   * @param enable_build_symmetric_hmat Flag indicating whether symmetric
   * \hmatrix will be built when the kernel function is symmetric.
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  fill_hmatrix_with_aca_plus_smp(
    const unsigned int                                 thread_num,
    std::vector<HMatrix<spacedim, RangeNumberType> *> &hmats,
    const std::vector<RangeNumberType> &               mass_matrix_factors,
    const ACAConfig &                                  aca_config,
    const std::vector<KernelFunction<spacedim> *> &    kernels,
    const std::vector<std::vector<unsigned int>> &     dof_to_cell_topo,
    const BEMValues<dim, spacedim, RangeNumberType> &  bem_values,
    const QGauss<dim> &                                fem_quadrature_formula,
    const DoFHandler<dim, spacedim> &                  kx_dof_handler,
    const DoFHandler<dim, spacedim> &                  ky_dof_handler,
    const MappingQGeneric<dim, spacedim> &             kx_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const MappingQGeneric<dim, spacedim> &ky_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const bool enable_build_symmetric_hmat = false)
  {
    AssertDimension(hmats.size(), kernels.size());

    const unsigned int kernel_num = kernels.size();

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
     * \mynote{Local variables captured by @p std::bind are by default
     * <strong>pass-by-value</strong>. Hence, for capturing large objects,
     * <strong>pass-by-reference</strong> should be adopted, which is realized
     * by adding the prefix @p std::cref or @p std::ref to the variables. Which
     * one should be used depends on whether the const reference is required.
     * Because @p PairCellWiseScratchData and @p PairCellWisePerTaskData will
     * be modified in each call of the work function that passed to
     * @p std::bind, they should be captured by @p std::bind via pass-by-value.
     *
     * Meanwhile, in a same working thread, the work function will be
     * called for each object specified by the index or iterator in
     * the blocked range, during which @p PairCellWiseScratchData and
     * @p PairCellWisePerTaskData can be reused. Therefore, they should be
     * passed into the work function by non-const reference.
     *
     * For other variables to be captured like @p bem_values, @p kx_dof_handler
     * and @p kx_mapping, since they will be not modified at all, they are
     * captured by const reference by @p std::bind first and then passed to the
     * work function by const reference.}
     */
    // Here we use <code>unsigned int</code> index to refer to the \hmatrix in
    // the leaf set and the kernel function in the provided list @p kernels.
    tbb::parallel_for(
      tbb::blocked_range<size_t>(0, leaf_set_size, grain_size),
      std::bind(
        static_cast<void (*)(const tbb::blocked_range<size_t> &,
                             const std::vector<std::vector<
                               HMatrix<spacedim, RangeNumberType> *> *> &,
                             const std::vector<RangeNumberType> &,
                             const ACAConfig &,
                             const std::vector<KernelFunction<spacedim> *> &,
                             const std::vector<std::vector<unsigned int>> &,
                             const BEMValues<dim, spacedim, RangeNumberType> &,
                             const QGauss<dim> &,
                             const DoFHandler<dim, spacedim> &,
                             const DoFHandler<dim, spacedim> &,
                             const MappingQGeneric<dim, spacedim> &,
                             const MappingQGeneric<dim, spacedim> &,
                             const bool)>(
          fill_hmatrix_leaf_node_subrange_with_aca_plus_for_kernel_list),
        std::placeholders::_1,
        std::cref(collection_of_leaf_sets),
        std::cref(mass_matrix_factors),
        std::cref(aca_config),
        std::cref(kernels),
        std::cref(dof_to_cell_topo),
        std::cref(bem_values),
        std::cref(fem_quadrature_formula),
        std::cref(kx_dof_handler),
        std::cref(ky_dof_handler),
        std::cref(kx_mapping),
        std::cref(ky_mapping),
        enable_build_symmetric_hmat),
      tbb::auto_partitioner());
  }
} // namespace IdeoBEM

#endif /* INCLUDE_ACA_PLUS_H_ */
