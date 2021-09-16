/**
 * \file hmatrix.h
 * \brief Definition of hierarchical matrix.
 * \ingroup hierarchical_matrices
 * \date 2021-06-06
 * \author Jihuan Tian
 */

#ifndef INCLUDE_HMATRIX_H_
#define INCLUDE_HMATRIX_H_

#include <deal.II/base/logstream.h>

#include <deal.II/lac/full_matrix.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "block_cluster.h"
#include "block_cluster_tree.h"
#include "generic_functors.h"
#include "lapack_full_matrix_ext.h"
#include "rkmatrix.h"

using namespace dealii;

/**
 * Matrix type of an HMaxtrix, which can be full matrix in the near field,
 * rank-k matrix in the far field and hierarchical matrix which does not
 * belong to the leaf set of a block cluster tree.
 */
enum HMatrixType
{
  FullMatrixType,         //!< FullMatrixType
  RkMatrixType,           //!< RkMatrixType
  HierarchicalMatrixType, //!< HierarchicalType
  UndefinedMatrixType     //!< UndefinedMatrixType
};

DeclException1(ExcInvalidHMatrixType,
               HMatrixType,
               << "Invalid HMatrix type " << arg1);

template <int spacedim, typename Number = double>
class HMatrix
{
public:
  /**
   * Declare the type for container size.
   */
  using size_type = std::make_unsigned<types::blas_int>::type;

  template <int spacedim1, typename Number1>
  friend void
  InitHMatrixWrtBlockClusterNode(
    HMatrix<spacedim1, Number1> &hmat,
    typename BlockClusterTree<spacedim1, Number1>::node_const_pointer_type
      bc_node);

  template <int spacedim1, typename Number1>
  friend void
  InitHMatrixWrtBlockClusterNode(
    HMatrix<spacedim1, Number1> &hmat,
    typename BlockClusterTree<spacedim1, Number1>::node_const_pointer_type
                                                                 bc_node,
    const std::vector<std::pair<HMatrix<spacedim1, Number1> *,
                                HMatrix<spacedim1, Number1> *>> &Sigma_P);

  template <int spacedim1, typename Number1>
  friend void
  InitHMatrixWrtBlockClusterNode(
    HMatrix<spacedim1, Number1> &hmat,
    typename BlockClusterTree<spacedim1, Number1>::node_const_pointer_type
                                                    bc_node,
    const std::pair<HMatrix<spacedim1, Number1> *,
                    HMatrix<spacedim1, Number1> *> &hmat_pair);

  template <int spacedim1, typename Number1>
  friend void
  InitAndCreateHMatrixChildrenWithoutAlloc(
    HMatrix<spacedim1, Number1> *hmat,
    typename BlockClusterTree<spacedim1, Number1>::node_const_pointer_type
               bc_node,
    const bool is_build_index_set_global_to_local_map);

  template <int spacedim1, typename Number1>
  friend void
  InitAndCreateHMatrixChildren(
    HMatrix<spacedim1, Number1> *hmat,
    typename BlockClusterTree<spacedim1, Number1>::node_const_pointer_type
                       bc_node,
    const unsigned int fixed_rank_k,
    bool               is_build_index_set_global_to_local_map);

  template <int spacedim1, typename Number1>
  friend void
  InitAndCreateHMatrixChildren(
    HMatrix<spacedim1, Number1> *hmat,
    typename BlockClusterTree<spacedim1, Number1>::node_const_pointer_type
                                        bc_node,
    const unsigned int                  fixed_rank_k,
    const LAPACKFullMatrixExt<Number1> &M,
    bool                                is_build_index_set_global_to_local_map);

  template <int spacedim1, typename Number1>
  friend void
  InitAndCreateHMatrixChildren(
    HMatrix<spacedim1, Number1> *hmat,
    typename BlockClusterTree<spacedim1, Number1>::node_const_pointer_type
                                        bc_node,
    const unsigned int                  fixed_rank_k,
    const LAPACKFullMatrixExt<Number1> &M,
    const std::map<types::global_dof_index, size_t>
      &row_index_global_to_local_map_for_M,
    const std::map<types::global_dof_index, size_t>
      &  col_index_global_to_local_map_for_M,
    bool is_build_index_set_global_to_local_map);

  template <int spacedim1, typename Number1>
  friend void
  InitAndCreateHMatrixChildren(
    HMatrix<spacedim1, Number1> *hmat,
    typename BlockClusterTree<spacedim1, Number1>::node_const_pointer_type
                                  bc_node,
    HMatrix<spacedim1, Number1> &&H);

  template <int spacedim1, typename Number1>
  friend void
  RefineHMatrixWrtExtendedBlockClusterTree(
    HMatrix<spacedim1, Number1> *starting_hmat,
    HMatrix<spacedim1, Number1> *current_hmat);

  template <int spacedim1, typename Number1>
  friend void
  convertHMatBlockToRkMatrix(HMatrix<spacedim1, Number1> *      hmat_block,
                             const unsigned int                 fixed_rank_k,
                             const HMatrix<spacedim1, Number1> *hmat_root_block,
                             size_t *                           calling_counter,
                             const std::string &output_file_base_name);

  friend void
  build_index_set_global_to_local_map(
    const std::vector<types::global_dof_index>
      &                                        index_set_as_local_to_global_map,
    std::map<types::global_dof_index, size_t> &global_to_local_map);

  // Friend functions for \hmatrix arithmetic operations.
  template <int spacedim1, typename Number1>
  friend void
  h_rk_mmult(HMatrix<spacedim1, Number1> &M1,
             const RkMatrix<Number1> &    M2,
             RkMatrix<Number1> &          M);

  template <int spacedim1, typename Number1>
  friend void
  h_rk_mmult_for_h_h_mmult(HMatrix<spacedim1, Number1> *      M1,
                           const HMatrix<spacedim1, Number1> *M2,
                           HMatrix<spacedim1, Number1> *      M,
                           bool is_M1M2_last_in_M_Sigma_P);

  template <int spacedim1, typename Number1>
  friend void
  rk_h_mmult(const RkMatrix<Number1> &    M1,
             HMatrix<spacedim1, Number1> &M2,
             RkMatrix<Number1> &          M);

  template <int spacedim1, typename Number1>
  friend void
  rk_h_mmult_for_h_h_mmult(const HMatrix<spacedim1, Number1> *M1,
                           HMatrix<spacedim1, Number1> *      M2,
                           HMatrix<spacedim1, Number1> *      M,
                           bool is_M1M2_last_in_M_Sigma_P);

  template <int spacedim1, typename Number1>
  friend void
  h_f_mmult(HMatrix<spacedim1, Number1> &       M1,
            const LAPACKFullMatrixExt<Number1> &M2,
            LAPACKFullMatrixExt<Number1> &      M);

  template <int spacedim1, typename Number1>
  friend void
  h_f_mmult(HMatrix<spacedim1, Number1> &       M1,
            const LAPACKFullMatrixExt<Number1> &M2,
            RkMatrix<Number1> &                 M);

  template <int spacedim1, typename Number1>
  friend void
  h_f_mmult_for_h_h_mmult(HMatrix<spacedim1, Number1> *      M1,
                          const HMatrix<spacedim1, Number1> *M2,
                          HMatrix<spacedim1, Number1> *      M,
                          bool is_M1M2_last_in_M_Sigma_P);

  template <int spacedim1, typename Number1>
  friend void
  f_h_mmult(const LAPACKFullMatrixExt<Number1> &M1,
            HMatrix<spacedim1, Number1> &       M2,
            LAPACKFullMatrixExt<Number1> &      M);

  template <int spacedim1, typename Number1>
  friend void
  f_h_mmult(const LAPACKFullMatrixExt<Number1> &M1,
            HMatrix<spacedim1, Number1> &       M2,
            RkMatrix<Number1> &                 M);

  template <int spacedim1, typename Number1>
  friend void
  f_h_mmult_for_h_h_mmult(const HMatrix<spacedim1, Number1> *M1,
                          HMatrix<spacedim1, Number1> *      M2,
                          HMatrix<spacedim1, Number1> *      M,
                          bool is_M1M2_last_in_M_Sigma_P);

  template <int spacedim1, typename Number1>
  friend void
  h_h_mmult_phase1_recursion(HMatrix<spacedim1, Number1> *         M,
                             BlockClusterTree<spacedim1, Number1> &Tind);

  template <int spacedim1, typename Number1>
  friend void
  h_h_mmult_phase2(HMatrix<spacedim1, Number1> &         M,
                   BlockClusterTree<spacedim1, Number1> &target_bc_tree,
                   const unsigned int                    fixed_rank);

  template <int spacedim1, typename Number1>
  friend void
  copy_hmatrix_node(HMatrix<spacedim1, Number1> &      hmat_dst,
                    const HMatrix<spacedim1, Number1> &hmat_src);

  template <int spacedim1, typename Number1>
  friend void
  copy_hmatrix_node(HMatrix<spacedim1, Number1> & hmat_dst,
                    HMatrix<spacedim1, Number1> &&hmat_src);

  template <int spacedim1, typename Number1>
  friend void
  copy_hmatrix(HMatrix<spacedim1, Number1> &      hmat_dst,
               const HMatrix<spacedim1, Number1> &hmat_src);

  template <int spacedim1, typename Number1>
  friend void
  print_h_submatrix_accessor(std::ostream &                     out,
                             const std::string &                name,
                             const HMatrix<spacedim1, Number1> &M);

  template <int spacedim1, typename Number1>
  friend void
  print_h_h_submatrix_mmult_accessor(std::ostream &                     out,
                                     const std::string &                name1,
                                     const HMatrix<spacedim1, Number1> &M1,
                                     const std::string &                name2,
                                     const HMatrix<spacedim1, Number1> &M2);

  /**
   * Default constructor.
   */
  HMatrix();

  /**
   * Construct the hierarchical structure without data from the root node of a
   * BlockClusterTree.
   */
  HMatrix(const BlockClusterTree<spacedim, Number> &bct,
          const unsigned int                        fixed_rank_k = 1);

  /**
   * Construct the hierarchical structure without data from a TreeNode in a
   * BlockClusterTree.
   */
  HMatrix(typename BlockClusterTree<spacedim, Number>::node_const_pointer_type
                             bc_node,
          const unsigned int fixed_rank_k = 1);

  /**
   * Construct from the root node of a BlockClusterTree while copying the data
   * of a global full matrix, which is created on the complete block cluster
   * \f$I \times J\f$.
   */
  HMatrix(const BlockClusterTree<spacedim, Number> &bct,
          const LAPACKFullMatrixExt<Number> &       M,
          const unsigned int                        fixed_rank_k = 1);

  /**
   * Construct from a TreeNode in a BlockClusterTree while copying the data of a
   * global full matrix, which is created on the complete block cluster \f$I
   * \times J\f$.
   */
  HMatrix(typename BlockClusterTree<spacedim, Number>::node_const_pointer_type
                                             bc_node,
          const LAPACKFullMatrixExt<Number> &M,
          const unsigned int                 fixed_rank_k = 1);

  /**
   * Construct from a \p TreeNode in a \p BlockClusterTree while moving the data
   * from the leaf set of the \hmatrix \p H.
   *
   * @param bc_node
   * @param H
   */
  HMatrix(typename BlockClusterTree<spacedim, Number>::node_const_pointer_type
                                      bc_node,
          HMatrix<spacedim, Number> &&H);

  /**
   * Construct from the root node of a BlockClusterTree while moving the data
   * from the leaf set of the \hmatrix \p H.
   *
   * @param bct
   * @param H
   */
  HMatrix(const BlockClusterTree<spacedim, Number> &bct,
          HMatrix<spacedim, Number> &&              H);

  /**
   * Deep copy constructor.
   * @param H
   */
  HMatrix(const HMatrix<spacedim, Number> &H);

  /**
   * Shallow copy constructor.
   *
   * After the copy operation, the data in the source matrix \p H are
   * transferred to the current \hmatrix node and \p H is
   * cleared.
   *
   * @param H
   */
  HMatrix(HMatrix<spacedim, Number> &&H);

  /**
   * Reinitialize the hierarchical structure without data from the root node of
   * a BlockClusterTree.
   * @param bct
   * @param fixed_rank_k
   */
  void
  reinit(const BlockClusterTree<spacedim, Number> &bct,
         const unsigned int                        fixed_rank_k = 1);

  /**
   * Reinitialize the hierarchical structure without data from a TreeNode in a
   * BlockClusterTree.
   * @param bc_node
   * @param fixed_rank_k
   */
  void
  reinit(typename BlockClusterTree<spacedim, Number>::node_const_pointer_type
                            bc_node,
         const unsigned int fixed_rank_k = 1);

  /**
   * Assignment via shallow copy.
   *
   * @param H
   * @return
   */
  HMatrix<spacedim, Number> &
  operator=(HMatrix<spacedim, Number> &&H);

  /**
   * Assignment via deep copy.
   *
   * @param H
   * @return
   */
  HMatrix<spacedim, Number> &
  operator=(const HMatrix<spacedim, Number> &H);

  /**
   * Convert an HMatrix to a full matrix by calling the internal recursive
   * function.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>This function only has the verification purpose. In reality, a large
   * dense matrix cannot be saved as a full matrix.</dd>
   * </dl>
   * @param matrix
   */
  template <typename MatrixType>
  void
  convertToFullMatrix(MatrixType &M) const;

  /**
   * TODO: Construct from a BlockClusterTree and a Sauter quadrature
   * object/functor.
   */

  /**
   * Release the memory and status of the \hmatrix hierarchy.
   */
  void
  release();

  /**
   * Clear the whole \hmatrix hierarchy.
   */
  void
  clear();

  /**
   * Clear the current \hmatrix node.
   */
  void
  clear_hmat_node();

  /**
   * Destructor which releases the memory by recursion.
   */
  ~HMatrix();

  /**
   * Get the matrix type of the current \hmatrix node.
   * @return
   */
  HMatrixType
  get_type() const;

  /**
   * Get the number of rows of the current \hmatrix node.
   * @return
   */
  size_type
  get_m() const;

  /**
   * Get the number of columns of the current \hmatrix node.
   * @return
   */
  size_type
  get_n() const;

  /**
   * Get the pointer to the rank-k matrix of the current
   * \hmatrix node.
   * @return
   */
  RkMatrix<Number> *
  get_rkmatrix();

  /**
   * Get the pointer to the rank-k matrix of the current
   * \hmatrix node (const version).
   * @return
   */
  const RkMatrix<Number> *
  get_rkmatrix() const;

  /**
   * Get the pointer to the full matrix of the current \hmatrix
   * node.
   * @return
   */
  LAPACKFullMatrixExt<Number> *
  get_fullmatrix();

  /**
   * Get the pointer to the full matrix of the current \hmatrix
   * node (const version).
   * @return
   */
  const LAPACKFullMatrixExt<Number> *
  get_fullmatrix() const;

  /**
   * Get the reference to the vector of submatrices of the current
   * \hmatrix node.
   * @return
   */
  std::vector<HMatrix<spacedim, Number> *> &
  get_submatrices();

  /**
   * Get the reference to the vector of submatrices of the current
   * \hmatrix node (const version).
   * @return
   */
  const std::vector<HMatrix<spacedim, Number> *> &
  get_submatrices() const;

  /**
   * Print the HMatrix.
   * @param out
   * @param precision
   * @param scientific
   * @param width
   * @param zero_string
   * @param denominator
   * @param threshold
   */
  void
  print_formatted(std::ostream &     out,
                  const unsigned int precision   = 3,
                  const bool         scientific  = true,
                  const unsigned int width       = 0,
                  const char *       zero_string = " ",
                  const double       denominator = 1.,
                  const double       threshold   = 0.) const;

  void
  print_matrix_info(std::ostream &out) const;

  /**
   * Write formatted full matrix leaf node to the output stream.
   *
   * The leaf node is written in the following format:
   *
   * >
   * [list-of-indices-in-cluster-tau],[list-of-indices-in-cluster-sigma],is_near_field,rank
   *
   * For example,
   *
   * > [1 2 3 ...],[7 8 9 ...],1,1
   *
   * @param out
   * @param singular_value_threshold
   */
  void
  write_fullmatrix_leaf_node(std::ostream &out,
                             const Number  singular_value_threshold = 0.) const;

  /**
   * Write formatted rank-k matrix leaf node to the output stream.
   *
   * The leaf node is written in the following format:
   *
   * >
   * [list-of-indices-in-cluster-tau],[list-of-indices-in-cluster-sigma],is_near_field,rank
   *
   * For example,
   *
   * > [1 2 3 ...],[7 8 9 ...],1,1
   *
   * @param out
   */
  void
  write_rkmatrix_leaf_node(std::ostream &out) const;

  /**
   * Write formatted leaf set to the output stream as well as the rank of each
   * matrix block by recursion.
   *
   * Each leaf node is written in the following format:
   *
   * >
   * [list-of-indices-in-cluster-tau],[list-of-indices-in-cluster-sigma],is_near_field,rank
   *
   * For example,
   *
   * > [1 2 3 ...],[7 8 9 ...],1,1
   * @param out
   * @param singular_value_threshold
   */
  void
  write_leaf_set(std::ostream &out,
                 const Number  singular_value_threshold = 0.) const;

  /**
   * Write formatted leaf set to the output stream as well as the rank of each
   * matrix block by iteration.
   *
   * Each leaf node is written in the following format:
   *
   * >
   * [list-of-indices-in-cluster-tau],[list-of-indices-in-cluster-sigma],is_near_field,rank
   *
   * For example,
   *
   * > [1 2 3 ...],[7 8 9 ...],1,1
   * @param out
   * @param singular_value_threshold
   */
  void
  write_leaf_set_by_iteration(std::ostream &out,
                              const Number singular_value_threshold = 0.) const;

  /**
   * Truncate all rank-k matrices in the leaf set of the
   * \hmatrix to rank-k matrices with the given \p new_rank,
   * while the full matrices in the leaf set, i.e. those near-field matrices,
   * are kept intact.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>This method implements the operator \f$\mathcal{T}_{r \leftarrow
   * s}^{\mathcal{H}}\f$ in (7.5) in Hackbusch's \hmatrix
   * book.</dd>
   * </dl>
   * @param new_rank
   */
  void
  truncate_to_rank(size_type new_rank);

  /**
   * Calculate matrix-vector multiplication as \f$y = y +
   * M \cdot x\f$.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>1. The recursive algorithm for \hmatrix-vector
   * multiplication needs to collect the results from different components in
   * the leaf set and corresponding vector block in \f$x\f$. More importantly,
   * there will be a series of such results contributing to a same block in the
   * result vector \f$y\f$. Therefore, if the interface of this function is
   * designed with the parameter \p add as that in the \p vmult function of \p
   * LAPACKFullMatrix in deal.ii, in all recursive calls of \p vmult except the
   * first one, this \p add flag should be set to \p true, irrespective of the
   * original flag value passed into the first call of \p vmult. Hence, we do
   * not include the \p add flag in the \p vmult function.
   * 2. The input vectors \p x and \p y are to be accessed via global DoF
   * indices.</dd>
   * </dl>
   *
   * @param y
   * @param x
   */
  void
  vmult(Vector<Number> &y, const Vector<Number> &x) const;

  /**
   * Calculate matrix-vector multiplication as \f$y = y +
   * M \cdot x\f$.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>1. The recursive algorithm for \hmatrix-vector
   * multiplication needs to collect the results from different components in
   * the leaf set and corresponding vector block in \f$x\f$. More importantly,
   * there will be a series of such results contributing to a same block in the
   * result vector \f$y\f$. Therefore, if the interface of this function is
   * designed with the parameter \p add as that in the \p vmult function of \p
   * LAPACKFullMatrix in deal.ii, in all recursive calls of \p vmult except the
   * first one, this \p add flag should be set to \p true, irrespective of the
   * original flag value passed into the first call of \p vmult. Hence, we do
   * not include the \p add flag in the \p vmult function.
   * 2. The input vectors \p x and \p y are to be accessed via local indices
   * with the assistance of \p row_index_global_to_local_map and \p
   * col_index_global_to_local_map.</dd>
   * </dl>
   *
   * @param y
   * @param x
   */
  void
  vmult_local_vector(Vector<Number> &y,
                     const std::map<types::global_dof_index, size_t>
                       &                   y_index_global_to_local_map,
                     const Vector<Number> &x,
                     const std::map<types::global_dof_index, size_t>
                       &x_index_global_to_local_map) const;

  /**
   * Calculate matrix-vector multiplication as \f$y = y +
   * M^T \cdot x\f$, i.e. the matrix \f$M\f$ is transposed.
   *
   * Because the matrix \f$M\f$ is transposed, the roles for \p row_indices and
   * \p col_indices should be swapped. Also refer to HMatrix::vmult.
   * @param y
   * @param x
   */
  void
  Tvmult(Vector<Number> &y, const Vector<Number> &x) const;

  /**
   * Calculate matrix-vector multiplication as \f$y = y +
   * M^T \cdot x\f$, i.e. the matrix \f$M\f$ is transposed.
   *
   * Because the matrix \f$M\f$ is transposed, the roles for \p row_indices and
   * \p col_indices should be swapped.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>The input vectors \p x and \p y are to be accessed via local indices
   * with the assistance of \p row_index_global_to_local_map and \p
   * col_index_global_to_local_map.</dd>
   * </dl>
   *
   * Also refer to HMatrix::vmult_local_vector.
   * @param y
   * @param x
   */
  void
  Tvmult_local_vector(Vector<Number> &y,
                      const std::map<types::global_dof_index, size_t>
                        &                   y_index_global_to_local_map,
                      const Vector<Number> &x,
                      const std::map<types::global_dof_index, size_t>
                        &x_index_global_to_local_map) const;

  /**
   * Perform \hmatrix MM multiplication reduction. This is
   * (7.21) in Hackbusch's \hmatrix book.
   */
  void
  h_h_mmult_reduction();

  /**
   * This function implements \p MM_H in Hackbusch's \hmatrix
   * book.
   */
  void
  h_h_mmult_horizontal_split(BlockClusterTree<spacedim, Number> &bc_tree);

  /**
   * This function implements \p MM_V in Hackbusch's \hmatrix
   * book.
   */
  void
  h_h_mmult_vertical_split(BlockClusterTree<spacedim, Number> &bc_tree);

  /**
   * This function implements \p MM_C in Hackbusch's \hmatrix
   * book.
   */
  void
  h_h_mmult_cross_split(BlockClusterTree<spacedim, Number> &bc_tree);

  /**
   * Multiplication of two \f$\mathcal{H}\f$-matrices \f$C = A \cdot B\f$.
   * @param C
   * @param B
   * @param bct_a
   * @param bct_b
   * @param bct_c
   * @param fixed_rank
   */
  void
  mmult(HMatrix<spacedim, Number> &               C,
        HMatrix<spacedim, Number> &               B,
        const BlockClusterTree<spacedim, Number> &bct_a,
        const BlockClusterTree<spacedim, Number> &bct_b,
        BlockClusterTree<spacedim, Number> &      bct_c,
        const unsigned int                        fixed_rank = 1);

  void
  mmult(HMatrix<spacedim, Number> &               C,
        HMatrix<spacedim, Number> &               B,
        const BlockClusterTree<spacedim, Number> &bct_a,
        const BlockClusterTree<spacedim, Number> &bct_b,
        BlockClusterTree<spacedim, Number> &      bct_c,
        const unsigned int                        fixed_rank,
        const bool                                adding);

  /**
   * Add the current HMatrix \p A with another HMatrix \p B into \p C, i.e.
   * whole matrix addition instead of addition limited to a specific block,
   * where \p C will be truncated to a fixed rank \p fixed_rank.
   *
   * This algorithm is intrinsically recursive, i.e. the addition of parent
   * HMatrices will perform the addition of each pair of child HMatrices
   * corresponding to a same block cluster. Strictly speaking, this member
   * function \p add is not a recursive function, because the class instance
   * which calls \p add changes from parent to child HMatrix.
   *
   * N.B.
   *
   * 1. The two operands should have the same partition.
   * 2. The hierarchical structure of \p C should be pre-generated.
   *
   * @param C
   * @param B
   * @param fixed_rank
   */
  void
  add(HMatrix<spacedim, Number> &      C,
      const HMatrix<spacedim, Number> &B,
      const size_type                  fixed_rank_k) const;

  /**
   * Add the HMatrix \p B into the current HMatrix \p A, i.e.
   * whole matrix addition instead of addition limited to a specific block,
   * where \p C will be truncated to a fixed rank \p fixed_rank.
   *
   * This algorithm is intrinsically recursive, i.e. the addition of parent
   * HMatrices will perform the addition of each pair of child HMatrices
   * corresponding to a same block cluster. Strictly speaking, this member
   * function \p add is not a recursive function, because the class instance
   * which calls \p add changes from parent to child HMatrix.
   *
   * N.B. The two operands should have the same partition.
   *
   * @param B
   * @param fixed_rank
   */
  void
  add(const HMatrix<spacedim, Number> &B, const size_type fixed_rank_k) const;

  /**
   * Calculate the inverse of the \hmatrix node via Gauss
   * elimination.
   *
   * @param M_inv
   * @param M_root The \hmatnode from which this recursive function is called for
   * the first time.
   * @param M_root_bct The \bct associated with \p M_root.
   * @param fixed_rank_k
   */
  void
  invert_by_gauss_elim(HMatrix<spacedim, Number> &               M_inv,
                       HMatrix<spacedim, Number> &               M_root,
                       const BlockClusterTree<spacedim, Number> &M_root_bct,
                       const size_type                           fixed_rank_k);

  /**
   * Coarsen the current \hmatrix so that it corresponds to the
   * partition determined by the \p subtree. Each rank-k matrix in the
   * hierarchical matrix structure will be truncated to \p fixed_rank_k.
   *
   * This function calls \p HMatrix<spacedim, Number>::coarsen_to_partition
   * internally. After that, the leaf set is rebuilt.
   *
   * This member function implements the operator \f$\mathcal{T}_{P' \leftarrow
   * P}^{\mathcal{H} \leftarrow \mathcal{H}}\f$ for the case \f$T(I \times J,
   * P') \subset T(I \times J, P)\f$ in (7.9) in Hackbusch's
   * \hmatrix book. Because there is no internal check about
   * this, users should ensure that the given \p subtree is really a subtree of
   * the block cluster tree associated with this \hmatrix
   * hierarchy.
   *
   * @param subtree
   * @param fixed_rank_k
   */
  void
  coarsen_to_subtree(const BlockClusterTree<spacedim, Number> &subtree,
                     const unsigned int                        fixed_rank_k);

  /**
   * Coarsen the current \hmatrix via recursive call so that its leaf set
   * complies with the given partition. Each rank-k matrix in the \hmatrix
   * structure will be truncated to \p fixed_rank_k.
   *
   * Since this is a recursive member function, it does not execute leaf set
   * rebuilding, which is an operation on the overall \hmat hierarchy.
   *
   * This member function implements the operator \f$\mathcal{T}_{P' \leftarrow
   * P}^{\mathcal{H} \leftarrow \mathcal{H}}\f$ for the case \f$T(I \times J,
   * P') \subset T(I \times J, P)\f$ in (7.9) in Hackbusch's
   * \hmatrix book. Because there is no internal check about
   * this, users should ensure this set inclusion relationship.
   *
   * @param partition
   * @param fixed_rank_k
   */
  void
  coarsen_to_partition(
    const std::vector<
      typename BlockClusterTree<spacedim, Number>::node_pointer_type>
      &                partition,
    const unsigned int fixed_rank_k);

  /**
   * Build the leaf set of the current \hmatrix node.
   */
  void
  build_leaf_set();

  /**
   * Get the reference to the leaf set of the current \hmatrix
   * node.
   * @return
   */
  std::vector<HMatrix<spacedim, Number> *> &
  get_leaf_set();

  /**
   * Get the reference to the leaf set of the current \hmatrix
   * node (const version).
   * @return
   */
  const std::vector<HMatrix<spacedim, Number> *> &
  get_leaf_set() const;

  /**
   * Find a block cluster in the leaf set of the current
   * \hmatrix and returns the iterator of the corresponding
   * \hmatrix node in the leaf set.
   *
   * @param block_cluster
   * @return
   */
  typename std::vector<HMatrix<spacedim, Number> *>::iterator
  find_block_cluster_in_leaf_set(
    const BlockCluster<spacedim, Number> &block_cluster);

  /**
   * Find a block cluster in the leaf set of the current
   * \hmatrix and returns the iterator of the corresponding
   * \hmatrix node in the leaf set (const version).
   *
   * @param block_cluster
   * @return
   */
  typename std::vector<HMatrix<spacedim, Number> *>::const_iterator
  find_block_cluster_in_leaf_set(
    const BlockCluster<spacedim, Number> &block_cluster) const;

  /**
   * Refine the current \hmatrix, whose associated block cluster
   * tree has been extended. The operation has no accuracy loss.
   *
   * This member function implements the operator \f$\mathcal{T}_{P' \leftarrow
   * P}^{\mathcal{H} \leftarrow \mathcal{H}}\f$ for the case \f$T(I \times J,
   * P') \supset T(I \times J, P)\f$ in (7.9) in Hackbusch's
   * \hmatrix book. Because there is no internal check about
   * this, users should ensure that the original block cluster tree associated
   * with this \hmatrix hierarchy has really been extended.
   */
  void
  refine_to_supertree();

  /**
   * Convert an \hmatrix between two different block cluster
   * trees \f$T\f$ and \f$T'\f$, where \f$T := T(I \times J, P)\f$ and \f$T' :=
   * T'(I \times J, P')\f$. The two trees have incompatible partitions and do
   * not contain each other. However, they are constructed on the same cluster
   * trees \f$T(I)\f$ and \f$T(J)\f$. This enables us to make a
   * <strong>shallow</strong> comparison of two block cluster nodes based on the
   * pointer addresses related to the comprising clusters, which is useful for
   * verify the equality of two block cluster nodes.
   *
   * The procedures of this algorithm are as below. Assume the current
   * \hmatrix to be converted is associated with the block
   * cluster tree \f$T\f$.
   *
   * 1. Extend \f$T\f$ to be finer than \f$T'\f$, from which we get the new
   * block cluster tree \f$T''\f$.
   * 2. Refine the original \hmatrix with respect to the
   * extended tree \f$T''\f$.
   * 3. Get and keep a record of the leaf set of the block cluster tree
   * \f$T'\f$, which will be used for matrix coarsening in the last step.
   * 4. Extend \f$T'\f$ to the finer block cluster tree \f$T''\f$, from which we
   * get \f$\tilde{T}'\f$.
   * 5. Build a new \hmatrix with respect to \f$\tilde{T}'\f$
   * with the actual data migrated from the leaf nodes of the original
   * \hmatrix.
   * 6. Coarsen the new \hmatrix to the original partition of
   * \f$T'\f$.
   * 7. Delete the hierarchy of the original \hmatrix.
   * 8. Assign the new \hmatrix object to the original
   * \hmatrix object.
   *
   * @param bct1 The block cluster tree which is associated with the current
   * \hmatrix.
   * @param bct2 The block cluster tree to which the current
   * \hmatrix is to be converted.
   */
  void
  convert_between_different_block_cluster_trees(
    BlockClusterTree<spacedim, Number> &bct1,
    BlockClusterTree<spacedim, Number> &bct2,
    const unsigned int                  fixed_rank_k2 = 1);

  /**
   * Remove a pair of \hmatrix nodes from the list of
   * matrix-matrix product subtasks to be performed, i.e. from the list
   * \p HMatrix::Sigma_P.
   */
  void
  remove_hmat_pair_from_mm_product_list(const HMatrix<spacedim, Number> *M1,
                                        const HMatrix<spacedim, Number> *M2);

  /**
   * Remove a pair of \hmatrix nodes from the list of
   * matrix-matrix product subtasks to be performed, i.e. from the list
   * \p HMatrix::Sigma_P.
   */
  void
  remove_hmat_pair_from_mm_product_list(
    const std::pair<const HMatrix<spacedim, Number> *,
                    const HMatrix<spacedim, Number> *> &hmat_pair);

  /**
   * Check the consistency of the tree node split modes which are associated
   * with the \hmatrix node pairs stored in the list
   * \f$\Sigma_P\f$ of the current \hmatrix node.
   * @return
   */
  TreeNodeSplitMode
  determine_mm_split_mode_from_Sigma_P();

private:
  /**
   * Convert an HMatrix to a full matrix by recursion.
   * @param matrix
   */
  template <typename MatrixType>
  void
  _convertToFullMatrix(MatrixType &M) const;

  /**
   * Collect \hmatrix nodes in the leaf set into a vector.
   * @param total_leaf_set
   */
  void
  _build_leaf_set(std::vector<HMatrix *> &total_leaf_set) const;

  void
  distribute_all_non_leaf_nodes_sigma_r_and_f_to_leaves();

  void
  distribute_sigma_r_and_f_to_leaves();

  void
  _distribute_sigma_r_and_f_to_leaves(HMatrix<spacedim, Number> &starting_hmat);

  /**
   * Matrix type.
   */
  HMatrixType type;

  /**
   * A list of submatrices of type HMatrix.
   */
  std::vector<HMatrix<spacedim, Number> *> submatrices;

  /**
   * A list of submatrices in the leaf set.
   */
  std::vector<HMatrix<spacedim, Number> *> leaf_set;

  /**
   * Pointer to the rank-k matrix. It is not null when the current HMatrix
   * object belongs to the far field.
   */
  RkMatrix<Number> *rkmatrix;

  /**
   * Pointer to the full matrix. It is not null when the current HMatrix object
   * belongs to the near field.
   */
  LAPACKFullMatrixExt<Number> *fullmatrix;

  /**
   * Pointer to the corresponding block cluster node in a BlockClusterTree.
   */
  typename BlockClusterTree<spacedim, Number>::node_pointer_type bc_node;

  /**
   * Pointer to the vector of global row indices, which is stored as the index
   * in the cluster \f$\tau\f$. It is a subset of \f$I\f$. By accessing this
   * vector using indices starting from 0, we actually obtain the mapping from
   * the current matrix's local row indices to the global row indices.
   */
  std::vector<types::global_dof_index> *row_indices;

  /**
   * Pointer to the vector of global column indices, which is stored as the
   * index set in the cluster \f$\sigma\f$. It is a subset of \f$J\f$. By
   * accessing this vector using indices starting from 0, we actually obtain the
   * mapping from the current matrix's local column indices to the global column
   * indices.
   */
  std::vector<types::global_dof_index> *col_indices;

  /**
   * Map from local row indices to global row indices for the cluster
   * \f$\tau\f$. The set of local row indices is the range \f$[0, \#\tau -
   * 1]\f$. The corresponding set of global row indices is a subset of \f$I\f$.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>This mapping is only constructed for H-matrices in the leaf set.</dd>
   * </dl>
   */
  std::map<types::global_dof_index, size_t> row_index_global_to_local_map;

  /**
   * Map from local column indices to global column indices for the cluster
   * \f$\sigma\f$. The set of local column indices is the range \f$[0, \#\sigma
   * - 1]\f$. The corresponding set of global column indices is a subset of
   * \f$J\f$.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>This mapping is only constructed for H-matrices in the leaf set.</dd>
   * </dl>
   */
  std::map<types::global_dof_index, size_t> col_index_global_to_local_map;

  /**
   * Total number of rows in the matrix.
   */
  size_type m;

  /**
   * Total number of columns in the matrix.
   */
  size_type n;

  /**
   * Block cluster tree when this matrix is the product of two
   * \f$\mathcal{H}\f$-matrices.
   */
  BlockClusterTree<spacedim, Number> Tind;

  /**
   * List of pairs of pointers to \hmatrix nodes for
   * multiplication.
   */
  std::vector<
    std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>>
    Sigma_P;

  /**
   * List of rank-k matrix pointers used in \hmatrix
   * multiplication.
   */
  std::vector<RkMatrix<Number> *> Sigma_R;

  /**
   * List of full matrix pointers used in \hmatrix
   * multiplication.
   */
  std::vector<LAPACKFullMatrixExt<Number> *> Sigma_F;
};


/**
 * Initialize an \hmatrix node with respect to a block cluster
 * node. The list \f$\Sigma_b^P\f$ is set to empty.
 * @param hmat
 * @param bc_node
 * @param Sigma_P
 */
template <int spacedim, typename Number = double>
void
InitHMatrixWrtBlockClusterNode(
  HMatrix<spacedim, Number> &                                          hmat,
  typename BlockClusterTree<spacedim, Number>::node_const_pointer_type bc_node)
{
  /**
   * Link \p hmat with \p bc_node.
   */
  hmat.bc_node =
    const_cast<typename BlockClusterTree<spacedim, Number>::node_pointer_type>(
      bc_node);

  /**
   * Link row and column indices stored in the clusters \f$\tau\f$ and
   * \f$\sigma\f$ respectively.
   */
  hmat.row_indices = const_cast<std::vector<types::global_dof_index> *>(
    &(hmat.bc_node->get_data_reference()
        .get_tau_node()
        ->get_data_reference()
        .get_index_set()));
  hmat.col_indices = const_cast<std::vector<types::global_dof_index> *>(
    &(hmat.bc_node->get_data_reference()
        .get_sigma_node()
        ->get_data_reference()
        .get_index_set()));

  /**
   * Update the matrix dimension of \p hmat.
   */
  hmat.m = hmat.row_indices->size();
  hmat.n = hmat.col_indices->size();

  hmat.Sigma_P.clear();
  hmat.Sigma_R.clear();
  hmat.Sigma_F.clear();
}


/**
 * Initialize an \hmatrix node with respect to a block cluster
 * node. Its member list \f$\Sigma_b^P\f$ will be merged with the given \p
 * Sigma_P.
 * @param hmat
 * @param bc_node
 * @param Sigma_P
 */
template <int spacedim, typename Number = double>
void
InitHMatrixWrtBlockClusterNode(
  HMatrix<spacedim, Number> &                                          hmat,
  typename BlockClusterTree<spacedim, Number>::node_const_pointer_type bc_node,
  const std::vector<std::pair<HMatrix<spacedim, Number> *,
                              HMatrix<spacedim, Number> *>> &          Sigma_P)
{
  /**
   * Link \p hmat with \p bc_node.
   */
  hmat.bc_node =
    const_cast<typename BlockClusterTree<spacedim, Number>::node_pointer_type>(
      bc_node);

  /**
   * Link row and column indices stored in the clusters \f$\tau\f$ and
   * \f$\sigma\f$ respectively.
   */
  hmat.row_indices = const_cast<std::vector<types::global_dof_index> *>(
    &(hmat.bc_node->get_data_reference()
        .get_tau_node()
        ->get_data_reference()
        .get_index_set()));
  hmat.col_indices = const_cast<std::vector<types::global_dof_index> *>(
    &(hmat.bc_node->get_data_reference()
        .get_sigma_node()
        ->get_data_reference()
        .get_index_set()));

  /**
   * Update the matrix dimension of \p hmat.
   */
  hmat.m = hmat.row_indices->size();
  hmat.n = hmat.col_indices->size();

  for (std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>
         &hmat_pair : Sigma_P)
    {
      hmat.Sigma_P.push_back(hmat_pair);
    }

  hmat.Sigma_R.clear();
  hmat.Sigma_F.clear();
}


/**
 * Initialize an \hmatrix node with respect to a block cluster
 * node. The given \p hmat_pair will be appended to the list
 * \f$\Sigma_b^P\f$.
 * @param hmat
 * @param bc_node
 * @param hmat_pair
 */
template <int spacedim, typename Number = double>
void
InitHMatrixWrtBlockClusterNode(
  HMatrix<spacedim, Number> &                                          hmat,
  typename BlockClusterTree<spacedim, Number>::node_const_pointer_type bc_node,
  const std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>
    &hmat_pair)
{
  /**
   * Link \p hmat with \p bc_node.
   */
  hmat.bc_node =
    const_cast<typename BlockClusterTree<spacedim, Number>::node_pointer_type>(
      bc_node);

  /**
   * Link row and column indices stored in the clusters \f$\tau\f$ and
   * \f$\sigma\f$ respectively.
   */
  hmat.row_indices = const_cast<std::vector<types::global_dof_index> *>(
    &(hmat.bc_node->get_data_reference()
        .get_tau_node()
        ->get_data_reference()
        .get_index_set()));
  hmat.col_indices = const_cast<std::vector<types::global_dof_index> *>(
    &(hmat.bc_node->get_data_reference()
        .get_sigma_node()
        ->get_data_reference()
        .get_index_set()));

  /**
   * Update the matrix dimension of \p hmat.
   */
  hmat.m = hmat.row_indices->size();
  hmat.n = hmat.col_indices->size();

  hmat.Sigma_P.push_back(hmat_pair);
  hmat.Sigma_R.clear();
  hmat.Sigma_F.clear();
}


/**
 * Recursively construct the children of an \hmatrix with
 * respect to a block cluster tree by starting from a tree node which is
 * associated with the current \hmatrix.
 *
 * The matrices in the leaf set are initialized with zero values. The rank of
 * the near field matrices are predefined fixed values.
 *
 * @param hmat Pointer to the current \hmatrix node, <strong>which has already been
 * created on the heap but with its internal data left empty.</strong>
 * @param bc_node Pointer to a TreeNode in a BlockClusterTree, which is to be
 * associated with \p hmat.
 */
template <int spacedim, typename Number = double>
void
InitAndCreateHMatrixChildrenWithoutAlloc(
  HMatrix<spacedim, Number> *                                          hmat,
  typename BlockClusterTree<spacedim, Number>::node_const_pointer_type bc_node,
  const bool is_build_index_set_global_to_local_map = true)
{
  /**
   * Link \p hmat with \p bc_node.
   */
  hmat->bc_node =
    const_cast<typename BlockClusterTree<spacedim, Number>::node_pointer_type>(
      bc_node);

  /**
   * Link row and column indices stored in the clusters \f$\tau\f$ and
   * \f$\sigma\f$ respectively.
   */
  hmat->row_indices = const_cast<std::vector<types::global_dof_index> *>(
    &(hmat->bc_node->get_data_reference()
        .get_tau_node()
        ->get_data_reference()
        .get_index_set()));
  hmat->col_indices = const_cast<std::vector<types::global_dof_index> *>(
    &(hmat->bc_node->get_data_reference()
        .get_sigma_node()
        ->get_data_reference()
        .get_index_set()));

  /**
   * Update the matrix dimension of \p hmat.
   */
  hmat->m = hmat->row_indices->size();
  hmat->n = hmat->col_indices->size();

  const unsigned int bc_node_child_num = bc_node->get_child_num();
  if (bc_node_child_num > 0)
    {
      /**
       * When the block cluster node \p bc_node has children, set the current \p
       * hmat type as \p HierarchicalMatrixType.
       */
      hmat->type = HierarchicalMatrixType;

      /**
       * Then we will continue constructing its hierarchical submatrices.
       */
      for (unsigned int i = 0; i < bc_node_child_num; i++)
        {
          /**
           * Create an empty HMatrix on the heap.
           */
          HMatrix<spacedim, Number> *child_hmat =
            new HMatrix<spacedim, Number>();

          InitAndCreateHMatrixChildrenWithoutAlloc(
            child_hmat,
            bc_node->get_child_pointer(i),
            is_build_index_set_global_to_local_map);

          /**
           * Append the initialized child to the list of submatrices of \p hmat.
           */
          hmat->submatrices.push_back(child_hmat);
        }
    }
  else
    {
      if (is_build_index_set_global_to_local_map)
        {
          /**
           * Build the maps from global row and column indices respectively to
           * local indices.
           */
          build_index_set_global_to_local_map(
            *(hmat->row_indices), hmat->row_index_global_to_local_map);
          build_index_set_global_to_local_map(
            *(hmat->col_indices), hmat->col_index_global_to_local_map);
        }

      /**
       * Update the current matrix type according to the identity of the block
       * cluster node. When the block cluster belongs to the near field, \p hmat
       * should be represented as a \p LAPACKFullMatrixExt. When the block
       * cluster belongs to the far field, \p hmat should be represented as an
       * \p RkMatrix. Correspondingly, new matrices, either full matrix or
       * rank-k matrix will be created on the heap and assigned to the current
       * \hmatrix.
       */
      if (bc_node->get_data_reference().get_is_near_field())
        {
          hmat->type       = FullMatrixType;
          hmat->fullmatrix = new LAPACKFullMatrixExt<Number>();
        }
      else
        {
          hmat->type     = RkMatrixType;
          hmat->rkmatrix = new RkMatrix<Number>();
        }
    }
}


/**
 * Recursively construct the children of an \hmatrix with
 * respect to a block cluster tree by starting from a tree node which is
 * associated with the current \hmatrix.
 *
 * The matrices in the leaf set are initialized with zero values. The rank of
 * the near field matrices are predefined fixed values.
 *
 * @param hmat Pointer to the current \hmatrix node, <strong>which has already been
 * created on the heap but with its internal data left empty.</strong>
 * @param bc_node Pointer to a TreeNode in a BlockClusterTree, which is to be
 * associated with \p hmat.
 */
template <int spacedim, typename Number = double>
void
InitAndCreateHMatrixChildren(
  HMatrix<spacedim, Number> *                                          hmat,
  typename BlockClusterTree<spacedim, Number>::node_const_pointer_type bc_node,
  const unsigned int fixed_rank_k,
  bool               is_build_index_set_global_to_local_map = true)
{
  /**
   * Link \p hmat with \p bc_node.
   */
  hmat->bc_node =
    const_cast<typename BlockClusterTree<spacedim, Number>::node_pointer_type>(
      bc_node);

  /**
   * Link row and column indices stored in the clusters \f$\tau\f$ and
   * \f$\sigma\f$ respectively.
   */
  hmat->row_indices = const_cast<std::vector<types::global_dof_index> *>(
    &(hmat->bc_node->get_data_reference()
        .get_tau_node()
        ->get_data_reference()
        .get_index_set()));
  hmat->col_indices = const_cast<std::vector<types::global_dof_index> *>(
    &(hmat->bc_node->get_data_reference()
        .get_sigma_node()
        ->get_data_reference()
        .get_index_set()));

  /**
   * Update the matrix dimension of \p hmat.
   */
  hmat->m = hmat->row_indices->size();
  hmat->n = hmat->col_indices->size();

  const unsigned int bc_node_child_num = bc_node->get_child_num();
  if (bc_node_child_num > 0)
    {
      /**
       * When the block cluster node \p bc_node has children, set the current \p
       * hmat type as \p HierarchicalMatrixType.
       */
      hmat->type = HierarchicalMatrixType;

      /**
       * Then we will continue constructing its hierarchical submatrices.
       */
      for (unsigned int i = 0; i < bc_node_child_num; i++)
        {
          /**
           * Create an empty HMatrix on the heap.
           */
          HMatrix<spacedim, Number> *child_hmat =
            new HMatrix<spacedim, Number>();

          InitAndCreateHMatrixChildren(child_hmat,
                                       bc_node->get_child_pointer(i),
                                       fixed_rank_k,
                                       is_build_index_set_global_to_local_map);

          /**
           * Append the initialized child to the list of submatrices of \p hmat.
           */
          hmat->submatrices.push_back(child_hmat);
        }
    }
  else
    {
      if (is_build_index_set_global_to_local_map)
        {
          /**
           * Build the maps from global row and column indices respectively to
           * local indices.
           */
          build_index_set_global_to_local_map(
            *(hmat->row_indices), hmat->row_index_global_to_local_map);
          build_index_set_global_to_local_map(
            *(hmat->col_indices), hmat->col_index_global_to_local_map);
        }

      /**
       * Update the current matrix type according to the identity of the block
       * cluster node. When the block cluster belongs to the near field, \p hmat
       * should be represented as a \p LAPACKFullMatrixExt. When the block
       * cluster belongs to the far field, \p hmat should be represented as an
       * \p RkMatrix. Correspondingly, new matrices, either full matrix or
       * rank-k matrix will be created on the heap and assigned to the current
       * \hmatrix.
       */
      if (bc_node->get_data_reference().get_is_near_field())
        {
          hmat->type       = FullMatrixType;
          hmat->fullmatrix = new LAPACKFullMatrixExt<Number>(hmat->m, hmat->n);
        }
      else
        {
          hmat->type     = RkMatrixType;
          hmat->rkmatrix = new RkMatrix<Number>(hmat->m, hmat->n, fixed_rank_k);
        }
    }
}


/**
 * Recursively construct the children of an \hmatrix with
 * respect to a block cluster tree by starting from a tree node which is
 * associated with the current \hmatrix.
 *
 * The matrices in the leaf set are initialized with the data in the given
 * global full matrix \p M, which is created on the complete block cluster index
 * set \f$I \times J\f$ and whose elements should be accessed via indices stored
 * in the block cluster. The rank of the near field matrices are predefined
 * fixed values.
 *
 * During the recursive calling of this function, the source data matrix \p M is
 * kept intact, which will not be restricted to small matrix blocks.
 *
 * @param hmat Pointer to the current \hmatrix node, <strong>which has already been
 * created on the heap but with its internal data left empty.</strong>
 * @param bc_node Pointer to a TreeNode in a BlockClusterTree, which is to be
 * associated with \p hmat.
 * @param M The global full matrix containing all the data required to initialize the
 * \hmatrix.
 */
template <int spacedim, typename Number = double>
void
InitAndCreateHMatrixChildren(
  HMatrix<spacedim, Number> *                                          hmat,
  typename BlockClusterTree<spacedim, Number>::node_const_pointer_type bc_node,
  const unsigned int                 fixed_rank_k,
  const LAPACKFullMatrixExt<Number> &M,
  bool is_build_index_set_global_to_local_map = true)
{
  /**
   * Link \p hmat with \p bc_node.
   */
  hmat->bc_node =
    const_cast<typename BlockClusterTree<spacedim, Number>::node_pointer_type>(
      bc_node);

  /**
   * Link row and column indices.
   */
  hmat->row_indices = const_cast<std::vector<types::global_dof_index> *>(
    &(hmat->bc_node->get_data_reference()
        .get_tau_node()
        ->get_data_reference()
        .get_index_set()));
  hmat->col_indices = const_cast<std::vector<types::global_dof_index> *>(
    &(hmat->bc_node->get_data_reference()
        .get_sigma_node()
        ->get_data_reference()
        .get_index_set()));

  /**
   * Update the matrix dimension of \p hmat.
   */
  hmat->m = hmat->row_indices->size();
  hmat->n = hmat->col_indices->size();

  const unsigned int bc_node_child_num = bc_node->get_child_num();

  if (bc_node_child_num > 0)
    {
      /**
       * When the block cluster node \p bc_node has children, set the current \p
       * hmat type as \p HierarchicalMatrixType. Then we will
       * continue constructing hierarchical submatrices.
       */
      hmat->type = HierarchicalMatrixType;

      for (unsigned int i = 0; i < bc_node_child_num; i++)
        {
          /**
           * Create an empty HMatrix on the heap.
           */
          HMatrix<spacedim, Number> *child_hmat =
            new HMatrix<spacedim, Number>();

          InitAndCreateHMatrixChildren(child_hmat,
                                       bc_node->get_child_pointer(i),
                                       fixed_rank_k,
                                       M,
                                       is_build_index_set_global_to_local_map);

          /**
           * Append the initialized child to the list of submatrices of \p hmat.
           */
          hmat->submatrices.push_back(child_hmat);
        }
    }
  else
    {
      if (is_build_index_set_global_to_local_map)
        {
          /**
           * Build the maps from global row and column indices respectively to
           * local indices.
           */
          build_index_set_global_to_local_map(
            *(hmat->row_indices), hmat->row_index_global_to_local_map);
          build_index_set_global_to_local_map(
            *(hmat->col_indices), hmat->col_index_global_to_local_map);
        }

      /**
       * Update the current matrix type according to the identity of the block
       * cluster node. When the block cluster belongs to the near field, \p hmat
       * should be represented as a \p LAPACKFullMatrixExt. When the block
       * cluster belongs to the far field, \p hmat should be represented as an
       * \p RkMatrix. Correspondingly, new matrices, either full matrix or
       * rank-k matrix will be created on the heap and assigned to the current
       * \hmatrix.
       */
      if (bc_node->get_data_reference().get_is_near_field())
        {
          hmat->type       = FullMatrixType;
          hmat->fullmatrix = new LAPACKFullMatrixExt<Number>(hmat->m, hmat->n);

          /**
           * Assign matrix values from \p M to the current HMatrix.
           */
          for (unsigned int i = 0; i < hmat->m; i++)
            {
              for (unsigned int j = 0; j < hmat->n; j++)
                {
                  (*hmat->fullmatrix)(i, j) =
                    M(hmat->row_indices->at(i), hmat->col_indices->at(j));
                }
            }
        }
      else
        {
          hmat->type     = RkMatrixType;
          hmat->rkmatrix = new RkMatrix<Number>(*(hmat->row_indices),
                                                *(hmat->col_indices),
                                                fixed_rank_k,
                                                M);
        }
    }
}


/**
 * Recursively construct the children of an \hmatrix with
 * respect to a block cluster tree by starting from a tree node which is
 * associated with the current \hmatrix.
 *
 * The matrices in the leaf set are initialized with the data in the given full
 * matrix \p M, which is created on the block cluster index set \f$\tau \times
 * \sigma\f$ associated with the current \hmatrix. Hence, this
 * full matrix is just a block of the original global full matrix created on the
 * block cluster index set \f$I \times J\f$. The rank of the near field matrices
 * are predefined fixed values.
 *
 * During the recursive calling of this function, the source data matrix \p M is
 * kept intact, which will not be restricted to small matrix blocks.
 *
 * @param hmat Pointer to the current \hmatrix node, <strong>which has already been
 * created on the heap but with its internal data left empty.</strong>
 * @param bc_node Pointer to a TreeNode in a BlockClusterTree, which is to be
 * associated with \p hmat.
 * @param M The full matrix, as a submatrix of the global full matrix,
 * containing all the data required to initialize the \hmatrix.
 * @param row_index_global_to_local_map_for_M The map from the global row
 * indices to the local indices of the matrix associated the
 * \hmatrix when first calling this recursive function.
 * @param col_index_global_to_local_map_for_M The map from the global column
 * indices to the local indices of the matrix associated the
 * \hmatrix when first calling this recursive function.
 */
template <int spacedim, typename Number = double>
void
InitAndCreateHMatrixChildren(
  HMatrix<spacedim, Number> *                                          hmat,
  typename BlockClusterTree<spacedim, Number>::node_const_pointer_type bc_node,
  const unsigned int                 fixed_rank_k,
  const LAPACKFullMatrixExt<Number> &M,
  const std::map<types::global_dof_index, size_t>
    &row_index_global_to_local_map_for_M,
  const std::map<types::global_dof_index, size_t>
    &  col_index_global_to_local_map_for_M,
  bool is_build_index_set_global_to_local_map = true)
{
  /**
   * Link \p hmat with \p bc_node.
   */
  hmat->bc_node =
    const_cast<typename BlockClusterTree<spacedim, Number>::node_pointer_type>(
      bc_node);

  /**
   * Link row and column indices.
   */
  hmat->row_indices = const_cast<std::vector<types::global_dof_index> *>(
    &(hmat->bc_node->get_data_reference()
        .get_tau_node()
        ->get_data_reference()
        .get_index_set()));
  hmat->col_indices = const_cast<std::vector<types::global_dof_index> *>(
    &(hmat->bc_node->get_data_reference()
        .get_sigma_node()
        ->get_data_reference()
        .get_index_set()));

  /**
   * Update the matrix dimension of \p hmat.
   */
  hmat->m = hmat->row_indices->size();
  hmat->n = hmat->col_indices->size();

  const unsigned int bc_node_child_num = bc_node->get_child_num();

  if (bc_node_child_num > 0)
    {
      /**
       * When the block cluster node \p bc_node has children, set the current \p
       * hmat type as \p HierarchicalMatrixType. Then we will
       * continue constructing hierarchical submatrices.
       */
      hmat->type = HierarchicalMatrixType;

      for (unsigned int i = 0; i < bc_node_child_num; i++)
        {
          /**
           * Create an empty HMatrix on the heap.
           */
          HMatrix<spacedim, Number> *child_hmat =
            new HMatrix<spacedim, Number>();

          InitAndCreateHMatrixChildren(child_hmat,
                                       bc_node->get_child_pointer(i),
                                       fixed_rank_k,
                                       M,
                                       row_index_global_to_local_map_for_M,
                                       col_index_global_to_local_map_for_M,
                                       is_build_index_set_global_to_local_map);

          /**
           * Append the initialized child to the list of submatrices of \p hmat.
           */
          hmat->submatrices.push_back(child_hmat);
        }
    }
  else
    {
      if (is_build_index_set_global_to_local_map)
        {
          /**
           * Build the maps from global row and column indices respectively to
           * local indices.
           */
          build_index_set_global_to_local_map(
            *(hmat->row_indices), hmat->row_index_global_to_local_map);
          build_index_set_global_to_local_map(
            *(hmat->col_indices), hmat->col_index_global_to_local_map);
        }

      /**
       * Update the current matrix type according to the identity of the block
       * cluster node. When the block cluster belongs to the near field, \p hmat
       * should be represented as a \p LAPACKFullMatrixExt. When the block
       * cluster belongs to the far field, \p hmat should be represented as an
       * \p RkMatrix. Correspondingly, new matrices, either full matrix or
       * rank-k matrix will be created on the heap and assigned to the current
       * \hmatrix.
       */
      if (bc_node->get_data_reference().get_is_near_field())
        {
          hmat->type       = FullMatrixType;
          hmat->fullmatrix = new LAPACKFullMatrixExt<Number>(hmat->m, hmat->n);

          /**
           * Assign matrix values from \p M to the current HMatrix.
           */
          for (unsigned int i = 0; i < hmat->m; i++)
            {
              for (unsigned int j = 0; j < hmat->n; j++)
                {
                  (*hmat->fullmatrix)(i, j) =
                    M(row_index_global_to_local_map_for_M.at(
                        hmat->row_indices->at(i)),
                      col_index_global_to_local_map_for_M.at(
                        hmat->col_indices->at(j)));
                }
            }
        }
      else
        {
          hmat->type = RkMatrixType;
          hmat->rkmatrix =
            new RkMatrix<Number>(*(hmat->row_indices),
                                 *(hmat->col_indices),
                                 fixed_rank_k,
                                 M,
                                 row_index_global_to_local_map_for_M,
                                 col_index_global_to_local_map_for_M);
        }
    }
}


/**
 * Recursively construct the children of an \hmatrix node with
 * respect to a block cluster tree by starting from a tree node which is
 * associated with the current \hmatrix node.
 *
 * The matrices in the leaf set take the data migrated from the leaf set of the
 * given \hmatrix \p M.
 *
 * @param hmat
 * @param M
 */
template <int spacedim, typename Number = double>
void
InitAndCreateHMatrixChildren(
  HMatrix<spacedim, Number> *                                          hmat,
  typename BlockClusterTree<spacedim, Number>::node_const_pointer_type bc_node,
  HMatrix<spacedim, Number> &&                                         H)
{
  /**
   * Link \p hmat with \p bc_node.
   */
  hmat->bc_node =
    const_cast<typename BlockClusterTree<spacedim, Number>::node_pointer_type>(
      bc_node);

  /**
   * Link row and column indices stored in the clusters \f$\tau\f$ and
   * \f$\sigma\f$ respectively..
   */
  hmat->row_indices = const_cast<std::vector<types::global_dof_index> *>(
    &(hmat->bc_node->get_data_reference()
        .get_tau_node()
        ->get_data_reference()
        .get_index_set()));
  hmat->col_indices = const_cast<std::vector<types::global_dof_index> *>(
    &(hmat->bc_node->get_data_reference()
        .get_sigma_node()
        ->get_data_reference()
        .get_index_set()));

  /**
   * Update the matrix dimension of \p hmat.
   */
  hmat->m = hmat->row_indices->size();
  hmat->n = hmat->col_indices->size();

  const unsigned int bc_node_child_num = bc_node->get_child_num();

  if (bc_node_child_num > 0)
    {
      /**
       * When the block cluster node \p bc_node has children, set the current \p
       * hmat type as \p HierarchicalMatrixType.
       */
      hmat->type = HierarchicalMatrixType;

      /**
       * Then we will continue constructing its hierarchical submatrices.
       */
      for (unsigned int i = 0; i < bc_node_child_num; i++)
        {
          /**
           * Create an empty HMatrix on the heap.
           */
          HMatrix<spacedim, Number> *child_hmat =
            new HMatrix<spacedim, Number>();

          InitAndCreateHMatrixChildren(child_hmat,
                                       bc_node->get_child_pointer(i),
                                       std::move(H));

          /**
           * Append the initialized child to the list of submatrices of \p hmat.
           */
          hmat->submatrices.push_back(child_hmat);
        }
    }
  else
    {
      /**
       * When the current \hmatrix node is a leaf, migrate the
       * data from the leaf set of \p H to it.
       */
      HMatrix<spacedim, Number> *matched_source_hmat = *(
        H.find_block_cluster_in_leaf_set(hmat->bc_node->get_data_reference()));
      /**
       * Shallow copy the found \hmatrix node in the leaf set to
       * the current \hmatrix node.
       */
      (*hmat) = std::move(*matched_source_hmat);
    }
}


/**
 * Refine an \hmatrix node with respect to its associated block
 * cluster tree which has already been extended to be finer than the original
 * tree. The \hmatrix node should be of either \p FullMatrixType
 * or \p RkMatrixType, i.e. it belongs to the leaf set of the block cluster tree
 * before extension.
 *
 * @param starting_hmat The pointer to the initial \hmatrix
 * node from which this recursive function is called for the first time, i.e.
 * the \hmatrix node from which the refinement begins.
 * @param current_hmat The pointer to the current \hmatrix node
 * being handled during the recursion. For the first time of calling this
 * function, \p current_hmat is the same as \p starting_hmat.
 */
template <int spacedim, typename Number>
void
RefineHMatrixWrtExtendedBlockClusterTree(
  HMatrix<spacedim, Number> *starting_hmat,
  HMatrix<spacedim, Number> *current_hmat)
{
  /**
   * <dl class="section">
   *   <dt>Work flow</dt>
   *   <dd>Because the \hmatrix node from which the refinement
   * begins belongs to the leaf set of the original block cluster tree, its
   * \hmatrix type can only be \p FullMatrixType or \p
   * RkMatrixType. Therefore, we make an assertion here.
   */
  Assert((starting_hmat->type == FullMatrixType) ||
           (starting_hmat->type == RkMatrixType),
         ExcInvalidHMatrixType(starting_hmat->type));

  /**
   * Determine the total number of children of the current
   * \hmatrix node by querying its associated block cluster
   * node. We do it like this is because the block cluster tree has already been
   * extended which contains a set of child node, while the hierarchy of
   * H-matrices has still not been extended yet.
   */
  const unsigned int bc_node_child_num = current_hmat->bc_node->get_child_num();

  if (bc_node_child_num > 0)
    {
      /**
       * If the associated block cluster node of the current
       * \hmatrix node has children, we firstly update the
       * \hmatrix type for the current \hmatrix
       * node as \p HierarchicalMatrix and this is only performed when the
       * current \hmatrix node is not the starting
       * \hmatrix node, because the original matrix type of the
       * starting \hmatrix node will be used later during
       * restriction operations to the current block cluster.
       */
      if (current_hmat != starting_hmat)
        {
          current_hmat->type = HierarchicalMatrixType;
        }

      for (unsigned int i = 0; i < bc_node_child_num; i++)
        {
          /**
           * For each of the children, create an empty \hmatrix
           * node on the heap and append it to the list of submatrices of the
           * current \hmatrix.
           */
          HMatrix<spacedim, Number> *child_hmat =
            new HMatrix<spacedim, Number>();

          current_hmat->submatrices.push_back(child_hmat);

          /**
           * Link the child \hmatrix node with the corresponding
           * block cluster node.
           */
          child_hmat->bc_node = const_cast<
            typename BlockClusterTree<spacedim, Number>::node_pointer_type>(
            current_hmat->bc_node->get_child_pointer(i));

          /**
           * Link row and column indices of the child \hmatrix
           * node to those index sets stored in clusters.
           */
          child_hmat->row_indices =
            const_cast<std::vector<types::global_dof_index> *>(
              &(child_hmat->bc_node->get_data_reference()
                  .get_tau_node()
                  ->get_data_reference()
                  .get_index_set()));
          child_hmat->col_indices =
            const_cast<std::vector<types::global_dof_index> *>(
              &(child_hmat->bc_node->get_data_reference()
                  .get_sigma_node()
                  ->get_data_reference()
                  .get_index_set()));

          /**
           * Update the matrix dimension of the child \hmatrix
           * node.
           */
          child_hmat->m = child_hmat->row_indices->size();
          child_hmat->n = child_hmat->col_indices->size();

          /**
           * Recursively call the function.
           */
          RefineHMatrixWrtExtendedBlockClusterTree(starting_hmat, child_hmat);
        }
    }
  else
    {
      /**
       * When the current \hmatrix node has no children, i.e. it
       * belongs to the leaf set of the extended block cluster tree.
       */
      if (current_hmat == starting_hmat)
        {
          /**
           * If the current \hmatrix node is still the same as
           * the starting \hmatrix node, there is no actual
           * refinement work to be done.
           */
        }
      else
        {
          //          /**
          //           * If the current \hmatrix node is not the
          //           starting
          //           * \hmatrix node, we firstly build the
          //           maps from
          //           * global row and column indices to their respective local
          //           indices.
          //           * When there will be further refinement from those nodes,
          //           these maps
          //           * will be used for matrix restriction.
          //           *
          //           * <dl class="section note">
          //           *   <dt>Note</dt>
          //           *   <dd>These two global to local maps are only needed
          //           for
          //           * \hmatrix nodes in the leaf set, because
          //           only these
          //           * \hmatrix nodes contain the actual full
          //           matrix data
          //           * or rank-k matrix data.</dd>
          //           * </dl>
          //           */
          //          build_index_set_global_to_local_map(
          //            *(current_hmat->row_indices),
          //            current_hmat->row_index_global_to_local_map);
          //          build_index_set_global_to_local_map(
          //            *(current_hmat->col_indices),
          //            current_hmat->col_index_global_to_local_map);

          /**
           * Update the current \hmatrix node type according to
           * the identity of the block cluster node: when the block cluster
           * belongs to the near field, \p current_hmat should be represented as
           * a full matrix \p LAPACKFullMatrixExt; when the block cluster
           * belongs to the far field, \p current_hmat should be represented as
           * a rank-k matrix \p RkMatrix. Correspondingly, new matrices, either
           * full matrix or rank-k matrix will be created on the heap and
           * assigned to the corresponding field of the current
           * \hmatrix.
           */
          if (current_hmat->bc_node->get_data_reference().get_is_near_field())
            {
              current_hmat->type = FullMatrixType;

              /**
               * Fill the current full matrix with the data extracted from
               * the starting \hmatrix node. This is actually a
               * restriction of the starting \hmatrix node to
               * the current \hmatrix node.
               */
              switch (starting_hmat->type)
                {
                  case FullMatrixType:
                    {
                      current_hmat->fullmatrix =
                        new LAPACKFullMatrixExt<Number>(
                          *(current_hmat->row_indices),
                          *(current_hmat->col_indices),
                          *(starting_hmat->fullmatrix),
                          starting_hmat->row_index_global_to_local_map,
                          starting_hmat->col_index_global_to_local_map);

                      break;
                    }
                  case RkMatrixType:
                    {
                      current_hmat->fullmatrix =
                        new LAPACKFullMatrixExt<Number>();
                      starting_hmat->rkmatrix->restrictToFullMatrix(
                        *(current_hmat->row_indices),
                        *(current_hmat->col_indices),
                        starting_hmat->row_index_global_to_local_map,
                        starting_hmat->col_index_global_to_local_map,
                        *(current_hmat->fullmatrix));

                      break;
                    }
                  default:
                    {
                      Assert(false, ExcInvalidHMatrixType(starting_hmat->type));

                      break;
                    }
                }
            }
          else
            {
              current_hmat->type = RkMatrixType;

              /**
               * Fill the current rank-k matrix with the data extracted
               * from the starting \hmatrix node.
               */
              switch (starting_hmat->type)
                {
                  case FullMatrixType:
                    {
                      current_hmat->rkmatrix = new RkMatrix<Number>(
                        *(current_hmat->row_indices),
                        *(current_hmat->col_indices),
                        *(starting_hmat->fullmatrix),
                        starting_hmat->row_index_global_to_local_map,
                        starting_hmat->col_index_global_to_local_map);

                      break;
                    }
                  case RkMatrixType:
                    {
                      current_hmat->rkmatrix = new RkMatrix<Number>(
                        *(current_hmat->row_indices),
                        *(current_hmat->col_indices),
                        *(starting_hmat->rkmatrix),
                        starting_hmat->row_index_global_to_local_map,
                        starting_hmat->col_index_global_to_local_map);

                      break;
                    }
                  default:
                    {
                      Assert(false, ExcInvalidHMatrixType(starting_hmat->type));

                      break;
                    }
                }
            }
        }
    }

  /**
   *   </dd>
   * </dl>
   */
}


/**
 * Convert an \hmatrix block \p hmat_block recursively into a
 * rank-k matrix or a full matrix, which depends on whether the block cluster
 * associated with \p hmat_block is large or not.
 *
 * Generally speaking, this method can be considered as the agglomeration of all
 * descendants of \p hmat_block.
 *
 * <dl class="section note">
 *   <dt>Note</dt>
 *   <dd>This method implements the operator \f$\mathcal{T}_r^{\mathcal{R}
 * \leftarrow \mathcal{H}}\f$, i.e. the algorithm \f$Convert\_H\f$ in (7.8) in
 * Hackbusch's \hmatrix book.</dd>
 * </dl>
 *
 * This \hmatrix block is implemented as a node in a whole
 * \hmatrix hierarchy. This conversion algorithm will
 * recursively descend in the hierarchical matrices for processing:
 *
 * 1. when the current matrix block belongs to the near field set
 * \f$P^-\f$, it is represented as a full matrix and no operations will be
 * applied to it;
 *
 * 2. when it belongs to the far field set \f$P^+\f$, it is already
 * a rank-k matrix, which will then be truncated to the given \p fixed_rank_k;
 *
 * 3. when it is not a leaf, i.e. it is a hierarchical matrix, this
 * function will be called recursively for each of its children. After that,
 *
 *   a. if the block cluster related to the current matrix is large, pairwise
 * agglomeration for rank-k matrices will be performed and a rank-k matrix will
 * be obtained with the given rank \p fixed_rank_k;
 *
 *   b. if the block cluster related to the current matrix is small,
 * agglomeration of full matrices will be performed and a full matrix will be
 * obtained.
 *
 * @param hmat_block the pointer to the current matrix block from which the
 * recursion will start.
 * @param fixed_rank_k the fixed rank to which the rank-k matrices in the far
 * field set will be truncated.
 * @param hmat_root_block the pointer to the root \hmatrix
 * block, which is only used for exporting matrix partition structure for
 * further visualization.
 * @param calling_counter the pointer to the counter which records the current
 * total number of calling times of this function. Its value will be used to
 * construct the name of the output file, which stores the matrix partition
 * structure.
 * @param output_file_base_name the based name of the output file which stores
 * the matrix partition structure.
 */
template <int spacedim, typename Number = double>
void
convertHMatBlockToRkMatrix(
  HMatrix<spacedim, Number> *      hmat_block,
  const unsigned int               fixed_rank_k,
  const HMatrix<spacedim, Number> *hmat_root_block = nullptr,
  size_t *                         calling_counter = nullptr,
  const std::string &output_file_base_name         = std::string("hmat-bct"))
{
  /**
   * <dl class="section">
   *   <dt>Work flow</dt>
   *   <dd>
   */
  using size_type = typename HMatrix<spacedim, Number>::size_type;

  Assert(hmat_block->type != UndefinedMatrixType,
         ExcInvalidHMatrixType(hmat_block->type));

  if (hmat_block->bc_node->is_leaf())
    {
      /**
       * When the current \hmatrix block belongs to the leaf
       * set.
       */
      if (hmat_block->bc_node->get_data_reference().get_is_near_field())
        {
          /**
           * When the current \hmatrix belongs to the near field
           * set, it should be of a full matrix type. Therefore, we make an
           * assertion here. After that we do nothing, since a near field node
           * should always be represented as a full matrix, thus the rank
           * truncation should not be applied.
           */
          Assert(hmat_block->type == FullMatrixType,
                 ExcInvalidHMatrixType(hmat_block->type));
        }
      else
        {
          /**
           * When the current \hmatrix belongs to the far field
           * set, it should be of a rank-k matrix type. Therefore, we make an
           * assertion here. After that the rank-k matrix block is truncated to
           * the specified rank.
           */
          Assert(hmat_block->type == RkMatrixType,
                 ExcInvalidHMatrixType(hmat_block->type));

          hmat_block->rkmatrix->truncate_to_rank(fixed_rank_k);
        }
    }
  else
    {
      /**
       * When the current \hmatrix block does not belong to the
       * leaf set, recursively convert each child of it to rank-k matrix if
       * possible.
       */
      for (auto submatrix : hmat_block->submatrices)
        {
          convertHMatBlockToRkMatrix(submatrix,
                                     fixed_rank_k,
                                     hmat_root_block,
                                     calling_counter,
                                     output_file_base_name);
        }

      if (hmat_block->bc_node->get_data_reference().get_is_near_field())
        {
          /**
           * When the current \hmatrix block belongs to the near
           * field set, we perform the operation of full matrix agglomeration.
           *
           * <dl class="section note">
           *   <dt>Note</dt>
           *   <dd>Normally, this case cannot happen because when an
           * \hmatrix block belongs to the near field, it is
           * represented as a full matrix and belongs to the leaf set. However,
           * this contradicts the precondition that the current
           * \hmatrix block does not belong to the leaf
           * set.
           *
           * But still this situation may happen during the conversion of an
           * \hmatrix to a different block cluster tree.</dd>
           * </dl>
           *
           * <strong>The general work flow for the agglomeration of a set of
           * full matrix blocks is as below.</strong>
           *
           * 1. Create a large full matrix on the heap and assemble all
           * submatrices into it which depends on the split mode of the block
           * cluster.
           *
           *   a. When it is \p CrossSplitMode, apply agglomeration of four full
           * submatrices.
           *
           *   b. When the split mode is \p HorizontalSplitMode, apply
           * agglomeration of two full submatrices via vertical stacking.
           *
           *   c. When the split mode is \p VerticalSplitMode, apply
           * agglomeration of two full submatrices via horizontal stacking.
           *
           * 2. Delete all submatrices associated with the current
           * \hmatrix and clear the \p std::vector storing
           * submatrix pointers.
           *
           * 3. Associate the new large full matrix with the current
           * \hmatrix.
           *
           * 4. Update the \hmatrix type as \p FullMatrix.
           *
           * <strong>About matrix assembly for \p CrossSplitMode</strong>
           *
           * Let the block cluster associated with the current
           * \hmatrix is \f$\tau \times \sigma\f$. Assume the
           * clusters are partitioned as \f$\tau = [\tau_1, \tau_2]\f$ and
           * \f$\sigma =
           * [\sigma_1, \sigma_2]\f$. Then the ordering of the child block
           * clusters are \f$\tau_1 \times \sigma_1, \tau_1 \times \sigma_2,
           * \tau_2 \times \sigma_1, \tau_2 \times \sigma_2\f$.
           */
          for (const HMatrix<spacedim, Number> *submatrix :
               hmat_block->submatrices)
            {
              Assert(submatrix->type == FullMatrixType,
                     ExcInvalidHMatrixType(submatrix->type));
            }

          /**
           * Build the map from the global DoF indices to the local row indices
           * of the current \hmatrix node, if necessary.
           */
          if (hmat_block->row_index_global_to_local_map.size() == 0)
            {
              build_index_set_global_to_local_map(
                *(hmat_block->row_indices),
                hmat_block->row_index_global_to_local_map);
            }

          /**
           * Build the map from the global DoF indices to the local column
           * indices of the current \hmatrix node, if necessary.
           */
          if (hmat_block->col_index_global_to_local_map.size() == 0)
            {
              build_index_set_global_to_local_map(
                *(hmat_block->col_indices),
                hmat_block->col_index_global_to_local_map);
            }

          LAPACKFullMatrixExt<Number> *fullmatrix;

          switch (hmat_block->bc_node->get_split_mode())
            {
              case CrossSplitMode:
                {
                  AssertDimension(hmat_block->submatrices.size(), 4);

                  fullmatrix = new LAPACKFullMatrixExt<Number>(
                    hmat_block->row_index_global_to_local_map,
                    hmat_block->col_index_global_to_local_map,
                    *(hmat_block->submatrices[0]->fullmatrix),
                    *(hmat_block->submatrices[0]->row_indices),
                    *(hmat_block->submatrices[0]->col_indices),
                    *(hmat_block->submatrices[1]->fullmatrix),
                    *(hmat_block->submatrices[1]->row_indices),
                    *(hmat_block->submatrices[1]->col_indices),
                    *(hmat_block->submatrices[2]->fullmatrix),
                    *(hmat_block->submatrices[2]->row_indices),
                    *(hmat_block->submatrices[2]->col_indices),
                    *(hmat_block->submatrices[3]->fullmatrix),
                    *(hmat_block->submatrices[3]->row_indices),
                    *(hmat_block->submatrices[3]->col_indices));

                  break;
                }
              case HorizontalSplitMode:
                {
                  AssertDimension(hmat_block->submatrices.size(), 2);

                  fullmatrix = new LAPACKFullMatrixExt<Number>(
                    hmat_block->row_index_global_to_local_map,
                    hmat_block->col_index_global_to_local_map,
                    *(hmat_block->submatrices[0]->fullmatrix),
                    *(hmat_block->submatrices[0]->row_indices),
                    *(hmat_block->submatrices[0]->col_indices),
                    *(hmat_block->submatrices[1]->fullmatrix),
                    *(hmat_block->submatrices[1]->row_indices),
                    *(hmat_block->submatrices[1]->col_indices),
                    true);

                  break;
                }
              case VerticalSplitMode:
                {
                  AssertDimension(hmat_block->submatrices.size(), 2);

                  fullmatrix = new LAPACKFullMatrixExt<Number>(
                    hmat_block->row_index_global_to_local_map,
                    hmat_block->col_index_global_to_local_map,
                    *(hmat_block->submatrices[0]->fullmatrix),
                    *(hmat_block->submatrices[0]->row_indices),
                    *(hmat_block->submatrices[0]->col_indices),
                    *(hmat_block->submatrices[1]->fullmatrix),
                    *(hmat_block->submatrices[1]->row_indices),
                    *(hmat_block->submatrices[1]->col_indices),
                    false);

                  break;
                }
              default:
                {
                  Assert(
                    false,
                    ExcMessage(
                      std::string("Invalid block cluster splitting mode: ") +
                      std::to_string(hmat_block->bc_node->get_split_mode())));
                }
            }

          for (HMatrix<spacedim, Number> *submatrix : hmat_block->submatrices)
            {
              if (submatrix != nullptr)
                {
                  delete submatrix;
                }
            }
          hmat_block->submatrices.clear();

          hmat_block->fullmatrix = fullmatrix;
          hmat_block->type       = FullMatrixType;
        }
      else
        {
          /**
           * When the current \hmatrix block belongs to the far
           * field set, perform the pairwise matrix agglomeration of rank-k
           * submatrices or full submatrices, which has been implemented into
           * the constructor of \p RkMatrix.
           */
          RkMatrix<Number> *rkmatrix;

          /**
           * Build the map from the global DoF indices to the local row indices
           * of the current \hmatrix node, if necessary.
           */
          if (hmat_block->row_index_global_to_local_map.size() == 0)
            {
              build_index_set_global_to_local_map(
                *(hmat_block->row_indices),
                hmat_block->row_index_global_to_local_map);
            }

          /**
           * Build the map from the global DoF indices to the local column
           * indices of the current \hmatrix node, if necessary.
           */
          if (hmat_block->col_index_global_to_local_map.size() == 0)
            {
              build_index_set_global_to_local_map(
                *(hmat_block->col_indices),
                hmat_block->col_index_global_to_local_map);
            }

          switch (hmat_block->bc_node->get_split_mode())
            {
              case CrossSplitMode:
                {
                  AssertDimension(hmat_block->submatrices.size(), 4);

                  if (hmat_block->submatrices[0]->type == RkMatrixType &&
                      hmat_block->submatrices[1]->type == RkMatrixType &&
                      hmat_block->submatrices[2]->type == RkMatrixType &&
                      hmat_block->submatrices[3]->type == RkMatrixType)
                    {
                      /**
                       * If the children of the current \hmatrix
                       * block are rank-k matrices, perform the pairwise rank-k
                       * matrix agglomeration directly.
                       */
                      rkmatrix = new RkMatrix<Number>(
                        fixed_rank_k,
                        hmat_block->row_index_global_to_local_map,
                        hmat_block->col_index_global_to_local_map,
                        *(hmat_block->submatrices[0]->rkmatrix),
                        *(hmat_block->submatrices[0]->row_indices),
                        *(hmat_block->submatrices[0]->col_indices),
                        *(hmat_block->submatrices[1]->rkmatrix),
                        *(hmat_block->submatrices[1]->row_indices),
                        *(hmat_block->submatrices[1]->col_indices),
                        *(hmat_block->submatrices[2]->rkmatrix),
                        *(hmat_block->submatrices[2]->row_indices),
                        *(hmat_block->submatrices[2]->col_indices),
                        *(hmat_block->submatrices[3]->rkmatrix),
                        *(hmat_block->submatrices[3]->row_indices),
                        *(hmat_block->submatrices[3]->col_indices));
                    }
                  else if (hmat_block->submatrices[0]->type == FullMatrixType &&
                           hmat_block->submatrices[1]->type == FullMatrixType &&
                           hmat_block->submatrices[2]->type == FullMatrixType &&
                           hmat_block->submatrices[3]->type == FullMatrixType)
                    {
                      /**
                       * If the children of the current \hmatrix
                       * block are full matrices, firstly convert all of them
                       * into rank-k matrices, then perform the pairwise rank-k
                       * matrix agglomeration.
                       */
                      RkMatrix<Number> rkmatrix11(
                        fixed_rank_k,
                        *(hmat_block->submatrices[0]->fullmatrix));
                      RkMatrix<Number> rkmatrix12(
                        fixed_rank_k,
                        *(hmat_block->submatrices[1]->fullmatrix));
                      RkMatrix<Number> rkmatrix21(
                        fixed_rank_k,
                        *(hmat_block->submatrices[2]->fullmatrix));
                      RkMatrix<Number> rkmatrix22(
                        fixed_rank_k,
                        *(hmat_block->submatrices[3]->fullmatrix));

                      rkmatrix = new RkMatrix<Number>(
                        fixed_rank_k,
                        hmat_block->row_index_global_to_local_map,
                        hmat_block->col_index_global_to_local_map,
                        rkmatrix11,
                        *(hmat_block->submatrices[0]->row_indices),
                        *(hmat_block->submatrices[0]->col_indices),
                        rkmatrix12,
                        *(hmat_block->submatrices[1]->row_indices),
                        *(hmat_block->submatrices[1]->col_indices),
                        rkmatrix21,
                        *(hmat_block->submatrices[2]->row_indices),
                        *(hmat_block->submatrices[2]->col_indices),
                        rkmatrix22,
                        *(hmat_block->submatrices[3]->row_indices),
                        *(hmat_block->submatrices[3]->col_indices));
                    }
                  else
                    {
                      /**
                       * Other cases are invalid.
                       */
                      Assert(false, ExcInternalError());
                    }

                  break;
                }
              case HorizontalSplitMode:
                {
                  AssertDimension(hmat_block->submatrices.size(), 2);

                  if (hmat_block->submatrices[0]->type == RkMatrixType &&
                      hmat_block->submatrices[1]->type == RkMatrixType)
                    {
                      /**
                       * If the children of the current \hmatrix
                       * block are rank-k matrices, perform the pairwise rank-k
                       * matrix agglomeration directly.
                       */
                      rkmatrix = new RkMatrix<Number>(
                        fixed_rank_k,
                        hmat_block->row_index_global_to_local_map,
                        hmat_block->col_index_global_to_local_map,
                        *(hmat_block->submatrices[0]->rkmatrix),
                        *(hmat_block->submatrices[0]->row_indices),
                        *(hmat_block->submatrices[0]->col_indices),
                        *(hmat_block->submatrices[1]->rkmatrix),
                        *(hmat_block->submatrices[1]->row_indices),
                        *(hmat_block->submatrices[1]->col_indices),
                        true);
                    }
                  else if (hmat_block->submatrices[0]->type == FullMatrixType &&
                           hmat_block->submatrices[1]->type == FullMatrixType)
                    {
                      /**
                       * If the children of the current \hmatrix
                       * block are full matrices, firstly convert all of them
                       * into rank-k matrices, then perform the pairwise rank-k
                       * matrix agglomeration.
                       */
                      RkMatrix<Number> rkmatrix1(
                        fixed_rank_k,
                        *(hmat_block->submatrices[0]->fullmatrix));
                      RkMatrix<Number> rkmatrix2(
                        fixed_rank_k,
                        *(hmat_block->submatrices[1]->fullmatrix));

                      rkmatrix = new RkMatrix<Number>(
                        fixed_rank_k,
                        hmat_block->row_index_global_to_local_map,
                        hmat_block->col_index_global_to_local_map,
                        rkmatrix1,
                        *(hmat_block->submatrices[0]->row_indices),
                        *(hmat_block->submatrices[0]->col_indices),
                        rkmatrix2,
                        *(hmat_block->submatrices[1]->row_indices),
                        *(hmat_block->submatrices[1]->col_indices),
                        true);
                    }
                  else
                    {
                      /**
                       * Other cases are invalid.
                       */
                      Assert(false, ExcInternalError());
                    }

                  break;
                }
              case VerticalSplitMode:
                {
                  AssertDimension(hmat_block->submatrices.size(), 2);

                  if (hmat_block->submatrices[0]->type == RkMatrixType &&
                      hmat_block->submatrices[1]->type == RkMatrixType)
                    {
                      /**
                       * If the children of the current \hmatrix
                       * block are rank-k matrices, perform the pairwise rank-k
                       * matrix agglomeration directly.
                       */
                      rkmatrix = new RkMatrix<Number>(
                        fixed_rank_k,
                        hmat_block->row_index_global_to_local_map,
                        hmat_block->col_index_global_to_local_map,
                        *(hmat_block->submatrices[0]->rkmatrix),
                        *(hmat_block->submatrices[0]->row_indices),
                        *(hmat_block->submatrices[0]->col_indices),
                        *(hmat_block->submatrices[1]->rkmatrix),
                        *(hmat_block->submatrices[1]->row_indices),
                        *(hmat_block->submatrices[1]->col_indices),
                        false);
                    }
                  else if (hmat_block->submatrices[0]->type == FullMatrixType &&
                           hmat_block->submatrices[1]->type == FullMatrixType)
                    {
                      /**
                       * If the children of the current \hmatrix
                       * block are full matrices, firstly convert all of them
                       * into rank-k matrices, then perform the pairwise rank-k
                       * matrix agglomeration.
                       */
                      RkMatrix<Number> rkmatrix1(
                        fixed_rank_k,
                        *(hmat_block->submatrices[0]->fullmatrix));
                      RkMatrix<Number> rkmatrix2(
                        fixed_rank_k,
                        *(hmat_block->submatrices[1]->fullmatrix));

                      rkmatrix = new RkMatrix<Number>(
                        fixed_rank_k,
                        hmat_block->row_index_global_to_local_map,
                        hmat_block->col_index_global_to_local_map,
                        rkmatrix1,
                        *(hmat_block->submatrices[0]->row_indices),
                        *(hmat_block->submatrices[0]->col_indices),
                        rkmatrix2,
                        *(hmat_block->submatrices[1]->row_indices),
                        *(hmat_block->submatrices[1]->col_indices),
                        false);
                    }
                  else
                    {
                      /**
                       * Other cases are invalid.
                       */
                      Assert(false, ExcInternalError());
                    }

                  break;
                }
              default:
                {
                  Assert(
                    false,
                    ExcMessage(
                      std::string("Invalid block cluster splitting mode: ") +
                      std::to_string(hmat_block->bc_node->get_split_mode())));
                }
            }

          for (HMatrix<spacedim, Number> *submatrix : hmat_block->submatrices)
            {
              if (submatrix != nullptr)
                {
                  delete submatrix;
                }
            }
          hmat_block->submatrices.clear();

          hmat_block->rkmatrix = rkmatrix;
          hmat_block->type     = RkMatrixType;
        }

      /**
       * Visualize the partition structure if a not-null pointer to the root
       * \hmatrix node and a a not-null pointer to a \p
       * calling_counter are provided.
       */
      if (hmat_root_block != nullptr && calling_counter != nullptr)
        {
          std::ofstream output_stream(
            output_file_base_name + std::to_string(*calling_counter) + ".dat");
          hmat_root_block->write_leaf_set(output_stream);
          output_stream.close();

          (*calling_counter) = (*calling_counter) + 1;
        }
    }
  /**
   *   </dd>
   * </dl>
   */
}


/**
 * Calculate the product of two \hmatrix nodes, where the second
 * one \p M2 has \p RkMatrixType and the result will also be a rank-k matrix.
 *
 * The arithmetic operation to be performed is
 * \f[
 * M = M_1 \cdot M_2 = M_1 (A B^T) = (M_1 A) B^T = A' B^T,
 * \f]
 * where \f$A' = M_1 A\f$ is calculated as a series of
 * \hmatrix-vector multiplications. For details,
 * \f[
 * M_1 A = M_1
 * \begin{bmatrix}
 * a_{\sigma,1} & \cdots & a_{\sigma,r}
 * \end{bmatrix} =
 * \begin{bmatrix}
 * M_1 a_{\sigma,1} & \cdots & M_1 a_{\sigma,r}
 * \end{bmatrix} =
 * \begin{bmatrix}
 * a'_{\tau,1} & \cdots & a'_{\tau,r}
 * \end{bmatrix}.
 * \f]
 * It can be seen that the formal rank \f$r\f$ of the result matrix \p M is the
 * same as that of \p M2.
 *
 * @param M1
 * @param M2
 * @param M
 */
template <int spacedim, typename Number = double>
void
h_rk_mmult(HMatrix<spacedim, Number> &M1,
           const RkMatrix<Number> &   M2,
           RkMatrix<Number> &         M)
{
  AssertDimension(M1.n, M2.m);

  /**
   * Create a temporary \p Vector storing a column \f$a_{\sigma,j}\f$ in the \p
   * A component of \p M2 and another \p Vector \f$a'_{\tau,j}\f$ storing the
   * matrix-vector product \f$M_1 \cdot a_{\sigma,j}\f$.
   */
  Vector<Number> col_vect_in_A(M2.A.m());
  Vector<Number> result_vect(M1.m);

  /**
   * Initialize the result rank-k matrix \p M with the formal rank of \p M2.
   * Its \p B component matrix is the same as that of \p M2.
   */
  M.reinit(M1.m, M2.n, M2.formal_rank);
  M.B = M2.B;

  /**
   * Build the map from global DoF indices to local matrix indices if necessary.
   */
  if (M1.row_index_global_to_local_map.size() == 0)
    {
      build_index_set_global_to_local_map(*(M1.row_indices),
                                          M1.row_index_global_to_local_map);
    }

  if (M1.col_index_global_to_local_map.size() == 0)
    {
      build_index_set_global_to_local_map(*(M1.col_indices),
                                          M1.col_index_global_to_local_map);
    }

  /**
   * Then we calculate the \p A component matrix of \p M, which is \p
   * M1*M2.A.
   */
  for (size_t j = 0; j < M2.formal_rank; j++)
    {
      M2.A.get_column(j, col_vect_in_A);
      result_vect = 0.;
      M1.vmult_local_vector(result_vect,
                            M1.row_index_global_to_local_map,
                            col_vect_in_A,
                            M1.col_index_global_to_local_map);

      /**
       * Fill the result vector into the \p A component matrix of \p M.
       */
      M.A.fill_col(j, result_vect);
    }
}


/**
 * Calculate the product of two \hmatrix nodes, where the second
 * one \p M2 has \p RkMatrixType and the result will also be a rank-k matrix.
 * This function is to be called by the matrix-matrix multiplication function.
 * @param M1
 * @param M2
 * @param M
 */
template <int spacedim, typename Number = double>
void
h_rk_mmult_for_h_h_mmult(HMatrix<spacedim, Number> *      M1,
                         const HMatrix<spacedim, Number> *M2,
                         HMatrix<spacedim, Number> *      M,
                         bool is_M1M2_last_in_M_Sigma_P = true)
{
  Assert(M2->type == RkMatrixType, ExcInvalidHMatrixType(M2->type));
  Assert(M2->rkmatrix, ExcInternalError());
  //  Assert(*(M1->col_indices) == *(M2->row_indices),
  //         ExcMessage("Incompatible index sets during matrix
  //         multiplication"));
  AssertDimension(M1->n, M2->m);

  // DEBUG
  //  print_h_h_submatrix_mmult_accessor(std::cout, "M1", *M1, "M2", *M2);

  if (is_M1M2_last_in_M_Sigma_P)
    {
      M->Sigma_P.pop_back();
    }
  else
    {
      M->remove_hmat_pair_from_mm_product_list(M1, M2);
    }

  RkMatrix<Number> *M_rk = new RkMatrix<Number>();
  h_rk_mmult(*M1, *(M2->rkmatrix), *M_rk);
  M->Sigma_R.push_back(M_rk);

  //  // DEBUG
  //  LAPACKFullMatrixExt<Number> M_rk_to_full;
  //  M_rk->convertToFullMatrix(M_rk_to_full);
  //  M_rk_to_full.print_formatted(std::cout, 8, false, 16, "0");

  AssertDimension(M->Sigma_F.size(), 0);
  M->type = RkMatrixType;
}


/**
 * Calculate the product of two \hmatrix nodes, where the first
 * one \p M1 has \p RkMatrixType and the result is also a rank-k matrix.
 *
 * The arithmetic operation to be performed is
 * \f[
 * M = M_1 \cdot M_2 = (A B^T) M_2 = A (B^T M_2) = A B'^T,
 * \f]
 * where \f$B' = M_2^T B\f$ is calculated as a series of
 * transposed \hmatrix-vector multiplications. For details,
 * \f[
 * M_2^T B = M_2^T
 * \begin{bmatrix}
 * b_{\sigma,1} & \cdots & b_{\sigma,r}
 * \end{bmatrix} =
 * \begin{bmatrix}
 * M_2^T b_{\sigma,1} & \cdots & M_2^T b_{\sigma,r}
 * \end{bmatrix} =
 * \begin{bmatrix}
 * b'_{\rho,1} & \cdots & b'_{\rho,r}
 * \end{bmatrix}.
 * \f]
 * It can be seen that the formal rank \f$r\f$ of the result matrix \p M is the
 * same as that of \p M1.
 *
 * @param M1
 * @param M2
 * @param M
 */
template <int spacedim, typename Number = double>
void
rk_h_mmult(const RkMatrix<Number> &   M1,
           HMatrix<spacedim, Number> &M2,
           RkMatrix<Number> &         M)
{
  AssertDimension(M1.n, M2.m);

  /**
   * Create a temporary \p Vector storing a column \f$b_{\sigma,j}\f$ in the \p
   * B component of \p M1 and another \p Vector \f$b'_{\rho,j}\f$ storing the
   * matrix-vector product \f$M_2^T \cdot b_{\sigma,j}\f$.
   */
  Vector<Number> col_vect_in_B(M1.B.m());
  Vector<Number> result_vect(M2.n);

  /**
   * Initialize the result rank-k matrix \p M with the formal rank of \p M1_rk.
   * Its \p A component matrix is the same as that of \p M1_rk.
   */
  M.reinit(M1.m, M2.n, M1.formal_rank);
  M.A = M1.A;

  /**
   * Build the map from global DoF indices to local matrix indices if necessary.
   */
  if (M2.row_index_global_to_local_map.size() == 0)
    {
      build_index_set_global_to_local_map(*(M2.row_indices),
                                          M2.row_index_global_to_local_map);
    }

  if (M2.col_index_global_to_local_map.size() == 0)
    {
      build_index_set_global_to_local_map(*(M2.col_indices),
                                          M2.col_index_global_to_local_map);
    }

  /**
   * Then we calculate the \p B component matrix of \p M, which is \p
   * M2^T*M1_rk.B.
   */
  for (size_t j = 0; j < M.formal_rank; j++)
    {
      M1.B.get_column(j, col_vect_in_B);
      result_vect = 0.;
      M2.Tvmult_local_vector(result_vect,
                             M2.col_index_global_to_local_map,
                             col_vect_in_B,
                             M2.row_index_global_to_local_map);

      /**
       * Fill the result vector into the \p B component matrix of \p M.
       */
      M.B.fill_col(j, result_vect);
    }
}


/**
 * Calculate the product of two \hmatrix nodes, where the first
 * one \p M1 has \p RkMatrixType and the result will also be a rank-k matrix.
 * This function is to be called by the matrix-matrix multiplication function.
 * @param M1
 * @param M2
 * @param M
 */
template <int spacedim, typename Number = double>
void
rk_h_mmult_for_h_h_mmult(const HMatrix<spacedim, Number> *M1,
                         HMatrix<spacedim, Number> *      M2,
                         HMatrix<spacedim, Number> *      M,
                         bool is_M1M2_last_in_M_Sigma_P = true)
{
  Assert(M1->type == RkMatrixType, ExcInvalidHMatrixType(M1->type));
  Assert(M1->rkmatrix, ExcInternalError());
  //  Assert(*(M1->col_indices) == *(M2->row_indices),
  //         ExcMessage("Incompatible index sets during matrix
  //         multiplication"));
  AssertDimension(M1->n, M2->m);

  //  // DEBUG
  //  print_h_h_submatrix_mmult_accessor(std::cout, "M1", *M1, "M2", *M2);

  if (is_M1M2_last_in_M_Sigma_P)
    {
      M->Sigma_P.pop_back();
    }
  else
    {
      M->remove_hmat_pair_from_mm_product_list(M1, M2);
    }

  RkMatrix<Number> *M_rk = new RkMatrix<Number>();
  rk_h_mmult(*(M1->rkmatrix), *M2, *M_rk);
  M->Sigma_R.push_back(M_rk);

  //  // DEBUG
  //  LAPACKFullMatrixExt<Number> M_rk_to_full;
  //  M_rk->convertToFullMatrix(M_rk_to_full);
  //  M_rk_to_full.print_formatted(std::cout, 8, false, 16, "0");

  AssertDimension(M->Sigma_F.size(), 0);
  M->type = RkMatrixType;
}


/**
 * Calculate the product of two \hmatrix nodes, where the second
 * one is a full matrix and the result is also represented as a full matrix
 * because the associated block cluster node \f$\tau\times\rho\f$ is small.
 * @param M1
 * @param M2
 * @param M
 */
template <int spacedim, typename Number = double>
void
h_f_mmult(HMatrix<spacedim, Number> &        M1,
          const LAPACKFullMatrixExt<Number> &M2,
          LAPACKFullMatrixExt<Number> &      M)
{
  AssertDimension(M1.n, M2.m());

  Vector<Number> col_vect_in_M2(M2.m());
  Vector<Number> result_vect(M1.m);

  const typename LAPACKFullMatrixExt<Number>::size_type n_rows = M1.m;
  const typename LAPACKFullMatrixExt<Number>::size_type n_cols = M2.n();

  /**
   * Build the map from global DoF indices to local matrix indices if necessary.
   */
  if (M1.row_index_global_to_local_map.size() == 0)
    {
      build_index_set_global_to_local_map(*(M1.row_indices),
                                          M1.row_index_global_to_local_map);
    }

  if (M1.col_index_global_to_local_map.size() == 0)
    {
      build_index_set_global_to_local_map(*(M1.col_indices),
                                          M1.col_index_global_to_local_map);
    }

  M.reinit(n_rows, n_cols);
  for (typename LAPACKFullMatrixExt<Number>::size_type j = 0; j < n_cols; j++)
    {
      M2.get_column(j, col_vect_in_M2);
      result_vect = 0.;
      M1.vmult_local_vector(result_vect,
                            M1.row_index_global_to_local_map,
                            col_vect_in_M2,
                            M1.col_index_global_to_local_map);
      M.fill_col(j, result_vect);
    }
}


/**
 * Calculate the product of two \hmatrix nodes, where the second
 * one is a full matrix and the result is represented as a rank-k matrix because
 * the associated block cluster is large.
 *
 * The second matrix \p M2 will be firstly converted to a rank-k matrix. Then
 * its multiplication with \p M1 will be carried by calling \p h_rk_mmult. Since
 * the conversion from a full matrix to a rank-k matrix will modify the original
 * data, a copy of \p M2 will be created.
 * @param M1
 * @param M2
 * @param M
 */
template <int spacedim, typename Number = double>
void
h_f_mmult(HMatrix<spacedim, Number> &        M1,
          const LAPACKFullMatrixExt<Number> &M2,
          RkMatrix<Number> &                 M)
{
  AssertDimension(M1.n, M2.m());

  /**
   * Create a local copy of the full matrix \p M2.
   */
  LAPACKFullMatrixExt<Number> M2_copy(M2);

  /**
   * Convert the full matrix \p M2 to a rank-k matrix.
   */
  RkMatrix<Number> M2_rk(M2_copy);

  h_rk_mmult(M1, M2_rk, M);
}


template <int spacedim, typename Number = double>
void
h_f_mmult_for_h_h_mmult(HMatrix<spacedim, Number> *      M1,
                        const HMatrix<spacedim, Number> *M2,
                        HMatrix<spacedim, Number> *      M,
                        bool is_M1M2_last_in_M_Sigma_P = true)
{
  Assert(M2->type == FullMatrixType, ExcInvalidHMatrixType(M2->type));
  Assert(M2->fullmatrix, ExcInternalError());
  //  Assert(*(M1->col_indices) == *(M2->row_indices),
  //         ExcMessage("Incompatible index sets during matrix
  //         multiplication"));
  AssertDimension(M1->n, M2->m);

  //  // DEBUG
  //  print_h_h_submatrix_mmult_accessor(std::cout, "M1", *M1, "M2", *M2);

  if (is_M1M2_last_in_M_Sigma_P)
    {
      M->Sigma_P.pop_back();
    }
  else
    {
      M->remove_hmat_pair_from_mm_product_list(M1, M2);
    }

  if (M->bc_node->get_data_reference().get_is_near_field())
    {
      /**
       * Full matrix is returned.
       */
      LAPACKFullMatrixExt<Number> *M_full = new LAPACKFullMatrixExt<Number>();
      h_f_mmult(*M1, *(M2->fullmatrix), *M_full);
      M->Sigma_F.push_back(M_full);

      //      // DEBUG
      //      M_full->print_formatted(std::cout, 8, false, 16, "0");

      AssertDimension(M->Sigma_R.size(), 0);
      M->type = FullMatrixType;
    }
  else
    {
      /**
       * Rank-k matrix is returned.
       */
      RkMatrix<Number> *M_rk = new RkMatrix<Number>();
      h_f_mmult(*M1, *(M2->fullmatrix), *M_rk);
      M->Sigma_R.push_back(M_rk);

      //      // DEBUG
      //      LAPACKFullMatrixExt<Number> M_rk_to_full;
      //      M_rk->convertToFullMatrix(M_rk_to_full);
      //      M_rk_to_full.print_formatted(std::cout, 8, false, 16, "0");

      AssertDimension(M->Sigma_F.size(), 0);
      M->type = RkMatrixType;
    }
}


template <int spacedim, typename Number = double>
void
f_h_mmult(const LAPACKFullMatrixExt<Number> &M1,
          HMatrix<spacedim, Number> &        M2,
          LAPACKFullMatrixExt<Number> &      M)
{
  AssertDimension(M1.n(), M2.m);

  Vector<Number> row_vect_in_M1(M1.n());
  Vector<Number> result_vect(M2.n);

  const typename LAPACKFullMatrixExt<Number>::size_type n_rows = M1.m();
  const typename LAPACKFullMatrixExt<Number>::size_type n_cols = M2.n;

  /**
   * Build the map from global DoF indices to local matrix indices if necessary.
   */
  if (M2.row_index_global_to_local_map.size() == 0)
    {
      build_index_set_global_to_local_map(*(M2.row_indices),
                                          M2.row_index_global_to_local_map);
    }

  if (M2.col_index_global_to_local_map.size() == 0)
    {
      build_index_set_global_to_local_map(*(M2.col_indices),
                                          M2.col_index_global_to_local_map);
    }

  M.reinit(n_rows, n_cols);
  for (typename LAPACKFullMatrixExt<Number>::size_type i = 0; i < n_rows; i++)
    {
      M1.get_row(i, row_vect_in_M1);
      result_vect = 0.;
      M2.Tvmult_local_vector(result_vect,
                             M2.col_index_global_to_local_map,
                             row_vect_in_M1,
                             M2.row_index_global_to_local_map);
      M.fill_row(i, result_vect);
    }
}


template <int spacedim, typename Number>
void
f_h_mmult(const LAPACKFullMatrixExt<Number> &M1,
          HMatrix<spacedim, Number> &        M2,
          RkMatrix<Number> &                 M)
{
  AssertDimension(M1.n(), M2.m);

  LAPACKFullMatrixExt<Number> M1_copy(M1);
  RkMatrix<Number>            M1_rk(M1_copy);
  rk_h_mmult(M1_rk, M2, M);
}


template <int spacedim, typename Number = double>
void
f_h_mmult_for_h_h_mmult(const HMatrix<spacedim, Number> *M1,
                        HMatrix<spacedim, Number> *      M2,
                        HMatrix<spacedim, Number> *      M,
                        bool is_M1M2_last_in_M_Sigma_P = true)
{
  Assert(M1->type == FullMatrixType, ExcInvalidHMatrixType(M1->type));
  Assert(M1->fullmatrix, ExcInternalError());
  //  Assert(*(M1->col_indices) == *(M2->row_indices),
  //         ExcMessage("Incompatible index sets during matrix
  //         multiplication"));
  AssertDimension(M1->n, M2->m);

  // DEBUG
  // print_h_h_submatrix_mmult_accessor(std::cout, "M1", *M1, "M2", *M2);

  if (is_M1M2_last_in_M_Sigma_P)
    {
      M->Sigma_P.pop_back();
    }
  else
    {
      M->remove_hmat_pair_from_mm_product_list(M1, M2);
    }

  if (M->bc_node->get_data_reference().get_is_near_field())
    {
      /**
       * Full matrix is returned.
       */
      LAPACKFullMatrixExt<Number> *M_full = new LAPACKFullMatrixExt<Number>();
      f_h_mmult(*(M1->fullmatrix), *M2, *M_full);
      M->Sigma_F.push_back(M_full);

      //      // DEBUG
      //      M_full->print_formatted(std::cout, 8, false, 16, "0");

      AssertDimension(M->Sigma_R.size(), 0);
      M->type = FullMatrixType;
    }
  else
    {
      /**
       * Rank-k matrix is returned.
       */
      RkMatrix<Number> *M_rk = new RkMatrix<Number>();
      f_h_mmult(*(M1->fullmatrix), *M2, *M_rk);
      M->Sigma_R.push_back(M_rk);

      //      // DEBUG
      //      LAPACKFullMatrixExt<Number> M_rk_to_full;
      //      M_rk->convertToFullMatrix(M_rk_to_full);
      //      M_rk_to_full.print_formatted(std::cout, 8, false, 16, "0");

      AssertDimension(M->Sigma_F.size(), 0);
      M->type = RkMatrixType;
    }
}


template <int spacedim, typename Number = double>
void
h_h_mmult_phase1_recursion(HMatrix<spacedim, Number> *         M,
                           BlockClusterTree<spacedim, Number> &Tind)
{
  M->h_h_mmult_reduction();

  if (M->Sigma_P.size() > 0)
    {
      /**
       * There are still multiplication subtasks stored in \p Sigma_P to be
       * handled recursively.
       */
      TreeNodeSplitMode split_mode_of_mm =
        M->determine_mm_split_mode_from_Sigma_P();

      switch (split_mode_of_mm)
        {
          case HorizontalSplitMode:
            {
              M->h_h_mmult_horizontal_split(Tind);

              break;
            }
          case VerticalSplitMode:
            {
              M->h_h_mmult_vertical_split(Tind);

              break;
            }
          case CrossSplitMode:
            {
              M->h_h_mmult_cross_split(Tind);

              break;
            }
          default:
            {
              Assert(
                false,
                ExcMessage(
                  "Inconsistent case met during H-matrix MM multiplication"));

              break;
            }
        }

      /**
       * After previous reduction and splitting, the matrix multiplication for
       * the current \hmatrix node should be replaced by the
       * multiplication subtasks for submatrices. These subtasks are recorded as
       * \hmatrix node pairs which are stored in
       * \f$\Sigma_b^P\f$ of the submatrices.
       */
      AssertDimension(M->Sigma_P.size(), 0);
      for (HMatrix<spacedim, Number> *submatrix : M->submatrices)
        {
          h_h_mmult_phase1_recursion(submatrix, Tind);
        }
    }
}


template <int spacedim, typename Number = double>
void
h_h_mmult_phase2(HMatrix<spacedim, Number> &         M,
                 BlockClusterTree<spacedim, Number> &target_bc_tree,
                 const unsigned int                  fixed_rank)
{
  /**
   * Collect terms in \p Sigma_R and \p Sigma_F for the leaf nodes.
   */
  for (HMatrix<spacedim, Number> *hmat : M.leaf_set)
    {
      /**
       * Here we make sure that \hmatrix pairs in the list
       * \f$\Sigma_b^P\f$ have all been processed and erased.
       */
      AssertDimension(hmat->Sigma_P.size(), 0);

      if (hmat->Sigma_R.size() > 0 && hmat->Sigma_F.size() == 0)
        {
          Assert(hmat->type == RkMatrixType, ExcInvalidHMatrixType(hmat->type));

          /**
           * Perform pairwise formatted addition for the list of rank-k
           * matrices.
           */
          hmat->rkmatrix = hmat->Sigma_R[0];
          for (size_t i = 1; i < hmat->Sigma_R.size(); i++)
            {
              hmat->rkmatrix->add(*(hmat->Sigma_R[i]), fixed_rank);
              delete hmat->Sigma_R[i];
              hmat->Sigma_R[i] = nullptr;
            }
          hmat->Sigma_R.clear();
        }
      else if (hmat->Sigma_R.size() == 0 && hmat->Sigma_F.size() > 0)
        {
          Assert(hmat->type == FullMatrixType,
                 ExcInvalidHMatrixType(hmat->type));

          hmat->fullmatrix = hmat->Sigma_F[0];
          for (size_t i = 1; i < hmat->Sigma_F.size(); i++)
            {
              hmat->fullmatrix->add(*(hmat->Sigma_F[i]));
              delete hmat->Sigma_F[i];
              hmat->Sigma_F[i] = nullptr;
            }
          hmat->Sigma_F.clear();
        }
      else
        {
          Assert(false, ExcInternalError());
        }
    }

  /**
   * Distribute matrices stored in \f$\Sigma_b^R\f$ and \f$\Sigma_b^F\f$ of each
   * non-leaf node to its leaf nodes.
   */
  M.distribute_all_non_leaf_nodes_sigma_r_and_f_to_leaves();

  /**
   * Convert the calculated product matrix to the specified matrix structure.
   */
  M.convert_between_different_block_cluster_trees(M.Tind,
                                                  target_bc_tree,
                                                  fixed_rank);
}


/**
 * Shallow copy an \hmatrix node into the target node, i.e. the
 * copy is limited within the current node without recursion into its
 * descendants. This function will be called by \p copy_hmatrix.
 *
 * N.B. Do not copy the list \p submatrices from the source submatrix,
 * because newly created child matrices will be pushed back into this
 * list.
 *
 * Do not copy the list \p leaf_set. After the whole \hmatrix
 * hierarchy has been constructed, the leaf set will be built in the
 * constructor.
 *
 * Do not copy the working data: \p Sigma_P, \p Sigma_F, \p Sigma_R and \p
 * Tind.
 * @param hmat_dst
 * @param hmat_src
 */
template <int spacedim, typename Number = double>
void
copy_hmatrix_node(HMatrix<spacedim, Number> &      hmat_dst,
                  const HMatrix<spacedim, Number> &hmat_src)
{
  hmat_dst.type = hmat_src.type;

  /**
   * Copy the rank-k matrix in the source submatrix if it is not \p NULL.
   */
  if (hmat_src.rkmatrix != nullptr)
    {
      hmat_dst.rkmatrix = new RkMatrix<Number>(*(hmat_src.rkmatrix));
    }
  else
    {
      hmat_dst.rkmatrix = nullptr;
    }

  /**
   * Copy the full matrix in the source submatrix if it is not \p NULL.
   */
  if (hmat_src.fullmatrix != nullptr)
    {
      hmat_dst.fullmatrix =
        new LAPACKFullMatrixExt<Number>(*(hmat_src.fullmatrix));
    }
  else
    {
      hmat_dst.fullmatrix = nullptr;
    }

  hmat_dst.bc_node     = hmat_src.bc_node;
  hmat_dst.row_indices = hmat_src.row_indices;
  hmat_dst.col_indices = hmat_src.col_indices;
  hmat_dst.row_index_global_to_local_map =
    hmat_src.row_index_global_to_local_map;
  hmat_dst.col_index_global_to_local_map =
    hmat_src.col_index_global_to_local_map;
  hmat_dst.m = hmat_src.m;
  hmat_dst.n = hmat_src.n;
}


/**
 * Deep copy an \hmatrix node into the target node, i.e. the
 * copy is limited within the current node without recursion into its
 * descendants.
 * @param hmat_dst
 * @param hmat_src
 */
template <int spacedim, typename Number = double>
void
copy_hmatrix_node(HMatrix<spacedim, Number> & hmat_dst,
                  HMatrix<spacedim, Number> &&hmat_src)
{
  hmat_dst.type        = hmat_src.type;
  hmat_dst.submatrices = hmat_src.submatrices;
  hmat_dst.leaf_set    = hmat_src.leaf_set;
  hmat_dst.rkmatrix    = hmat_src.rkmatrix;
  hmat_dst.fullmatrix  = hmat_src.fullmatrix;
  hmat_dst.bc_node     = hmat_src.bc_node;
  hmat_dst.row_indices = hmat_src.row_indices;
  hmat_dst.col_indices = hmat_src.col_indices;
  hmat_dst.row_index_global_to_local_map =
    hmat_src.row_index_global_to_local_map;
  hmat_dst.col_index_global_to_local_map =
    hmat_src.col_index_global_to_local_map;
  hmat_dst.m       = hmat_src.m;
  hmat_dst.n       = hmat_src.n;
  hmat_dst.Sigma_P = hmat_src.Sigma_P;
  hmat_dst.Sigma_R = hmat_src.Sigma_R;
  hmat_dst.Sigma_F = hmat_src.Sigma_F;
  hmat_dst.Tind    = std::move(hmat_src.Tind);
}


/**
 * Recursively copy an \hmatrix into the target matrix.
 * @param M_dst
 * @param M_src
 */
template <int spacedim, typename Number = double>
void
copy_hmatrix(HMatrix<spacedim, Number> &      hmat_dst,
             const HMatrix<spacedim, Number> &hmat_src)
{
  /**
   * Copy the current \hmatrix node.
   */
  copy_hmatrix_node(hmat_dst, hmat_src);

  /**
   * Recursively copy child \hmatrix nodes.
   */
  for (HMatrix<spacedim, Number> *submatrix : hmat_src.submatrices)
    {
      Assert(submatrix, ExcInternalError());

      /**
       * Create a corresponding child \hmatrix node on the heap
       * and push it back into the \p submatrices list of the current
       * \hmatrix node.
       */
      HMatrix<spacedim, Number> *child_hmat = new HMatrix<spacedim, Number>();
      copy_hmatrix(*child_hmat, *submatrix);
      hmat_dst.submatrices.push_back(child_hmat);
    }
}


template <int spacedim, typename Number = double>
void
print_h_submatrix_accessor(std::ostream &                   out,
                           const std::string &              name,
                           const HMatrix<spacedim, Number> &M)
{
  out << name + std::string("([") << std::flush;
  print_vector_values(out, *(M.row_indices), ",", false);
  out << "],[" << std::flush;
  print_vector_values(out, *(M.col_indices), ",", false);
  out << "])" << std::endl;
}


template <int spacedim, typename Number = double>
void
print_h_h_submatrix_mmult_accessor(std::ostream &                   out,
                                   const std::string &              name1,
                                   const HMatrix<spacedim, Number> &M1,
                                   const std::string &              name2,
                                   const HMatrix<spacedim, Number> &M2)
{
  out << name1 + std::string("([") << std::flush;
  print_vector_indices(out, *(M1.row_indices), ",", false, false);
  out << "],[" << std::flush;
  print_vector_indices(out, *(M1.col_indices), ",", false, false);
  out << "]) * " << name2 + std::string("([") << std::flush;
  print_vector_indices(out, *(M2.row_indices), ",", false, false);
  out << "],[" << std::flush;
  print_vector_indices(out, *(M2.col_indices), ",", false, false);
  out << "])" << std::endl;
}


template <int spacedim, typename Number>
HMatrix<spacedim, Number>::HMatrix()
  : type(UndefinedMatrixType)
  , submatrices(0)
  , leaf_set(0)
  , rkmatrix(nullptr)
  , fullmatrix(nullptr)
  , bc_node(nullptr)
  , row_indices(nullptr)
  , col_indices(nullptr)
  , row_index_global_to_local_map()
  , col_index_global_to_local_map()
  , m(0)
  , n(0)
  , Tind()
  , Sigma_P(0)
  , Sigma_R(0)
  , Sigma_F(0)
{}


template <int spacedim, typename Number>
HMatrix<spacedim, Number>::HMatrix(
  const BlockClusterTree<spacedim, Number> &bct,
  const unsigned int                        fixed_rank_k)
  : type(UndefinedMatrixType)
  , submatrices(0)
  , leaf_set(0)
  , rkmatrix(nullptr)
  , fullmatrix(nullptr)
  , bc_node(nullptr)
  , row_indices(nullptr)
  , col_indices(nullptr)
  , row_index_global_to_local_map()
  , col_index_global_to_local_map()
  , m(0)
  , n(0)
  , Tind()
  , Sigma_P(0)
  , Sigma_R(0)
  , Sigma_F(0)
{
  InitAndCreateHMatrixChildren(this, bct.get_root(), fixed_rank_k);
  build_leaf_set();
}


template <int spacedim, typename Number>
HMatrix<spacedim, Number>::HMatrix(
  typename BlockClusterTree<spacedim, Number>::node_const_pointer_type bc_node,
  const unsigned int fixed_rank_k)
  : type(UndefinedMatrixType)
  , submatrices(0)
  , leaf_set(0)
  , rkmatrix(nullptr)
  , fullmatrix(nullptr)
  , bc_node(nullptr)
  , row_indices(nullptr)
  , col_indices(nullptr)
  , row_index_global_to_local_map()
  , col_index_global_to_local_map()
  , m(0)
  , n(0)
  , Tind()
  , Sigma_P(0)
  , Sigma_R(0)
  , Sigma_F(0)
{
  InitAndCreateHMatrixChildren(this, bc_node, fixed_rank_k);
  build_leaf_set();
}


template <int spacedim, typename Number>
HMatrix<spacedim, Number>::HMatrix(
  const BlockClusterTree<spacedim, Number> &bct,
  const LAPACKFullMatrixExt<Number> &       M,
  const unsigned int                        fixed_rank_k)
  : type(UndefinedMatrixType)
  , submatrices(0)
  , leaf_set(0)
  , rkmatrix(nullptr)
  , fullmatrix(nullptr)
  , bc_node(nullptr)
  , row_indices(nullptr)
  , col_indices(nullptr)
  , row_index_global_to_local_map()
  , col_index_global_to_local_map()
  , m(0)
  , n(0)
  , Tind()
  , Sigma_P(0)
  , Sigma_R(0)
  , Sigma_F(0)
{
  InitAndCreateHMatrixChildren(this, bct.get_root(), fixed_rank_k, M);
  build_leaf_set();
}


template <int spacedim, typename Number>
HMatrix<spacedim, Number>::HMatrix(
  typename BlockClusterTree<spacedim, Number>::node_const_pointer_type bc_node,
  const LAPACKFullMatrixExt<Number> &                                  M,
  const unsigned int fixed_rank_k)
  : type(UndefinedMatrixType)
  , submatrices(0)
  , leaf_set(0)
  , rkmatrix(nullptr)
  , fullmatrix(nullptr)
  , bc_node(nullptr)
  , row_indices(nullptr)
  , col_indices(nullptr)
  , row_index_global_to_local_map()
  , col_index_global_to_local_map()
  , m(0)
  , n(0)
  , Tind()
  , Sigma_P(0)
  , Sigma_R(0)
  , Sigma_F(0)
{
  InitAndCreateHMatrixChildren(this, bc_node, fixed_rank_k, M);
  build_leaf_set();
}


template <int spacedim, typename Number>
HMatrix<spacedim, Number>::HMatrix(
  typename BlockClusterTree<spacedim, Number>::node_const_pointer_type bc_node,
  HMatrix<spacedim, Number> &&                                         H)
  : type(UndefinedMatrixType)
  , submatrices(0)
  , leaf_set(0)
  , rkmatrix(nullptr)
  , fullmatrix(nullptr)
  , bc_node(nullptr)
  , row_indices(nullptr)
  , col_indices(nullptr)
  , row_index_global_to_local_map()
  , col_index_global_to_local_map()
  , m(0)
  , n(0)
  , Tind(std::move(H.Tind))
  , Sigma_P(0)
  , Sigma_R(0)
  , Sigma_F(0)
{
  InitAndCreateHMatrixChildren(this, bc_node, std::move(H));
  build_leaf_set();
}


template <int spacedim, typename Number>
HMatrix<spacedim, Number>::HMatrix(
  const BlockClusterTree<spacedim, Number> &bct,
  HMatrix<spacedim, Number> &&              H)
  : type(UndefinedMatrixType)
  , submatrices(0)
  , leaf_set(0)
  , rkmatrix(nullptr)
  , fullmatrix(nullptr)
  , bc_node(nullptr)
  , row_indices(nullptr)
  , col_indices(nullptr)
  , row_index_global_to_local_map()
  , col_index_global_to_local_map()
  , m(0)
  , n(0)
  , Tind(std::move(H.Tind))
  , Sigma_P(0)
  , Sigma_R(0)
  , Sigma_F(0)
{
  InitAndCreateHMatrixChildren(this, bct.get_root(), std::move(H));
  build_leaf_set();
}


template <int spacedim, typename Number>
HMatrix<spacedim, Number>::HMatrix(const HMatrix<spacedim, Number> &H)
  : type(UndefinedMatrixType)
  , submatrices(0)
  , leaf_set(0)
  , rkmatrix(nullptr)
  , fullmatrix(nullptr)
  , bc_node(nullptr)
  , row_indices(nullptr)
  , col_indices(nullptr)
  , row_index_global_to_local_map()
  , col_index_global_to_local_map()
  , m(0)
  , n(0)
  , Tind()
  , Sigma_P(0)
  , Sigma_R(0)
  , Sigma_F(0)
{
  copy_hmatrix(*this, H);
  build_leaf_set();
}


template <int spacedim, typename Number>
HMatrix<spacedim, Number>::HMatrix(HMatrix<spacedim, Number> &&H)
  : type(H.type)
  , submatrices(H.submatrices)
  , leaf_set(H.leaf_set)
  , rkmatrix(H.rkmatrix)
  , fullmatrix(H.fullmatrix)
  , bc_node(H.bc_node)
  , row_indices(H.row_indices)
  , col_indices(H.col_indices)
  , row_index_global_to_local_map(H.row_index_global_to_local_map)
  , col_index_global_to_local_map(H.col_index_global_to_local_map)
  , m(H.m)
  , n(H.n)
  , Tind(std::move(H.Tind))
  , Sigma_P(H.Sigma_P)
  , Sigma_R(H.Sigma_R)
  , Sigma_F(H.Sigma_F)
{
  H.clear_hmat_node();
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::reinit(const BlockClusterTree<spacedim, Number> &bct,
                                  const unsigned int fixed_rank_k)
{
  release();
  InitAndCreateHMatrixChildren(this, bct.get_root(), fixed_rank_k);
  build_leaf_set();
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::reinit(
  typename BlockClusterTree<spacedim, Number>::node_const_pointer_type bc_node,
  const unsigned int fixed_rank_k)
{
  release();
  InitAndCreateHMatrixChildren(this, bc_node, fixed_rank_k);
  build_leaf_set();
}


template <int spacedim, typename Number>
HMatrix<spacedim, Number> &
HMatrix<spacedim, Number>::operator=(HMatrix<spacedim, Number> &&H)
{
  release();
  copy_hmatrix_node((*this), std::move(H));
  H.clear_hmat_node();

  return (*this);
}


template <int spacedim, typename Number>
HMatrix<spacedim, Number> &
HMatrix<spacedim, Number>::operator=(const HMatrix<spacedim, Number> &H)
{
  release();
  copy_hmatrix(*this, H);
  build_leaf_set();

  return (*this);
}


template <int spacedim, typename Number>
template <typename MatrixType>
void
HMatrix<spacedim, Number>::convertToFullMatrix(MatrixType &M) const
{
  M.reinit(m, n);
  _convertToFullMatrix(M);
}


template <int spacedim, typename Number>
template <typename MatrixType>
void
HMatrix<spacedim, Number>::_convertToFullMatrix(MatrixType &M) const
{
  LAPACKFullMatrixExt<Number> matrix_block;

  switch (type)
    {
      case FullMatrixType:
        Assert(fullmatrix, ExcInternalError());

        for (size_type i = 0; i < m; i++)
          {
            for (size_type j = 0; j < n; j++)
              {
                M(row_indices->at(i), col_indices->at(j)) = (*fullmatrix)(i, j);
              }
          }

        break;
      case RkMatrixType:
        Assert(rkmatrix, ExcInternalError());

        rkmatrix->convertToFullMatrix(matrix_block);

        for (size_type i = 0; i < m; i++)
          {
            for (size_type j = 0; j < n; j++)
              {
                M(row_indices->at(i), col_indices->at(j)) = matrix_block(i, j);
              }
          }

        break;
      case HierarchicalMatrixType:
        for (HMatrix *submatrix : submatrices)
          {
            submatrix->_convertToFullMatrix(M);
          }

        break;
      default:
        Assert(false, ExcInvalidHMatrixType(type));
    }
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::_build_leaf_set(
  std::vector<HMatrix *> &total_leaf_set) const
{
  switch (type)
    {
      case FullMatrixType:
        {
          total_leaf_set.push_back(const_cast<HMatrix *>(this));

          break;
        }
      case RkMatrixType:
        {
          total_leaf_set.push_back(const_cast<HMatrix *>(this));

          break;
        }
      case HierarchicalMatrixType:
        {
          for (HMatrix *submatrix : submatrices)
            {
              submatrix->_build_leaf_set(total_leaf_set);
            }

          break;
        }
      default:
        {
          Assert(false, ExcInvalidHMatrixType(type));

          break;
        }
    }
}


template <int spacedim, typename Number>
void
HMatrix<spacedim,
        Number>::distribute_all_non_leaf_nodes_sigma_r_and_f_to_leaves()
{
  /**
   * Only non-leaf \hmatrix nodes need to be processed.
   */
  if (submatrices.size() > 0)
    {
      /**
       * Since the current \hmatrix node has children, its type
       * should be \p HierarchicalMatrixType.
       */
      Assert(type == HierarchicalMatrixType, ExcInvalidHMatrixType(type));

      /**
       * Distribute matrices in \f$\Sigma_b^R\f$ and \f$\Sigma_b^F\f$ of the
       * current \hmatrix node to its leaves, which is also a
       * recursive function call.
       */
      distribute_sigma_r_and_f_to_leaves();

      /**
       * Distribute matrices in \f$\Sigma_b^R\f$ and \f$\Sigma_b^F\f$ of each
       * child matrix of the current \hmatrix node to its leaves
       */
      for (HMatrix<spacedim, Number> *submatrix : submatrices)
        {
          Assert(submatrix, ExcInternalError());

          submatrix->distribute_all_non_leaf_nodes_sigma_r_and_f_to_leaves();
        }
    }
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::distribute_sigma_r_and_f_to_leaves()
{
  if (Sigma_R.size() > 0 || Sigma_F.size() > 0)
    {
      if (row_index_global_to_local_map.size() == 0)
        {
          build_index_set_global_to_local_map(*row_indices,
                                              row_index_global_to_local_map);
        }

      if (col_index_global_to_local_map.size() == 0)
        {
          build_index_set_global_to_local_map(*col_indices,
                                              col_index_global_to_local_map);
        }

      _distribute_sigma_r_and_f_to_leaves(*this);

      for (auto &rkmatrix_in_starting_hmat : Sigma_R)
        {
          if (rkmatrix_in_starting_hmat != nullptr)
            {
              delete rkmatrix_in_starting_hmat;
              rkmatrix_in_starting_hmat = nullptr;
            }
        }

      Sigma_R.clear();

      for (auto &fullmatrix_in_starting_hmat : Sigma_F)
        {
          if (fullmatrix_in_starting_hmat != nullptr)
            {
              delete fullmatrix_in_starting_hmat;
              fullmatrix_in_starting_hmat = nullptr;
            }
        }

      Sigma_F.clear();
    }
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::_distribute_sigma_r_and_f_to_leaves(
  HMatrix<spacedim, Number> &starting_hmat)
{
  if (submatrices.size() > 0)
    {
      for (HMatrix<spacedim, Number> *submatrix : submatrices)
        {
          submatrix->_distribute_sigma_r_and_f_to_leaves(starting_hmat);
        }
    }
  else
    {
      switch (type)
        {
          case FullMatrixType:
            {
              /**
               * Restrict each rank-k matrix in the list \p Sigma_R of \p
               * starting_hmat to the block as a full matrix.
               */
              for (RkMatrix<Number> *rkmatrix_in_starting_hmat :
                   starting_hmat.Sigma_R)
                {
                  Assert(rkmatrix_in_starting_hmat, ExcInternalError());

                  LAPACKFullMatrixExt<Number> fullmatrix_restricted;

                  rkmatrix_in_starting_hmat->restrictToFullMatrix(
                    *row_indices,
                    *col_indices,
                    starting_hmat.row_index_global_to_local_map,
                    starting_hmat.col_index_global_to_local_map,
                    fullmatrix_restricted);

                  fullmatrix->add(fullmatrix_restricted);
                }

              /**
               * Restrict each full matrix in the list \p Sigma_F of \p
               * starting_hmat to the block as a full matrix.
               */
              for (LAPACKFullMatrixExt<Number> *fullmatrix_in_starting_hmat :
                   starting_hmat.Sigma_F)
                {
                  Assert(fullmatrix_in_starting_hmat, ExcInternalError());

                  LAPACKFullMatrixExt<Number> fullmatrix_restricted(
                    *row_indices,
                    *col_indices,
                    *fullmatrix_in_starting_hmat,
                    starting_hmat.row_index_global_to_local_map,
                    starting_hmat.col_index_global_to_local_map);

                  fullmatrix->add(fullmatrix_restricted);
                }

              break;
            }
          case RkMatrixType:
            {
              /**
               * Restrict each rank-k matrix in the list \p Sigma_R of \p
               * starting_hmat to the block as a rank-k matrix.
               */
              for (RkMatrix<Number> *rkmatrix_in_starting_hmat :
                   starting_hmat.Sigma_R)
                {
                  Assert(rkmatrix_in_starting_hmat, ExcInternalError());

                  RkMatrix<Number> rkmatrix_restricted(
                    *row_indices,
                    *col_indices,
                    *rkmatrix_in_starting_hmat,
                    starting_hmat.row_index_global_to_local_map,
                    starting_hmat.col_index_global_to_local_map);

                  rkmatrix->add(rkmatrix_restricted);
                }

              /**
               * Restrict each full matrix in the list \p Sigma_F of \p
               * starting_hmat to the block as a rank-k matrix.
               */
              for (LAPACKFullMatrixExt<Number> *fullmatrix_in_starting_hmat :
                   starting_hmat.Sigma_F)
                {
                  Assert(fullmatrix_in_starting_hmat, ExcInternalError());

                  RkMatrix<Number> rkmatrix_restricted(
                    *row_indices,
                    *col_indices,
                    *fullmatrix_in_starting_hmat,
                    starting_hmat.row_index_global_to_local_map,
                    starting_hmat.col_index_global_to_local_map);

                  rkmatrix->add(rkmatrix_restricted);
                }

              break;
            }
          default:
            {
              Assert(false, ExcInvalidHMatrixType(type));
              break;
            }
        }
    }
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::release()
{
  if (rkmatrix != nullptr)
    {
      delete rkmatrix;
      rkmatrix = nullptr;
    }

  if (fullmatrix != nullptr)
    {
      delete fullmatrix;
      fullmatrix = nullptr;
    }

  for (auto submatrix : submatrices)
    {
      /**
       * The deletion of \p submatrix will call the destructor of this
       * sub-HMatrix, which will further recursively call the destructor of the
       * submatrices of this sub-HMatrix. Hence, this destructor is
       * intrinsically recursive.
       */
      if (submatrix != nullptr)
        {
          delete submatrix;
        }
    }

  submatrices.clear();
  leaf_set.clear();

  type        = UndefinedMatrixType;
  bc_node     = nullptr;
  row_indices = nullptr;
  col_indices = nullptr;
  row_index_global_to_local_map.clear();
  col_index_global_to_local_map.clear();
  m = 0;
  n = 0;

  Sigma_P.clear();

  for (auto &r : Sigma_R)
    {
      if (r != nullptr)
        {
          delete r;
          r = nullptr;
        }
    }
  Sigma_R.clear();

  for (auto &f : Sigma_F)
    {
      if (f != nullptr)
        {
          delete f;
          f = nullptr;
        }
    }
  Sigma_F.clear();

  Tind.release();
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::clear()
{
  /**
   * Recursively clear submatrices.
   */
  if (submatrices.size() > 0)
    {
      Assert(type == HierarchicalMatrixType, ExcInvalidHMatrixType(type));

      for (HMatrix *submatrix : submatrices)
        {
          submatrix->clear();
        }
    }

  /**
   * Clear the current matrix node.
   */
  clear_hmat_node();
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::clear_hmat_node()
{
  type = UndefinedMatrixType;
  submatrices.clear();
  leaf_set.clear();
  rkmatrix    = nullptr;
  fullmatrix  = nullptr;
  bc_node     = nullptr;
  row_indices = nullptr;
  col_indices = nullptr;
  row_index_global_to_local_map.clear();
  col_index_global_to_local_map.clear();
  m = 0;
  n = 0;
  Sigma_P.clear();
  Sigma_R.clear();
  Sigma_F.clear();
  Tind.clear();
}


template <int spacedim, typename Number>
HMatrix<spacedim, Number>::~HMatrix()
{
  release();
}


template <int spacedim, typename Number>
HMatrixType
HMatrix<spacedim, Number>::get_type() const
{
  return type;
}


template <int spacedim, typename Number>
typename HMatrix<spacedim, Number>::size_type
HMatrix<spacedim, Number>::get_m() const
{
  return m;
}


template <int spacedim, typename Number>
typename HMatrix<spacedim, Number>::size_type
HMatrix<spacedim, Number>::get_n() const
{
  return n;
}


template <int spacedim, typename Number>
RkMatrix<Number> *
HMatrix<spacedim, Number>::get_rkmatrix()
{
  return rkmatrix;
}


template <int spacedim, typename Number>
const RkMatrix<Number> *
HMatrix<spacedim, Number>::get_rkmatrix() const
{
  return rkmatrix;
}


template <int spacedim, typename Number>
LAPACKFullMatrixExt<Number> *
HMatrix<spacedim, Number>::get_fullmatrix()
{
  return fullmatrix;
}


template <int spacedim, typename Number>
const LAPACKFullMatrixExt<Number> *
HMatrix<spacedim, Number>::get_fullmatrix() const
{
  return fullmatrix;
}


template <int spacedim, typename Number>
std::vector<HMatrix<spacedim, Number> *> &
HMatrix<spacedim, Number>::get_submatrices()
{
  return submatrices;
}


template <int spacedim, typename Number>
const std::vector<HMatrix<spacedim, Number> *> &
HMatrix<spacedim, Number>::get_submatrices() const
{
  return submatrices;
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::print_formatted(std::ostream &     out,
                                           const unsigned int precision,
                                           const bool         scientific,
                                           const unsigned int width,
                                           const char *       zero_string,
                                           const double       denominator,
                                           const double       threshold) const
{
  switch (type)
    {
      case FullMatrixType:
        fullmatrix->print_formatted(out,
                                    precision,
                                    scientific,
                                    width,
                                    zero_string,
                                    denominator,
                                    threshold);

        break;
      case RkMatrixType:
        rkmatrix->print_formatted(out,
                                  precision,
                                  scientific,
                                  width,
                                  zero_string,
                                  denominator,
                                  threshold);

        break;
      case HierarchicalMatrixType:
        for (HMatrix *submatrix : submatrices)
          {
            submatrix->print_formatted(out,
                                       precision,
                                       scientific,
                                       width,
                                       zero_string,
                                       denominator,
                                       threshold);
          }

        break;
      case UndefinedMatrixType:
      default:
        Assert(false, ExcInvalidHMatrixType(type));
        break;
    }
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::print_matrix_info(std::ostream &out) const
{
  /**
   * Print the size of \f$\Sigma_b^P\f$, \f$\Sigma_b^R\f$ and \f$\Sigma_b^F\f$.
   */
  print_h_submatrix_accessor(std::cout, "M", *this);
  out << "(#level, #Sigma_b^P, #Sigma_b^R, #Sigma_b^F)=("
      << bc_node->get_level() << "," << Sigma_P.size() << "," << Sigma_R.size()
      << "," << Sigma_F.size() << ")" << std::endl;

  for (std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>
         hmat_pair : Sigma_P)
    {
      out << "  Sigma_P products: ";
      print_h_h_submatrix_mmult_accessor(
        std::cout, "M1", *(hmat_pair.first), "M2", *(hmat_pair.second));
    }

  for (const HMatrix<spacedim, Number> *submatrix : submatrices)
    {
      submatrix->print_matrix_info(out);
    }
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::write_fullmatrix_leaf_node(
  std::ostream &out,
  const Number  singular_value_threshold) const
{
  Assert(type == FullMatrixType, ExcInvalidHMatrixType(type));

  const std::vector<types::global_dof_index> &tau_index_set =
    bc_node->get_data_reference()
      .get_tau_node()
      ->get_data_reference()
      .get_index_set();
  const std::vector<types::global_dof_index> &sigma_index_set =
    bc_node->get_data_reference()
      .get_sigma_node()
      ->get_data_reference()
      .get_index_set();

  /**
   * Print index set of cluster \f$\tau\f$.
   */
  out << "[";
  print_vector_values(out, tau_index_set, " ", false);
  out << "],";

  /**
   * Print index set of cluster \f$\sigma\f$.
   */
  out << "[";
  print_vector_values(out, sigma_index_set, " ", false);
  out << "],";

  /**
   * Print the \p is_near_field flag.
   */
  out << (bc_node->get_data_reference().get_is_near_field() ? 1 : 0) << ",";

  /**
   * Make a copy of the matrix block and calculate its rank using
   * SVD.
   */
  LAPACKFullMatrixExt<Number> copy(*fullmatrix);
  const size_t                rank = copy.rank(singular_value_threshold);

  /**
   * Print the \p rank flag.
   */
  out << rank << "\n";
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::write_rkmatrix_leaf_node(std::ostream &out) const
{
  Assert(type == RkMatrixType, ExcInvalidHMatrixType(type));

  const std::vector<types::global_dof_index> &tau_index_set =
    bc_node->get_data_reference()
      .get_tau_node()
      ->get_data_reference()
      .get_index_set();
  const std::vector<types::global_dof_index> &sigma_index_set =
    bc_node->get_data_reference()
      .get_sigma_node()
      ->get_data_reference()
      .get_index_set();

  /**
   * Print index set of cluster \f$\tau\f$.
   */
  out << "[";
  print_vector_values(out, tau_index_set, " ", false);
  out << "],";

  /**
   * Print index set of cluster \f$\sigma\f$.
   */
  out << "[";
  print_vector_values(out, sigma_index_set, " ", false);
  out << "],";

  /**
   * Print the \p is_near_field flag.
   */
  out << (bc_node->get_data_reference().get_is_near_field() ? 1 : 0) << ",";

  /**
   * Print the \p rank flag.
   */
  out << rkmatrix->get_rank() << "\n";
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::write_leaf_set(
  std::ostream &out,
  const Number  singular_value_threshold) const
{
  switch (type)
    {
      case FullMatrixType:
        {
          write_fullmatrix_leaf_node(out, singular_value_threshold);

          break;
        }
      case RkMatrixType:
        {
          write_rkmatrix_leaf_node(out);

          break;
        }
      case HierarchicalMatrixType:
        {
          for (HMatrix *submatrix : submatrices)
            {
              submatrix->write_leaf_set(out);
            }

          break;
        }
      default:
        {
          Assert(false, ExcInvalidHMatrixType(type));

          break;
        }
    }
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::write_leaf_set_by_iteration(
  std::ostream &out,
  const Number  singular_value_threshold) const
{
  for (HMatrix *leaf_node : leaf_set)
    {
      switch (leaf_node->type)
        {
          case FullMatrixType:
            {
              leaf_node->write_fullmatrix_leaf_node(out,
                                                    singular_value_threshold);

              break;
            }
          case RkMatrixType:
            {
              leaf_node->write_rkmatrix_leaf_node(out);

              break;
            }
          default:
            {
              Assert(false, ExcInvalidHMatrixType(type));

              break;
            }
        }
    }
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::truncate_to_rank(size_type new_rank)
{
  switch (type)
    {
      case HierarchicalMatrixType:
        {
          for (HMatrix *submatrix : submatrices)
            {
              submatrix->truncate_to_rank(new_rank);
            }

          break;
        }
      case FullMatrixType:
        {
          /**
           * Do nothing.
           */

          break;
        }
      case RkMatrixType:
        {
          /**
           * Truncate the RkMatrix in-place.
           */
          rkmatrix->truncate_to_rank(new_rank);

          break;
        }
      case UndefinedMatrixType:
      default:
        {
          Assert(false, ExcInvalidHMatrixType(type));
          break;
        }
    }
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::vmult(Vector<Number> &      y,
                                 const Vector<Number> &x) const
{
  Vector<Number> local_y(m);
  Vector<Number> local_x(n);

  switch (type)
    {
      case HierarchicalMatrixType:
        for (HMatrix *submatrix : submatrices)
          {
            submatrix->vmult(y, x);
          }

        break;
      case FullMatrixType:
        /**
         * Restrict vector x to the current matrix block.
         */
        for (size_type j = 0; j < n; j++)
          {
            local_x(j) = x(col_indices->at(j));
          }

        fullmatrix->vmult(local_y, local_x);

        /**
         * Merge back the result vector \p local_y to \p y.
         */
        for (size_type i = 0; i < m; i++)
          {
            y(row_indices->at(i)) += local_y(i);
          }

        break;
      case RkMatrixType:
        /**
         * Restrict vector x to the current matrix block.
         */
        for (size_type j = 0; j < n; j++)
          {
            local_x(j) = x(col_indices->at(j));
          }

        rkmatrix->vmult(local_y, local_x);

        /**
         * Merge back the result vector \p local_y to \p y.
         */
        for (size_type i = 0; i < m; i++)
          {
            y(row_indices->at(i)) += local_y(i);
          }

        break;
      case UndefinedMatrixType:
      default:
        Assert(false, ExcInvalidHMatrixType(type));
    }
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::vmult_local_vector(
  Vector<Number> &                                 y,
  const std::map<types::global_dof_index, size_t> &y_index_global_to_local_map,
  const Vector<Number> &                           x,
  const std::map<types::global_dof_index, size_t> &x_index_global_to_local_map)
  const
{
  Vector<Number> local_y(m);
  Vector<Number> local_x(n);

  switch (type)
    {
      case HierarchicalMatrixType:
        for (HMatrix *submatrix : submatrices)
          {
            submatrix->vmult_local_vector(y,
                                          y_index_global_to_local_map,
                                          x,
                                          x_index_global_to_local_map);
          }

        break;
      case FullMatrixType:
        /**
         * Restrict vector x to the current matrix block.
         */
        for (size_type j = 0; j < n; j++)
          {
            local_x(j) = x(x_index_global_to_local_map.at(col_indices->at(j)));
          }

        fullmatrix->vmult(local_y, local_x);

        /**
         * Merge back the result vector \p local_y to \p y.
         */
        for (size_type i = 0; i < m; i++)
          {
            y(y_index_global_to_local_map.at(row_indices->at(i))) += local_y(i);
          }

        break;
      case RkMatrixType:
        /**
         * Restrict vector x to the current matrix block.
         */
        for (size_type j = 0; j < n; j++)
          {
            local_x(j) = x(x_index_global_to_local_map.at(col_indices->at(j)));
          }

        rkmatrix->vmult(local_y, local_x);

        /**
         * Merge back the result vector \p local_y to \p y.
         */
        for (size_type i = 0; i < m; i++)
          {
            y(y_index_global_to_local_map.at(row_indices->at(i))) += local_y(i);
          }

        break;
      case UndefinedMatrixType:
      default:
        Assert(false, ExcInvalidHMatrixType(type));
    }
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::Tvmult(Vector<Number> &      y,
                                  const Vector<Number> &x) const
{
  Vector<Number> local_y(n);
  Vector<Number> local_x(m);

  switch (type)
    {
      case HierarchicalMatrixType:
        for (HMatrix *submatrix : submatrices)
          {
            submatrix->Tvmult(y, x);
          }

        break;
      case FullMatrixType:
        /**
         * Restrict vector x to the current matrix block.
         */
        for (size_type j = 0; j < m; j++)
          {
            local_x(j) = x(row_indices->at(j));
          }

        fullmatrix->Tvmult(local_y, local_x);

        /**
         * Merge back the result vector \p local_y to \p y.
         */
        for (size_type i = 0; i < n; i++)
          {
            y(col_indices->at(i)) += local_y(i);
          }

        break;
      case RkMatrixType:
        /**
         * Restrict vector x to the current matrix block.
         */
        for (size_type j = 0; j < m; j++)
          {
            local_x(j) = x(row_indices->at(j));
          }

        rkmatrix->Tvmult(local_y, local_x);

        /**
         * Merge back the result vector \p local_y to \p y.
         */
        for (size_type i = 0; i < n; i++)
          {
            y(col_indices->at(i)) += local_y(i);
          }

        break;
      case UndefinedMatrixType:
      default:
        Assert(false, ExcInvalidHMatrixType(type));
    }
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::Tvmult_local_vector(
  Vector<Number> &                                 y,
  const std::map<types::global_dof_index, size_t> &y_index_global_to_local_map,
  const Vector<Number> &                           x,
  const std::map<types::global_dof_index, size_t> &x_index_global_to_local_map)
  const
{
  Vector<Number> local_y(n);
  Vector<Number> local_x(m);

  switch (type)
    {
      case HierarchicalMatrixType:
        for (HMatrix *submatrix : submatrices)
          {
            submatrix->Tvmult_local_vector(y,
                                           y_index_global_to_local_map,
                                           x,
                                           x_index_global_to_local_map);
          }

        break;
      case FullMatrixType:
        /**
         * Restrict vector x to the current matrix block.
         */
        for (size_type j = 0; j < m; j++)
          {
            local_x(j) = x(x_index_global_to_local_map.at(row_indices->at(j)));
          }

        fullmatrix->Tvmult(local_y, local_x);

        /**
         * Merge back the result vector \p local_y to \p y.
         */
        for (size_type i = 0; i < n; i++)
          {
            y(y_index_global_to_local_map.at(col_indices->at(i))) += local_y(i);
          }

        break;
      case RkMatrixType:
        /**
         * Restrict vector x to the current matrix block.
         */
        for (size_type j = 0; j < m; j++)
          {
            local_x(j) = x(x_index_global_to_local_map.at(row_indices->at(j)));
          }

        rkmatrix->Tvmult(local_y, local_x);

        /**
         * Merge back the result vector \p local_y to \p y.
         */
        for (size_type i = 0; i < n; i++)
          {
            y(y_index_global_to_local_map.at(col_indices->at(i))) += local_y(i);
          }

        break;
      case UndefinedMatrixType:
      default:
        Assert(false, ExcInvalidHMatrixType(type));
    }
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::h_h_mmult_reduction()
{
  std::vector<
    std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>>
    Sigma_P_cannot_reduced;

  while (Sigma_P.size() > 0)
    {
      // Get the last element in the list.
      std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>
        &                        hmat_pair = Sigma_P.back();
      HMatrix<spacedim, Number> *M1        = hmat_pair.first;
      HMatrix<spacedim, Number> *M2        = hmat_pair.second;

      /**
       * When one of the operands is either full matrix or rank-k matrix,
       * perform direct multiplication.
       */
      if (M1->type == RkMatrixType)
        {
          rk_h_mmult_for_h_h_mmult(M1, M2, this);
        }
      else if (M2->type == RkMatrixType)
        {
          h_rk_mmult_for_h_h_mmult(M1, M2, this);
        }
      else if (M1->type == FullMatrixType)
        {
          f_h_mmult_for_h_h_mmult(M1, M2, this);
        }
      else if (M2->type == FullMatrixType)
        {
          h_f_mmult_for_h_h_mmult(M1, M2, this);
        }
      else if (M1->bc_node->get_split_mode() == VerticalSplitMode &&
               M2->bc_node->get_split_mode() == HorizontalSplitMode)
        {
          // Internal splitting
          Sigma_P.pop_back();

          Assert(M1->submatrices[0], ExcInternalError());
          Assert(M2->submatrices[0], ExcInternalError());
          Sigma_P.push_back(
            std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>(
              M1->submatrices[0], M2->submatrices[0]));

          Assert(M1->submatrices[1], ExcInternalError());
          Assert(M2->submatrices[1], ExcInternalError());
          Sigma_P.push_back(
            std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>(
              M1->submatrices[1], M2->submatrices[1]));
        }
      else
        {
          /**
           * Migrate the current H-matrix node pair to the list \p
           * Sigma_P_cannot_reduced.
           */
          Sigma_P_cannot_reduced.push_back(hmat_pair);
          /**
           * Remove the current H-matrix node pair from the original list in \p
           * M.
           */
          Sigma_P.pop_back();
        }
    }

  /**
   * Merge the elements in \p Sigma_P_cannot_reduced back to \p Sigma_P in \p M
   * for further processing.
   */
  for (std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>
         hmat_pair : Sigma_P_cannot_reduced)
    {
      Sigma_P.push_back(hmat_pair);
    }
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::h_h_mmult_horizontal_split(
  BlockClusterTree<spacedim, Number> &bc_tree)
{
  /**
   * Split the block cluster \f$b\f$ in \f$T_{\rm ind}\f$.
   */
  split_block_cluster_node(bc_node, bc_tree, HorizontalSplitMode);

  HMatrix<spacedim, Number> *child_hmat0 = new HMatrix<spacedim, Number>();
  InitHMatrixWrtBlockClusterNode(*child_hmat0, bc_node->get_child_pointer(0));
  /**
   * Append the initialized child to the list of submatrices
   * of \p M.
   */
  submatrices.push_back(child_hmat0);

  HMatrix<spacedim, Number> *child_hmat1 = new HMatrix<spacedim, Number>();
  InitHMatrixWrtBlockClusterNode(*child_hmat1, bc_node->get_child_pointer(1));
  /**
   * Append the initialized child to the list of submatrices
   * of \p hmat.
   */
  submatrices.push_back(child_hmat1);

  /**
   * Iterate over each multiplication subtask.
   */
  while (Sigma_P.size() > 0)
    {
      // Get the last element in the list.
      std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>
        &hmat_pair = Sigma_P.back();

      /**
       * Create \f$\mathcal{H}\f$-matrices corresponding to the
       * child block clusters after splitting.
       */
      if (hmat_pair.first->bc_node->get_split_mode() == HorizontalSplitMode &&
          hmat_pair.second->bc_node->get_split_mode() == UnsplitMode)
        {
          Assert(hmat_pair.first->submatrices[0], ExcInternalError());
          child_hmat0->Sigma_P.push_back(
            std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>(
              hmat_pair.first->submatrices[0], hmat_pair.second));

          Assert(hmat_pair.first->submatrices[1], ExcInternalError());
          child_hmat1->Sigma_P.push_back(
            std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>(
              hmat_pair.first->submatrices[1], hmat_pair.second));

          /**
           * Remove the current \hmatrix pair from
           * the list \p Sigma_P of the current matrix node.
           */
          Sigma_P.pop_back();
        }
      else if (hmat_pair.first->bc_node->get_split_mode() == CrossSplitMode &&
               hmat_pair.second->bc_node->get_split_mode() ==
                 HorizontalSplitMode)
        {
          Assert(hmat_pair.first->submatrices[0], ExcInternalError());
          Assert(hmat_pair.second->submatrices[0], ExcInternalError());
          child_hmat0->Sigma_P.push_back(
            std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>(
              hmat_pair.first->submatrices[0],
              hmat_pair.second->submatrices[0]));

          Assert(hmat_pair.first->submatrices[1], ExcInternalError());
          Assert(hmat_pair.second->submatrices[1], ExcInternalError());
          child_hmat0->Sigma_P.push_back(
            std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>(
              hmat_pair.first->submatrices[1],
              hmat_pair.second->submatrices[1]));

          Assert(hmat_pair.first->submatrices[2], ExcInternalError());
          child_hmat1->Sigma_P.push_back(
            std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>(
              hmat_pair.first->submatrices[2],
              hmat_pair.second->submatrices[0]));

          Assert(hmat_pair.first->submatrices[3], ExcInternalError());
          child_hmat1->Sigma_P.push_back(
            std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>(
              hmat_pair.first->submatrices[3],
              hmat_pair.second->submatrices[1]));

          /**
           * Remove the current \hmatrix pair from
           * the list \p Sigma_P of the current matrix node.
           */
          Sigma_P.pop_back();
        }
      else
        {
          Assert(false, ExcInternalError());
        }
    }

  /**
   * Update the matrix type of the current
   * \hmatrix.
   */
  type = HierarchicalMatrixType;
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::h_h_mmult_vertical_split(
  BlockClusterTree<spacedim, Number> &bc_tree)
{
  /**
   * Split the block cluster \f$b\f$ in \f$T_{\rm ind}\f$.
   */
  split_block_cluster_node(bc_node, bc_tree, VerticalSplitMode);

  HMatrix<spacedim, Number> *child_hmat0 = new HMatrix<spacedim, Number>();
  InitHMatrixWrtBlockClusterNode(*child_hmat0, bc_node->get_child_pointer(0));
  /**
   * Append the initialized child to the list of submatrices
   * of \p M.
   */
  submatrices.push_back(child_hmat0);

  HMatrix<spacedim, Number> *child_hmat1 = new HMatrix<spacedim, Number>();
  InitHMatrixWrtBlockClusterNode(*child_hmat1, bc_node->get_child_pointer(1));
  /**
   * Append the initialized child to the list of submatrices
   * of \p hmat.
   */
  submatrices.push_back(child_hmat1);

  /**
   * Iterate over each multiplication subtask.
   */
  while (Sigma_P.size() > 0)
    {
      // Get the last element in the list.
      std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>
        &hmat_pair = Sigma_P.back();

      /**
       * Create \f$\mathcal{H}\f$-matrices corresponding to the
       * child block clusters after splitting.
       */
      if (hmat_pair.first->bc_node->get_split_mode() == UnsplitMode &&
          hmat_pair.second->bc_node->get_split_mode() == VerticalSplitMode)
        {
          Assert(hmat_pair.second->submatrices[0], ExcInternalError());
          child_hmat0->Sigma_P.push_back(
            std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>(
              hmat_pair.first, hmat_pair.second->submatrices[0]));

          Assert(hmat_pair.second->submatrices[1], ExcInternalError());
          child_hmat1->Sigma_P.push_back(
            std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>(
              hmat_pair.first, hmat_pair.second->submatrices[1]));

          /**
           * Remove the current \hmatrix pair from
           * the list \p Sigma_P of the current matrix node.
           */
          Sigma_P.pop_back();
        }
      else if (hmat_pair.first->bc_node->get_split_mode() ==
                 VerticalSplitMode &&
               hmat_pair.second->bc_node->get_split_mode() == CrossSplitMode)
        {
          Assert(hmat_pair.first->submatrices[0], ExcInternalError());
          Assert(hmat_pair.second->submatrices[0], ExcInternalError());
          child_hmat0->Sigma_P.push_back(
            std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>(
              hmat_pair.first->submatrices[0],
              hmat_pair.second->submatrices[0]));

          Assert(hmat_pair.first->submatrices[1], ExcInternalError());
          Assert(hmat_pair.second->submatrices[2], ExcInternalError());
          child_hmat0->Sigma_P.push_back(
            std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>(
              hmat_pair.first->submatrices[1],
              hmat_pair.second->submatrices[2]));

          Assert(hmat_pair.second->submatrices[1], ExcInternalError());
          child_hmat1->Sigma_P.push_back(
            std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>(
              hmat_pair.first->submatrices[0],
              hmat_pair.second->submatrices[1]));

          Assert(hmat_pair.second->submatrices[3], ExcInternalError());
          child_hmat1->Sigma_P.push_back(
            std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>(
              hmat_pair.first->submatrices[1],
              hmat_pair.second->submatrices[3]));

          /**
           * Remove the current \hmatrix pair from
           * the list \p Sigma_P of the current matrix node.
           */
          Sigma_P.pop_back();
        }
      else
        {
          Assert(false, ExcInternalError());
        }
    }

  /**
   * Update the matrix type of the current
   * \hmatrix.
   */
  type = HierarchicalMatrixType;
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::h_h_mmult_cross_split(
  BlockClusterTree<spacedim, Number> &bc_tree)
{
  /**
   * Split the block cluster \f$b\f$ in \f$T_{\rm ind}\f$.
   */
  split_block_cluster_node(bc_node, bc_tree, CrossSplitMode);

  HMatrix<spacedim, Number> *child_hmat0 = new HMatrix<spacedim, Number>();
  InitHMatrixWrtBlockClusterNode(*child_hmat0, bc_node->get_child_pointer(0));
  /**
   * Append the initialized child to the list of submatrices
   * of \p M.
   */
  submatrices.push_back(child_hmat0);

  HMatrix<spacedim, Number> *child_hmat1 = new HMatrix<spacedim, Number>();
  InitHMatrixWrtBlockClusterNode(*child_hmat1, bc_node->get_child_pointer(1));
  /**
   * Append the initialized child to the list of submatrices
   * of \p hmat.
   */
  submatrices.push_back(child_hmat1);

  HMatrix<spacedim, Number> *child_hmat2 = new HMatrix<spacedim, Number>();
  InitHMatrixWrtBlockClusterNode(*child_hmat2, bc_node->get_child_pointer(2));
  /**
   * Append the initialized child to the list of submatrices
   * of \p M.
   */
  submatrices.push_back(child_hmat2);

  HMatrix<spacedim, Number> *child_hmat3 = new HMatrix<spacedim, Number>();
  InitHMatrixWrtBlockClusterNode(*child_hmat3, bc_node->get_child_pointer(3));
  /**
   * Append the initialized child to the list of submatrices
   * of \p hmat.
   */
  submatrices.push_back(child_hmat3);

  /**
   * Iterate over each multiplication subtask.
   */
  while (Sigma_P.size() > 0)
    {
      // Get the last element in the list.
      std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>
        &hmat_pair = Sigma_P.back();

      /**
       * Create \f$\mathcal{H}\f$-matrices corresponding to the
       * child block clusters after splitting.
       */
      if (hmat_pair.first->bc_node->get_split_mode() == CrossSplitMode &&
          hmat_pair.second->bc_node->get_split_mode() == CrossSplitMode)
        {
          /**
           * \f$\Sigma_{b_s(1)}^P := \Sigma_{b_s(1)}^P \cup
           * \{[\tilde{M}_1(1), \tilde{M}_2(1)], [\tilde{M}_1(2),
           * \tilde{M}_2(3)]\}\f$
           */
          Assert(hmat_pair.first->submatrices[0], ExcInternalError());
          Assert(hmat_pair.second->submatrices[0], ExcInternalError());
          child_hmat0->Sigma_P.push_back(
            std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>(
              hmat_pair.first->submatrices[0],
              hmat_pair.second->submatrices[0]));

          Assert(hmat_pair.first->submatrices[1], ExcInternalError());
          Assert(hmat_pair.second->submatrices[2], ExcInternalError());
          child_hmat0->Sigma_P.push_back(
            std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>(
              hmat_pair.first->submatrices[1],
              hmat_pair.second->submatrices[2]));

          /**
           * \f$\Sigma_{b_s(2)}^P := \Sigma_{b_s(2)}^P \cup
           * \{[\tilde{M}_1(1), \tilde{M}_2(2)], [\tilde{M}_1(2),
           * \tilde{M}_2(4)]\}\f$
           */
          Assert(hmat_pair.second->submatrices[1], ExcInternalError());
          child_hmat1->Sigma_P.push_back(
            std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>(
              hmat_pair.first->submatrices[0],
              hmat_pair.second->submatrices[1]));

          Assert(hmat_pair.second->submatrices[3], ExcInternalError());
          child_hmat1->Sigma_P.push_back(
            std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>(
              hmat_pair.first->submatrices[1],
              hmat_pair.second->submatrices[3]));

          /**
           * \f$\Sigma_{b_s(3)}^P := \Sigma_{b_s(3)}^P \cup
           * \{[\tilde{M}_1(3), \tilde{M}_2(1)], [\tilde{M}_1(4),
           * \tilde{M}_2(3)]\}\f$
           */
          Assert(hmat_pair.first->submatrices[2], ExcInternalError());
          child_hmat2->Sigma_P.push_back(
            std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>(
              hmat_pair.first->submatrices[2],
              hmat_pair.second->submatrices[0]));

          Assert(hmat_pair.first->submatrices[3], ExcInternalError());
          child_hmat2->Sigma_P.push_back(
            std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>(
              hmat_pair.first->submatrices[3],
              hmat_pair.second->submatrices[2]));

          /**
           * \f$\Sigma_{b_s(4)}^P := \Sigma_{b_s(4)}^P \cup
           * \{[\tilde{M}_1(3), \tilde{M}_2(2)], [\tilde{M}_1(4),
           * \tilde{M}_2(4)]\}\f$
           */
          child_hmat3->Sigma_P.push_back(
            std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>(
              hmat_pair.first->submatrices[2],
              hmat_pair.second->submatrices[1]));

          child_hmat3->Sigma_P.push_back(
            std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>(
              hmat_pair.first->submatrices[3],
              hmat_pair.second->submatrices[3]));

          /**
           * Remove the current \hmatrix pair from
           * the list \p Sigma_P of the current matrix node.
           */
          Sigma_P.pop_back();
        }
      else if (hmat_pair.first->bc_node->get_split_mode() ==
                 HorizontalSplitMode &&
               hmat_pair.second->bc_node->get_split_mode() == VerticalSplitMode)
        {
          Assert(hmat_pair.first->submatrices[0], ExcInternalError());
          Assert(hmat_pair.second->submatrices[0], ExcInternalError());
          child_hmat0->Sigma_P.push_back(
            std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>(
              hmat_pair.first->submatrices[0],
              hmat_pair.second->submatrices[0]));

          Assert(hmat_pair.second->submatrices[1], ExcInternalError());
          child_hmat1->Sigma_P.push_back(
            std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>(
              hmat_pair.first->submatrices[0],
              hmat_pair.second->submatrices[1]));

          Assert(hmat_pair.first->submatrices[1], ExcInternalError());
          child_hmat2->Sigma_P.push_back(
            std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>(
              hmat_pair.first->submatrices[1],
              hmat_pair.second->submatrices[0]));

          child_hmat3->Sigma_P.push_back(
            std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>(
              hmat_pair.first->submatrices[1],
              hmat_pair.second->submatrices[1]));

          /**
           * Remove the current \hmatrix pair from
           * the list \p Sigma_P of the current matrix node.
           */
          Sigma_P.pop_back();
        }
      else
        {
          Assert(false, ExcInternalError());
        }
    }

  /**
   * Update the matrix type of the current
   * \hmatrix.
   */
  type = HierarchicalMatrixType;
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::mmult(
  HMatrix<spacedim, Number> &               C,
  HMatrix<spacedim, Number> &               B,
  const BlockClusterTree<spacedim, Number> &bct_a,
  const BlockClusterTree<spacedim, Number> &bct_b,
  BlockClusterTree<spacedim, Number> &      bct_c,
  const unsigned int                        fixed_rank)
{
  /**
   * <dl class="section">
   *   <dt>Work flow</dt>
   *   <dd>
   */

  /**
   * <ul>
   * <li>Release the resource of the result matrix.
   */
  C.release();

  /**
   * <li>Initialize the induced block cluster tree \f$T_{\rm ind}\f$ for the
   * result matrix with a single root node.
   */
  C.Tind = BlockClusterTree<spacedim, Number>(
    bct_a.get_root()->get_data_reference().get_tau_node(),
    bct_b.get_root()->get_data_reference().get_sigma_node(),
    bct_c.get_n_min());

  /**
   * <li>Associate with the root node of the induced block cluster tree \f$T_{\rm ind}\f$.
   */
  std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>
    initial_hmat_pair(this, &B);
  InitHMatrixWrtBlockClusterNode(C, C.Tind.get_root(), initial_hmat_pair);
  /**
   * <li>Perform recursive multiplication while constructing the induced block
   * cluster tree \f$T_{\rm ind}\f$.
   */
  h_h_mmult_phase1_recursion(&C, C.Tind);

  /**
   * <li>After the construction of the induced block cluster tree \f$T_{\rm ind}\f$,
   * rebuild its leaf set as well as near field and far field sets, and update
   * the tree depth and maximum level.
   */
  C.Tind.build_leaf_set();
  C.Tind.categorize_near_and_far_field_sets();
  C.Tind.calc_depth_and_max_level();

  /**
   * DEBUG: Print the structure of the \f$T_{\rm ind}\f$ block cluster tree.
   */
  std::ofstream out1("Tind_after_phase1.dat");
  C.Tind.write_leaf_set(out1);
  out1.close();

  /**
   * <li>Build the leaf set of the result matrix.
   */
  C.build_leaf_set();

  //  // DEBUG
  //  std::cout << "=== Product matrix info before phase 2 ===" << std::endl;
  //  C.print_matrix_info(std::cout);

  h_h_mmult_phase2(C, bct_c, fixed_rank);

  //  // DEBUG
  //  std::cout << "=== Product matrix info after phase 2 ===" << std::endl;
  //  C.print_matrix_info(std::cout);

  /**
   * DEBUG: Print the structure of the \f$T_{\rm ind}\f$ block cluster tree.
   */
  std::ofstream out2("Tind_after_phase2.dat");
  C.Tind.write_leaf_set(out2);
  out2.close();

  /**
   *
   *   </dd>
   * </dl>
   */
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::mmult(
  HMatrix<spacedim, Number> &               C,
  HMatrix<spacedim, Number> &               B,
  const BlockClusterTree<spacedim, Number> &bct_a,
  const BlockClusterTree<spacedim, Number> &bct_b,
  BlockClusterTree<spacedim, Number> &      bct_c,
  const unsigned int                        fixed_rank,
  const bool                                adding)
{
  if (adding)
    {
      HMatrix<spacedim, Number> C_prime;
      mmult(C_prime, B, bct_a, bct_b, bct_c, fixed_rank);
      C.add(C_prime, fixed_rank);
    }
  else
    {
      mmult(C, B, bct_a, bct_b, bct_c, fixed_rank);
    }
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::add(HMatrix<spacedim, Number> &      C,
                               const HMatrix<spacedim, Number> &B,
                               const size_type fixed_rank_k) const
{
  /**
   * <strong>Work flow</strong>
   */

  switch (type)
    {
      case HierarchicalMatrixType:
        {
          /**
           * Recursively add each pair of submatrices.
           */
          for (size_type i = 0; i < submatrices.size(); i++)
            {
              submatrices.at(i)->add(*(C.submatrices.at(i)),
                                     *(B.submatrices.at(i)),
                                     fixed_rank_k);
            }

          break;
        }
      case FullMatrixType:
        {
          /**
           * Perform addition of full matrices.
           */
          this->fullmatrix->add(*(C.fullmatrix), *(B.fullmatrix));

          break;
        }
      case RkMatrixType:
        {
          /**
           * Perform addition of rank-k matrices.
           */
          this->rkmatrix->add(*(C.rkmatrix), *(B.rkmatrix), fixed_rank_k);

          break;
        }
      case UndefinedMatrixType:
        Assert(false, ExcInvalidHMatrixType(type));
        break;
    }
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::add(const HMatrix<spacedim, Number> &B,
                               const size_type fixed_rank_k) const
{
  /**
   * <strong>Work flow</strong>
   */

  switch (type)
    {
      case HierarchicalMatrixType:
        {
          /**
           * Recursively add each pair of submatrices.
           */
          for (size_type i = 0; i < submatrices.size(); i++)
            {
              submatrices.at(i)->add(*(B.submatrices.at(i)), fixed_rank_k);
            }

          break;
        }
      case FullMatrixType:
        {
          /**
           * Perform addition of full matrices.
           */
          this->fullmatrix->add(*(B.fullmatrix));

          break;
        }
      case RkMatrixType:
        {
          /**
           * Perform addition of rank-k matrices.
           */
          this->rkmatrix->add(*(B.rkmatrix), fixed_rank_k);

          break;
        }
      case UndefinedMatrixType:
        Assert(false, ExcInvalidHMatrixType(type));
        break;
    }
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::invert_by_gauss_elim(
  HMatrix<spacedim, Number> &               M_inv,
  HMatrix<spacedim, Number> &               M_root,
  const BlockClusterTree<spacedim, Number> &M_root_bct,
  const size_type                           fixed_rank_k)
{
  AssertDimension(m, n);

  /**
   * If the current matrix block to be handled has a same \f$\tau\f$ cluster and
   * \f$\sigma\f$ cluster and belongs to the leaf set of \p M_root, directly
   * calculate its inverse as full matrix.
   */
  if (*(bc_node->get_data_reference().get_tau_node()) ==
        *(bc_node->get_data_reference().get_sigma_node()) &&
      M_root.find_block_cluster_in_leaf_set(bc_node->get_data_reference()) !=
        M_root.get_leaf_set().end())
    {
      Assert(type == FullMatrixType, ExcInvalidHMatrixType(type));

      this->fullmatrix->invert_by_gauss_elim(*(M_inv.fullmatrix));
    }
  else
    {
      AssertDimension(
        bc_node->get_data_reference().get_tau_node()->get_child_num(),
        bc_node->get_data_reference().get_sigma_node()->get_child_num());

      /**
       * Number of matrix block in a row.
       */
      const size_type k =
        bc_node->get_data_reference().get_tau_node()->get_child_num();

      /**
       * Stage 1: eliminate the lower triangular part of the matrix.
       */
      for (size_type l = 0; l < k; l++)
        {
          /**
           * Calculate the inverse of the diagonal block \f$M
           * \vert_{\tau[l]\times\tau[l]}\f$. The formula \f$l + l \cdot k\f$
           * calculates the 1D index of the diagonal block in \p submatrices.
           * This is because the submatrices of the current
           * \hmatrix node is stored in the following order:
           *
           * <code>
           * submatrices = {tau[0]*sigma[0], tau[0]*sigma[1], tau[1]*sigma[0],
           * tau[1]*sigma[1]}
           * </code>
           *
           * Since \f$\tau\f$ is the same as \f$\sigma\f$, we have
           *
           * <code>
           * submatrices = {tau[0]*tau[0], tau[0]*tau[1], tau[1]*tau[0],
           * tau[1]*tau[1]}
           * </code>
           *
           * Hence, the index of \p tau[0]*tau[0] in \p submatrices is 0 and the
           * index of \p tau[1]*tau[1] in \p submatrices is 3. The former
           * index is calculated as <code>0 + 0 * 2 = 0</code>, while the latter
           * index is calculated as <code>1 + 1 * 2 = 3</code>.
           */
          const size_type diag_block_index_in_submatrices = l + l * k;
          submatrices[diag_block_index_in_submatrices]->invert_by_gauss_elim(
            *(M_inv.submatrices[diag_block_index_in_submatrices]),
            M_root,
            M_root_bct,
            fixed_rank_k);

          for (size_type j = l + 1; j < k; j++)
            {
              /**
               * Create subtrees used for matrix multiplication.
               */
              BlockClusterTree<spacedim, Number> bct_op1(
                M_inv.submatrices[diag_block_index_in_submatrices]->bc_node,
                M_root_bct.get_eta(),
                M_root_bct.get_n_min());
              BlockClusterTree<spacedim, Number> bct_op2(
                submatrices[l + j * k]->bc_node,
                M_root_bct.get_eta(),
                M_root_bct.get_n_min());
              BlockClusterTree<spacedim, Number> bct_res(
                submatrices[l + j * k]->bc_node,
                M_root_bct.get_eta(),
                M_root_bct.get_n_min());

              HMatrix<spacedim, Number> *C = new HMatrix<spacedim, Number>();
              M_inv.submatrices[diag_block_index_in_submatrices]->mmult(
                *C,
                *(submatrices[l + j * k]),
                bct_op1,
                bct_op2,
                bct_res,
                fixed_rank_k);

              /**
               * Migrate the newly created \hmat to the target submatrix.
               */
              *(submatrices[l + j * k]) = std::move(*C);
            }
        }

      /**
       * Stage 2: eliminate the upper triangular part of the matrix.
       */
    }
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::coarsen_to_subtree(
  const BlockClusterTree<spacedim, Number> &subtree,
  const unsigned int                        fixed_rank_k)
{
  coarsen_to_partition(subtree.get_leaf_set(), fixed_rank_k);
  build_leaf_set();
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::coarsen_to_partition(
  const std::vector<
    typename BlockClusterTree<spacedim, Number>::node_pointer_type> &partition,
  const unsigned int fixed_rank_k)
{
  /**
   * N.B. The function call \p find_pointer_data here involves the comparison of
   * two block cluster nodes, which internally compares the contained two block
   * clusters, which further compares the contained tau node and sigma node
   * pointers. Therefore, at the moment, the inner most comparison is shallow
   * comparison.
   */
  if (find_pointer_data(partition.begin(), partition.end(), this->bc_node) !=
      partition.end())
    {
      /**
       * The block cluster node associated with the current \hmatrix node
       * belongs to the given \p partition. Then \f$\mathcal{T}_r^{\mathcal{R}
       * \leftarrow \mathcal{H}}\f$ will be applied to this \hmatrix node.
       */
      convertHMatBlockToRkMatrix(this, fixed_rank_k);
    }
  else
    {
      /**
       * When the block cluster node associated with the current
       * \hmatrix node does not belong to the \p partition,
       * recursively call this same member function of its each child.
       */
      for (HMatrix *submatrix : submatrices)
        {
          submatrix->coarsen_to_partition(partition, fixed_rank_k);
        }
    }
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::build_leaf_set()
{
  leaf_set.clear();
  _build_leaf_set(leaf_set);
}


template <int spacedim, typename Number>
std::vector<HMatrix<spacedim, Number> *> &
HMatrix<spacedim, Number>::get_leaf_set()
{
  return leaf_set;
}


template <int spacedim, typename Number>
const std::vector<HMatrix<spacedim, Number> *> &
HMatrix<spacedim, Number>::get_leaf_set() const
{
  return leaf_set;
}


template <int spacedim, typename Number>
typename std::vector<HMatrix<spacedim, Number> *>::iterator
HMatrix<spacedim, Number>::find_block_cluster_in_leaf_set(
  const BlockCluster<spacedim, Number> &block_cluster)
{
  typename std::vector<HMatrix<spacedim, Number> *>::iterator iter;
  for (iter = leaf_set.begin(); iter != leaf_set.end(); iter++)
    {
      /**
       * Perform a shallow comparison, i.e. compare by pointer address, of the
       * block clusters.
       *
       * <dl class="section note">
       *   <dt>Note</dt>
       *   <dd>The data held by those previously found leaf nodes of the source
       * \hmatrix have already been migrated to the leaf nodes
       * of the new \hmatrix, which will make the data fields in
       * these leaf nodes being empty. Hence, we will bypass them.</dd>
       * </dl>
       */
      if ((*iter)->bc_node != nullptr)
        {
          if ((*iter)->bc_node->get_data_reference() == block_cluster)
            {
              break;
            }
        }
    }

  return iter;
}


template <int spacedim, typename Number>
typename std::vector<HMatrix<spacedim, Number> *>::const_iterator
HMatrix<spacedim, Number>::find_block_cluster_in_leaf_set(
  const BlockCluster<spacedim, Number> &block_cluster) const
{
  for (typename std::vector<HMatrix<spacedim, Number> *>::const_iterator iter =
         leaf_set.cbegin();
       iter != leaf_set.cend();
       iter++)
    {
      /**
       * Perform a shallow comparison, i.e. compare by pointer address, of the
       * block clusters.
       *
       * <dl class="section note">
       *   <dt>Note</dt>
       *   <dd>The data held by those previously found leaf nodes of the source
       * \hmatrix have already been migrated to the leaf nodes
       * of the new \hmatrix, which will make the data fields in
       * these leaf nodes being empty. Hence, we will bypass them.</dd>
       * </dl>
       */
      if ((*iter)->bc_node != nullptr)
        {
          if ((*iter)->bc_node->get_data_reference() == block_cluster)
            {
              return iter;
            }
        }
    }
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::refine_to_supertree()
{
  /**
   * <dl class="section">
   *   <dt>Work flow</dt>
   *   <dd>
   */

  /**
   * Iterate over the leaf set of the \hmatrix hierarchy.
   */
  for (HMatrix *hmat_leaf_node : leaf_set)
    {
      /**
       * Refine from the current \hmatrix leaf node.
       */
      if (hmat_leaf_node->row_index_global_to_local_map.size() == 0)
        {
          build_index_set_global_to_local_map(
            *(hmat_leaf_node->row_indices),
            hmat_leaf_node->row_index_global_to_local_map);
        }

      if (hmat_leaf_node->col_index_global_to_local_map.size() == 0)
        {
          build_index_set_global_to_local_map(
            *(hmat_leaf_node->col_indices),
            hmat_leaf_node->col_index_global_to_local_map);
        }

      RefineHMatrixWrtExtendedBlockClusterTree(hmat_leaf_node, hmat_leaf_node);

      /**
       * After the refinement operation, we check the number of child matrices
       * of the current \hmatrix leaf node.
       */
      if (hmat_leaf_node->submatrices.size() > 0)
        {
          /**
           * If the current \hmatrix leaf node has a non-empty
           * collection of submatrices, it has really been refined. Then delete
           * its originally associated matrix data, either a full matrix or a
           * rank-k matrix, and modify its matrix type as \p
           * HierarchicalMatrixType.
           */
          switch (hmat_leaf_node->type)
            {
              case FullMatrixType:
                {
                  delete hmat_leaf_node->fullmatrix;
                  hmat_leaf_node->fullmatrix = nullptr;

                  break;
                }
              case RkMatrixType:
                {
                  delete hmat_leaf_node->rkmatrix;
                  hmat_leaf_node->rkmatrix = nullptr;

                  break;
                }
              default:
                {
                  Assert(false, ExcInvalidHMatrixType(hmat_leaf_node->type));

                  break;
                }
            }

          hmat_leaf_node->type = HierarchicalMatrixType;
        }
    }

  /**
   * After the refinement operation for all the leaf nodes of the original
   * \hmatrix hierarchy finishes, rebuild the leaf set of the
   * new \hmatrix hierarchy.
   */
  build_leaf_set();

  /**
   *   </dd>
   * </dl>
   */
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::convert_between_different_block_cluster_trees(
  BlockClusterTree<spacedim, Number> &bct1,
  BlockClusterTree<spacedim, Number> &bct2,
  const unsigned int                  fixed_rank_k2)
{
  /**
   * Make a copy of the leaf set of the target block cluster tree, which will be
   * used for the final coarsening.
   */
  std::vector<typename BlockClusterTree<spacedim, Number>::node_pointer_type>
    target_leaf_set(bct2.get_leaf_set());

  /**
   * Extend the block cluster tree associated with the current
   * \hmatrix to the coarsest tree which is finer than the
   * target block cluster tree. If the block cluster tree has really been
   * extended, refine the \hmatrix to its extended block cluster
   * tree.
   */
  if (bct1.extend_finer_than_partition(target_leaf_set))
    {
      this->refine_to_supertree();
    }

  /**
   * Extend \p bct2 to the finer partition obtained from \p bct1 as above. N.B.
   * Now the leaf set of \p bct1 after refinement is the same as that of \p
   * bct2 after this extension.
   */
  bool is_bct2_extended = bct2.extend_to_finer_partition(bct1.get_leaf_set());

  /**
   * Create a new \hmatrix with respect to the extended \p bct2,
   * which accepts the data migrated from the leaf set of the current
   * \hmatrix.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd><ul>
   *   <li>This hierarchical structure of the new \hmatrix is
   * built with respect to the extended block cluster tree \p bct2.
   *   <li>The shallow copy constructor cannot be used here because the new
   *   \hmatrix has a different block cluster tree structure
   *   from the current \hmatrix, even though they have the
   *   same partition after tree extension.
   *   </ul></dd>
   * </dl>
   */
  HMatrix<spacedim, Number> hmat_new(bct2, std::move(*this));

  if (is_bct2_extended)
    {
      /**
       * Coarsen the new \hmatrix to the original leaf set of \p
       * bct2. Then rebuild its leaf set.
       */
      hmat_new.coarsen_to_partition(target_leaf_set, fixed_rank_k2);
      hmat_new.build_leaf_set();

      /**
       * \mynote{The structure of the \bct associated with the \hmat is still
       * same as before, which has more levels than the \hmat. Therefore, we
       * should prune the \bct to make it consistent with the \hmat.}
       */
      //      std::ofstream out1("target_bct_with_extension.dat");
      //      bct2.write_leaf_set(out1);
      //      out1.close();
      //      std::cout << "=== bct2 with extension ===\n";
      //      std::cout << "Number of nodes: " << bct2.get_node_num()
      //                << "\nDepth: " << bct2.get_depth()
      //                << "\nMax level: " << bct2.get_max_level() << std::endl;
      bct2.prune_to_partition(target_leaf_set, true);
      //      std::ofstream out2("target_bct_after_pruning.dat");
      //      bct2.write_leaf_set(out2);
      //      out2.close();
      //      std::cout << "=== bct2 after pruning ===\n";
      //      std::cout << "Number of nodes: " << bct2.get_node_num()
      //                << "\nDepth: " << bct2.get_depth()
      //                << "\nMax level: " << bct2.get_max_level() << std::endl;
    }

  /**
   * Move the new \hmatrix to the current
   * \hmatrix by shallow assignment.
   */
  (*this) = std::move(hmat_new);
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::remove_hmat_pair_from_mm_product_list(
  const HMatrix<spacedim, Number> *M1,
  const HMatrix<spacedim, Number> *M2)
{
  std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>
  current_hmat_pair(const_cast<HMatrix<spacedim, Number> *>(M1),
                    const_cast<HMatrix<spacedim, Number> *>(M2));

  typename std::vector<std::pair<HMatrix<spacedim, Number> *,
                                 HMatrix<spacedim, Number> *>>::iterator
    matched_hmat_pair_iter =
      std::find(Sigma_P.begin(), Sigma_P.end(), current_hmat_pair);

  if (matched_hmat_pair_iter != Sigma_P.end())
    {
      Sigma_P.erase(matched_hmat_pair_iter);
    }
}


template <int spacedim, typename Number>
void
HMatrix<spacedim, Number>::remove_hmat_pair_from_mm_product_list(
  const std::pair<const HMatrix<spacedim, Number> *,
                  const HMatrix<spacedim, Number> *> &hmat_pair)
{
  typename std::vector<std::pair<HMatrix<spacedim, Number> *,
                                 HMatrix<spacedim, Number> *>>::iterator
    matched_hmat_pair_iter =
      std::find(Sigma_P.begin(), Sigma_P.end(), hmat_pair);

  if (matched_hmat_pair_iter != Sigma_P.end())
    {
      Sigma_P.erase(matched_hmat_pair_iter);
    }
}


template <int spacedim, typename Number>
TreeNodeSplitMode
HMatrix<spacedim, Number>::determine_mm_split_mode_from_Sigma_P()
{
  TreeNodeSplitMode initial_split_mode, current_split_mode;

  size_type counter = 0;
  for (const std::pair<HMatrix<spacedim, Number> *, HMatrix<spacedim, Number> *>
         &hmat_pair : Sigma_P)
    {
      if ((hmat_pair.first->bc_node->get_split_mode() == HorizontalSplitMode &&
           hmat_pair.second->bc_node->get_split_mode() == UnsplitMode) ||
          (hmat_pair.first->bc_node->get_split_mode() == CrossSplitMode &&
           hmat_pair.second->bc_node->get_split_mode() == HorizontalSplitMode))
        {
          current_split_mode = HorizontalSplitMode;
        }
      else if ((hmat_pair.first->bc_node->get_split_mode() == UnsplitMode &&
                hmat_pair.second->bc_node->get_split_mode() ==
                  VerticalSplitMode) ||
               (hmat_pair.first->bc_node->get_split_mode() ==
                  VerticalSplitMode &&
                hmat_pair.second->bc_node->get_split_mode() == CrossSplitMode))
        {
          current_split_mode = VerticalSplitMode;
        }
      else if ((hmat_pair.first->bc_node->get_split_mode() == CrossSplitMode &&
                hmat_pair.second->bc_node->get_split_mode() ==
                  CrossSplitMode) ||
               (hmat_pair.first->bc_node->get_split_mode() ==
                  HorizontalSplitMode &&
                hmat_pair.second->bc_node->get_split_mode() ==
                  VerticalSplitMode))
        {
          current_split_mode = CrossSplitMode;
        }
      else
        {
          Assert(false,
                 ExcMessage(
                   "Inconsistent case met during H-matrix MM multiplication"));
        }

      if (counter == 0)
        {
          initial_split_mode = current_split_mode;
        }
      else if (current_split_mode != initial_split_mode)
        {
          Assert(false,
                 ExcMessage(
                   "Inconsistent case met during H-matrix MM multiplication"));
        }

      counter++;
    }

  return initial_split_mode;
}

#endif /* INCLUDE_HMATRIX_H_ */
