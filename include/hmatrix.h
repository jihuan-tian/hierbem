/**
 * \file hmatrix.h
 * \brief Definition of hierarchical matrix.
 * \ingroup hierarchical_matrices
 * \date 2021-06-06
 * \author Jihuan Tian
 */

#ifndef INCLUDE_HMATRIX_H_
#define INCLUDE_HMATRIX_H_

#include <deal.II/lac/full_matrix.h>

#include <vector>

#include "block_cluster.h"
#include "block_cluster_tree.h"
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
  InitAndCreateHMatrixChildren(
    HMatrix<spacedim1, Number1> *hmat,
    typename BlockClusterTree<spacedim1, Number1>::node_const_pointer_type
                       bc_node,
    const unsigned int fixed_rank_k);

  template <int spacedim1, typename SrcMatrixType, typename Number1>
  friend void
  InitAndCreateHMatrixChildren(
    HMatrix<spacedim1, Number1> *hmat,
    typename BlockClusterTree<spacedim1, Number1>::node_const_pointer_type
                         bc_node,
    const unsigned int   fixed_rank_k,
    const SrcMatrixType &M);

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
   * of a matrix.
   */
  template <typename SrcMatrixType>
  HMatrix(const BlockClusterTree<spacedim, Number> &bct,
          const SrcMatrixType &                     M,
          const unsigned int                        fixed_rank_k = 1);

  /**
   * Construct from a TreeNode in a BlockClusterTree while copying the data of a
   * FullMatrix.
   */
  template <typename SrcMatrixType>
  HMatrix(typename BlockClusterTree<spacedim, Number>::node_const_pointer_type
                               bc_node,
          const SrcMatrixType &M,
          const unsigned int   fixed_rank_k = 1);

  /**
   * Copy constructor
   * @param matrix
   */
  HMatrix(const HMatrix<spacedim, Number> &matrix);

  /**
   * Convert an HMatrix to a full matrix by calling the internal recursive
   * function.
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
   * Destructor which releases the memory by recursion.
   */
  ~HMatrix();

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

  /**
   * Convert or truncate all rank-k matrices in the leaf set to \p new_rank
   * matrices.
   *
   * The full matrices in the leaf set, i.e. those near-field matrices, are kept
   * intact.
   *
   * @param new_rank
   */
  void
  truncate_to_fixed_rank(size_type new_rank);

  /**
   * Calculate matrix-vector multiplication as \f$y = y +
   * M \cdot x\f$.
   *
   * <blockquote>
   * It should be noted that the recursive algorithm for
   * \f$\mathcal{H}\f$-matrix-vector multiplication needs to collect the results
   * from different components in the leaf set and corresponding vector block in
   * \f$x\f$. More importantly, there will be a series of such results
   * contributing to a same block in the vector \f$y\f$. Therefore, if the
   * interface of this function is designed with the parameter \p add as that in
   * the \p vmult function of \p LAPACKFullMatrix, in all recursive calls
   * of \p vmult except the first one, this \p add flag should be set to \p
   * true, irrespective of the original flag value passed into the first call of
   * \p vmult. Hence, we do not include the \p add flag in the \p vmult
   * function.
   * </blockquote>
   * @param y
   * @param x
   */
  void
  vmult(Vector<Number> &y, const Vector<Number> &x) const;

  /**
   * Calculate matrix-vector multiplication as \f$y = y +
   * M^T \cdot x\f$, i.e. the matrix \f$M\f$ is transposed.
   *
   * Because the matrix \f$M\f$ is transposed, the roles for \p row_indices and
   * \p col_indices should be swapped.
   *
   * Also refer to HMatrix::vmult.
   * @param y
   * @param x
   */
  void
  Tvmult(Vector<Number> &y, const Vector<Number> &x) const;

  /**
   * Add the current HMatrix \p A with another HMatrix \p B into \p C, i.e.
   * whole matrix addition instead of addition limited to a specific block,
   * where \p C will be truncated to a fixed rank \p fixed_rank.
   *
   * This algorithm is intrinsically recursive, i.e. the addition of parent
   * HMatrices will perform the addition of each pair of child HMatrices
   * corresponding to a same block cluster. Strictly speaking, this member
   * function \p add is not a recursive function, because the class instance
   * calling \p add changes from parent to child HMatrix.
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

private:
  /**
   * Convert an HMatrix to a full matrix by recursion.
   * @param matrix
   */
  template <typename MatrixType>
  void
  _convertToFullMatrix(MatrixType &M) const;

  /**
   * Matrix type.
   */
  HMatrixType type;

  /**
   * A list of submatrices of type HMatrix.
   */
  std::vector<HMatrix *> submatrices;

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
   * Row indices.
   */
  std::vector<types::global_dof_index> *row_indices;

  /**
   * Column indices.
   */
  std::vector<types::global_dof_index> *col_indices;

  /**
   * Total number of rows in the matrix.
   */
  unsigned int m;

  /**
   * Total number of columns in the matrix.
   */
  unsigned int n;
};


/**
 * Recursively construct the children of an HMatrix with respect to a block
 * cluster tree by starting from a tree node. The matrices in the leaf set are
 * initialized with zero values. The rank of the near field matrices are
 * predefined fixed values.
 * @param hmat Pointer to the current HMatrix.
 * @param bc_node Pointer to a TreeNode in a BlockClusterTree.
 */
template <int spacedim, typename Number = double>
void
InitAndCreateHMatrixChildren(
  HMatrix<spacedim, Number> *                                          hmat,
  typename BlockClusterTree<spacedim, Number>::node_const_pointer_type bc_node,
  const unsigned int fixed_rank_k)
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
  hmat->row_indices = &(bc_node->get_data_reference()
                          .get_tau_node()
                          ->get_data_reference()
                          .get_index_set());
  hmat->col_indices = &(bc_node->get_data_reference()
                          .get_sigma_node()
                          ->get_data_reference()
                          .get_index_set());

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
                                       fixed_rank_k);

          /**
           * Append the initialized child to the list of submatrices of \p hmat.
           */
          hmat->submatrices.push_back(child_hmat);
        }
    }
  else
    {
      /**
       * Update the current matrix type from the block cluster node. When the
       * block cluster belongs to the near field, \p hmat should be represented
       * as a \p LAPACKFullMatrixExt. When the block cluster belongs to the far
       * field, \p hmat should be represented as an \p RkMatrix.
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
 * Recursively construct the children of an HMatrix with respect to a block
 * cluster tree by starting from a tree node. The matrices in the leaf set are
 * initialized with the data in the given matrix \p M. The rank of the near
 * field matrices are predefined fixed values.
 * @param hmat Pointer to the current HMatrix.
 * @param bc_node Pointer to a TreeNode in a BlockClusterTree.
 */
template <int spacedim, typename SrcMatrixType, typename Number = double>
void
InitAndCreateHMatrixChildren(
  HMatrix<spacedim, Number> *                                          hmat,
  typename BlockClusterTree<spacedim, Number>::node_const_pointer_type bc_node,
  const unsigned int   fixed_rank_k,
  const SrcMatrixType &M)
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
    &(bc_node->get_data_reference()
        .get_tau_node()
        ->get_data_reference()
        .get_index_set()));
  hmat->col_indices = const_cast<std::vector<types::global_dof_index> *>(
    &(bc_node->get_data_reference()
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
                                       M);

          /**
           * Append the initialized child to the list of submatrices of \p hmat.
           */
          hmat->submatrices.push_back(child_hmat);
        }
    }
  else
    {
      /**
       * Update the current matrix type from the block cluster node. When the
       * block cluster belongs to the near field, \p hmat should be represented
       * as a \p LAPACKFullMatrixExt. When the block cluster belongs to the far
       * field, \p hmat should be represented as an \p RkMatrix.
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


template <int spacedim, typename Number>
HMatrix<spacedim, Number>::HMatrix()
  : type(UndefinedMatrixType)
  , submatrices(0)
  , rkmatrix(nullptr)
  , fullmatrix(nullptr)
  , bc_node(nullptr)
  , row_indices(nullptr)
  , col_indices(nullptr)
  , m(0)
  , n(0)
{}


template <int spacedim, typename Number>
HMatrix<spacedim, Number>::HMatrix(
  const BlockClusterTree<spacedim, Number> &bct,
  const unsigned int                        fixed_rank_k)
  : type(UndefinedMatrixType)
  , submatrices(0)
  , rkmatrix(nullptr)
  , fullmatrix(nullptr)
  , bc_node(nullptr)
  , row_indices(nullptr)
  , col_indices(nullptr)
  , m(0)
  , n(0)
{
  InitAndCreateHMatrixChildren(this, bct.get_root(), fixed_rank_k);
}


template <int spacedim, typename Number>
HMatrix<spacedim, Number>::HMatrix(
  typename BlockClusterTree<spacedim, Number>::node_const_pointer_type bc_node,
  const unsigned int fixed_rank_k)
  : type(UndefinedMatrixType)
  , submatrices(0)
  , rkmatrix(nullptr)
  , fullmatrix(nullptr)
  , bc_node(nullptr)
  , row_indices(nullptr)
  , col_indices(nullptr)
  , m(0)
  , n(0)
{
  InitAndCreateHMatrixChildren(this, bc_node, fixed_rank_k);
}


template <int spacedim, typename Number>
template <typename SrcMatrixType>
HMatrix<spacedim, Number>::HMatrix(
  const BlockClusterTree<spacedim, Number> &bct,
  const SrcMatrixType &                     M,
  const unsigned int                        fixed_rank_k)
  : type(UndefinedMatrixType)
  , submatrices(0)
  , rkmatrix(nullptr)
  , fullmatrix(nullptr)
  , bc_node(nullptr)
  , row_indices(nullptr)
  , col_indices(nullptr)
  , m(0)
  , n(0)
{
  InitAndCreateHMatrixChildren(this, bct.get_root(), fixed_rank_k, M);
}


template <int spacedim, typename Number>
template <typename SrcMatrixType>
HMatrix<spacedim, Number>::HMatrix(
  typename BlockClusterTree<spacedim, Number>::node_const_pointer_type bc_node,
  const SrcMatrixType &                                                M,
  const unsigned int fixed_rank_k)
  : type(UndefinedMatrixType)
  , submatrices(0)
  , rkmatrix(nullptr)
  , fullmatrix(nullptr)
  , bc_node(nullptr)
  , row_indices(nullptr)
  , col_indices(nullptr)
  , m(0)
  , n(0)
{
  InitAndCreateHMatrixChildren(this, bc_node, fixed_rank_k, M);
}


template <int spacedim, typename Number>
HMatrix<spacedim, Number>::HMatrix(const HMatrix<spacedim, Number> &matrix)
  : type(matrix.type)
  , submatrices(matrix.submatrices)
  , rkmatrix(matrix.rkmatrix)
  , fullmatrix(matrix.fullmatrix)
  , bc_node(matrix.bc_node)
  , row_indices(matrix.row_indices)
  , col_indices(matrix.col_indices)
  , m(matrix.m)
  , n(matrix.n)
{}


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
        for (size_type i = 0; i < m; i++)
          {
            for (size_type j = 0; j < n; j++)
              {
                M(row_indices->at(i), col_indices->at(j)) = (*fullmatrix)(i, j);
              }
          }

        break;
      case RkMatrixType:
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
      case UndefinedMatrixType:
      default:
        Assert(false, ExcInvalidHMatrixType(type));
    }
}


template <int spacedim, typename Number>
HMatrix<spacedim, Number>::~HMatrix()
{
  if (rkmatrix != nullptr)
    {
      delete rkmatrix;
    }

  if (fullmatrix != nullptr)
    {
      delete fullmatrix;
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
HMatrix<spacedim, Number>::truncate_to_fixed_rank(size_type new_rank)
{
  switch (type)
    {
      case HierarchicalMatrixType:
        {
          for (HMatrix *submatrix : submatrices)
            {
              submatrix->truncate_to_fixed_rank(new_rank);
            }

          break;
        }
      case FullMatrixType:
        {
          //          /**
          //           * Create a new RkMatrix on the heap from the current full
          //           matrix.
          //           */
          //          RkMatrix<Number> *rkmatrix_from_fullmatrix =
          //            new RkMatrix<Number>(rank, *fullmatrix);
          //
          //          /**
          //           * Link the created RkMatrix to the current HMatrix node
          //           and delete
          //           * the full matrix, then modify the matrix type.
          //           */
          //          rkmatrix = rkmatrix_from_fullmatrix;
          //          delete fullmatrix;
          //          fullmatrix = nullptr;
          //          type       = RkMatrixType;
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
  /**
   * <b>Work flow</b>
   */
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
         * - Restrict vector x to the current matrix block.
         */
        for (size_type j = 0; j < n; j++)
          {
            local_x(j) = x(col_indices->at(j));
          }

        fullmatrix->vmult(local_y, local_x);

        /**
         * - Merge back the result vector \p local_y to \p y.
         */
        for (size_type i = 0; i < m; i++)
          {
            y(row_indices->at(i)) += local_y(i);
          }

        break;
      case RkMatrixType:
        /**
         * - Restrict vector x to the current matrix block.
         */
        for (size_type j = 0; j < n; j++)
          {
            local_x(j) = x(col_indices->at(j));
          }

        rkmatrix->vmult(local_y, local_x);

        /**
         * - Merge back the result vector \p local_y to \p y.
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
              submatrices.at(i)->add(C.submatrices.at(i),
                                     B.submatrices.at(i),
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

          break;
        }
      case UndefinedMatrixType:
        Assert(false, ExcInvalidHMatrixType(type));
        break;
    }
}

#endif /* INCLUDE_HMATRIX_H_ */
