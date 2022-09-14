/**
 * \file hmatrix_support.h
 * \brief Adapted from deal.ii/lac/lapack_support.h and define some features for
 * an \hmatrix.
 *
 * \ingroup hierarchical_matrices
 *
 * \date 2021-10-20
 * \author Jihuan Tian
 */
#ifndef INCLUDE_HMATRIX_SUPPORT_H_
#define INCLUDE_HMATRIX_SUPPORT_H_

#include <deal.II/base/exceptions.h>

#include <array>
#include <string>

using namespace dealii;

namespace HMatrixSupport
{
  /**
   * Most functions that one can apply to an \hmatrix (e.g., by calling the
   * member functions of this class) change the content of the \hmatrix in some
   * ways. For example, they may invert the matrix, or may replace it by a
   * matrix whose columns represent the eigenvectors of the original content of
   * the matrix. The elements of this enumeration are therefore used to track
   * what is currently being stored by this object.
   *
   * \comment{This above documentation is extracted from deal.ii.}
   */
  enum State
  {
    /// Contents is actually a matrix.
    matrix,
    /// Contents is the inverse of a matrix.
    inverse_matrix,
    /// Contents is an LU decomposition.
    lu,
    /// Contents is a Cholesky decomposition.
    cholesky,
    /// Eigenvalue vector is filled
    eigenvalues,
    /// Matrix contains singular value decomposition,
    svd,
    /// Matrix is the inverse of a singular value decomposition
    inverse_svd,
    /// Contents is something useless.
    unusable = 0x8000
  };


  /**
   * Function printing the name of a State.
   */
  inline const char *
  state_name(State s)
  {
    switch (s)
      {
        case matrix:
          return "matrix";
        case inverse_matrix:
          return "inverse matrix";
        case lu:
          return "lu decomposition";
        case cholesky:
          return "cholesky decomposition";
        case eigenvalues:
          return "eigenvalues";
        case svd:
          return "svd";
        case inverse_svd:
          return "inverse_svd";
        case unusable:
          return "unusable";
        default:
          return "unknown";
      }
  }


  /**
   * A matrix can have certain features allowing for optimization, but hard to
   * test. These are listed here.
   *
   * <ul>
   * <li>@p general: the \hmatrix is general and all the \hmatnodes in the
   * \hmatrix hierarchy are created and allocated with memory.
   * <li>@p symmetric: the \hmatrix is symmetric and only the diagonal blocks
   * in the near field and matrix blocks in the lower triangular part are
   * created and allocated with memory. For the matrix blocks in the upper
   * triangular part, basic matrix information, such as dimension, is still
   * maintained but the memory for data is not allocated. The matrix blocks in
   * the upper triangular part is assumed to be symmetric to those related
   * blocks in the lower triangular part.
   * <li>@p upper_triangular: only the matrix blocks in the upper triangular
   * part and the diagonal blocks in the near field are created and allocated
   * with memory.The matrix blocks in the lower triangular part are zero valued.
   * Their basic matrix information, such as dimension, is still maintained but
   * the memory for the data is not allocated.
   * <li>@p lower_triangular: only the matrix blocks in the lower triangular
   * part and the diagonal blocks in the near field are created and allocated
   * with memory.The matrix blocks in the upper triangular part are zero valued.
   * Their basic matrix information, such as dimension, is still stored but the
   * memory for the data is not allocated.
   * </ul>
   */
  enum Property
  {
    /// No special properties
    general = 0,
    /// Matrix is symmetric
    symmetric = 1,
    /// Matrix is upper triangular
    upper_triangular = 2,
    /// Matrix is lower triangular
    lower_triangular = 4
  };


  /**
   * Function printing the name of a Property.
   */
  inline const char *
  property_name(const Property s)
  {
    switch (s)
      {
        case general:
          return "general";
        case symmetric:
          return "symmetric";
        case upper_triangular:
          return "upper triangular";
        case lower_triangular:
          return "lower triangular";
      }

    Assert(false, ExcNotImplemented());
    return "invalid";
  }


  /**
   * Type of the matrix block, which can be diagonal block, upper triangular
   * block or lower triangular block.
   *
   * \mynote{N.B. Here the diagonal/upper-triangular/lower-triangular block does
   * not mean the matrix block itself is a
   * diagonal/upper-triangular/lower-triangular matrix, but means that the
   * matrix block belongs to the diagonal/upper-triangular/lower-triangular part
   * of the original matrix.
   *
   * For an \hmatrix built with respect to a quad-\bct,
   * 1. the top level \hmatnode is a diagonal block;
   * 2. if the current \hmatnode is a diagonal block and has submatrices, the
   * first and last submatrices are still diagonal blocks, while the second
   * submatrix is a upper triangular block and the third submatrix is a lower
   * triangular block;
   * 3. if the current \hmatnode is a upper (lower) triangular block and has
   * submatrices, all of them are upper (lower) triangular blocks.}
   */
  enum BlockType
  {
    /// Undefined block type
    undefined_block = 0,
    /// Diagonal block
    diagonal_block = 1,
    /// Upper triangular block
    upper_triangular_block = 2,
    /// Lower triangular block
    lower_triangular_block = 4
  };


  /**
   * Function printing the name of the block type.
   */
  inline const char *
  block_type_name(const BlockType b)
  {
    switch (b)
      {
        case undefined_block:
          return "undefined block";
        case diagonal_block:
          return "diagonal block";
        case upper_triangular_block:
          return "upper triangular block";
        case lower_triangular_block:
          return "lower triangular block";
      }

    Assert(false, ExcNotImplemented());
    return "invalid";
  }

  /**
   * Determine the block types of the four submatrices from the given block type
   * of their parent \hmatrix.
   */
  void
  infer_submatrix_block_types_from_parent_hmat(
    const BlockType                           parent_hmat_block_type,
    std::array<HMatrixSupport::BlockType, 4> &submatrix_block_types);
} // namespace HMatrixSupport


#endif /* INCLUDE_HMATRIX_SUPPORT_H_ */
