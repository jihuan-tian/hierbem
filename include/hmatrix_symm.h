/**
 * @file hmatrix_symmetric.h
 * @brief Symmetric \hmat by inheriting from @p HMatrix.
 *
 * At the moment, its purpose is to wrap some member functions of @p HMatrix,
 * such as @p HMatrix::vmult, which will be used as preconditioner in
 * @p SolverCG.
 *
 * @date 2022-10-27
 * @author Jihuan Tian
 */
#ifndef INCLUDE_HMATRIX_SYMM_H_
#define INCLUDE_HMATRIX_SYMM_H_


#include "hmatrix.h"

template <int spacedim, typename Number = double>
class HMatrixSymm : public HMatrix<spacedim, Number>
{
public:
  /**
   * Default constructor
   */
  HMatrixSymm();

  /**
   * Construct the hierarchical structure without data from the root node of a
   * BlockClusterTree.
   */
  HMatrixSymm(const BlockClusterTree<spacedim, Number> &bct,
              const unsigned int                        fixed_rank_k = 1);

  /**
   * Construct the hierarchical structure without data from a TreeNode in a
   * BlockClusterTree.
   */
  HMatrixSymm(
    typename BlockClusterTree<spacedim, Number>::node_const_pointer_type
                       bc_node,
    const unsigned int fixed_rank_k = 1);

  /**
   * Construct from the root node of a BlockClusterTree while copying the data
   * of a <strong>global</strong> full matrix, which is created on the complete
   * block cluster \f$I \times J\f$.
   *
   * \mynote{Since this \hmatrix is the global matrix, i.e. on the same level
   * as the global full matrix @p M in the \hmatrix hierarchy, its block type
   * is set to @p HMatrixSupport::diagonal_block by default.}
   */
  HMatrixSymm(const BlockClusterTree<spacedim, Number> &bct,
              const LAPACKFullMatrixExt<Number> &       M,
              const unsigned int                        fixed_rank_k);

  /**
   * Construct from the root node of a BlockClusterTree while copying the data
   * of a <strong>global</strong> full matrix, which is created on the complete
   * block cluster \f$I \times J\f$.
   *
   * This version has no rank truncation.
   *
   * \mynote{Since this \hmatrix is the global matrix, i.e. on the same level
   * as the global full matrix @p M in the \hmatrix hierarchy, its block type
   * is set to @p HMatrixSupport::diagonal_block by default.}
   */
  HMatrixSymm(const BlockClusterTree<spacedim, Number> &bct,
              const LAPACKFullMatrixExt<Number> &       M);

  /**
   * Construct from a TreeNode in a BlockClusterTree while copying the data of a
   * global full matrix, which is created on the complete block cluster \f$I
   * \times J\f$.
   *
   * \mynote{The current \hmatrix to be built may only be a matrix block in the
   * global matrix, while the full matrix @p M is global, i.e., the \hmatrix
   * and the full matrix @p M are not on a same level in the \hmatrix
   * hierarchy. But because the current \hmatrix is symmetric, its block type is
   * still diagonal and its property should be symmetric.}
   */
  HMatrixSymm(
    typename BlockClusterTree<spacedim, Number>::node_const_pointer_type
                                       bc_node,
    const LAPACKFullMatrixExt<Number> &M,
    const unsigned int                 fixed_rank_k);

  /**
   * Construct from a TreeNode in a BlockClusterTree while copying the data of a
   * global full matrix, which is created on the complete block cluster \f$I
   * \times J\f$.
   *
   * This version has no rank truncation.
   *
   * \mynote{The current \hmatrix to be built may only be a matrix block in the
   * global matrix, while the full matrix @p M is global, i.e., the \hmatrix
   * and the full matrix @p M are not on a same level in the \hmatrix
   * hierarchy. But because the current \hmatrix is symmetric, its block type is
   * still diagonal and its property should be symmetric.}
   */
  HMatrixSymm(
    typename BlockClusterTree<spacedim, Number>::node_const_pointer_type
                                       bc_node,
    const LAPACKFullMatrixExt<Number> &M);

  /**
   * Deep copy constructor
   *
   * @param H
   */
  HMatrixSymm(const HMatrixSymm<spacedim, Number> &H);

  /**
   * Shallow copy constructor
   *
   * @param H
   */
  HMatrixSymm(HMatrixSymm<spacedim, Number> &&H) noexcept;

  /**
   * Deep assignment operator
   *
   * @param
   * @return
   */
  HMatrixSymm<spacedim, Number> &
  operator=(const HMatrixSymm<spacedim, Number> &H);

  /**
   * Shallow assignment operator
   * @param
   * @return
   */
  HMatrixSymm<spacedim, Number> &
  operator=(HMatrixSymm<spacedim, Number> &&H) noexcept;

  /**
   * Calculate matrix-vector multiplication as \f$y = y + M \cdot x\f$.
   *
   * @param y
   * @param x
   */
  void
  vmult(Vector<Number> &y, const Vector<Number> &x) const;
};


template <int spacedim, typename Number>
HMatrixSymm<spacedim, Number>::HMatrixSymm()
  : HMatrix<spacedim, Number>()
{}


template <int spacedim, typename Number>
HMatrixSymm<spacedim, Number>::HMatrixSymm(
  const BlockClusterTree<spacedim, Number> &bct,
  const unsigned int                        fixed_rank_k)
  : HMatrix<spacedim, Number>(bct,
                              fixed_rank_k,
                              HMatrixSupport::Property::symmetric,
                              HMatrixSupport::BlockType::diagonal_block)
{}


template <int spacedim, typename Number>
HMatrixSymm<spacedim, Number>::HMatrixSymm(
  typename BlockClusterTree<spacedim, Number>::node_const_pointer_type bc_node,
  const unsigned int fixed_rank_k)
  : HMatrix<spacedim, Number>(bc_node,
                              fixed_rank_k,
                              HMatrixSupport::Property::symmetric,
                              HMatrixSupport::BlockType::diagonal_block)
{}


template <int spacedim, typename Number>
HMatrixSymm<spacedim, Number>::HMatrixSymm(
  const BlockClusterTree<spacedim, Number> &bct,
  const LAPACKFullMatrixExt<Number> &       M,
  const unsigned int                        fixed_rank_k)
  : HMatrix<spacedim, Number>(bct,
                              M,
                              fixed_rank_k,
                              HMatrixSupport::BlockType::diagonal_block)
{
  Assert(M.get_property() == LAPACKSupport::Property::symmetric,
         ExcInternalError());
}


template <int spacedim, typename Number>
HMatrixSymm<spacedim, Number>::HMatrixSymm(
  const BlockClusterTree<spacedim, Number> &bct,
  const LAPACKFullMatrixExt<Number> &       M)
  : HMatrix<spacedim, Number>(bct, M, HMatrixSupport::BlockType::diagonal_block)
{
  Assert(M.get_property() == LAPACKSupport::Property::symmetric,
         ExcInternalError());
}


template <int spacedim, typename Number>
HMatrixSymm<spacedim, Number>::HMatrixSymm(
  typename BlockClusterTree<spacedim, Number>::node_const_pointer_type bc_node,
  const LAPACKFullMatrixExt<Number> &                                  M,
  const unsigned int fixed_rank_k)
  : HMatrix<spacedim, Number>(bc_node,
                              M,
                              fixed_rank_k,
                              HMatrixSupport::Property::symmetric,
                              HMatrixSupport::BlockType::diagonal_block)
{
  Assert(M.get_property() == LAPACKSupport::Property::symmetric,
         ExcInternalError());
}


template <int spacedim, typename Number>
HMatrixSymm<spacedim, Number>::HMatrixSymm(
  typename BlockClusterTree<spacedim, Number>::node_const_pointer_type bc_node,
  const LAPACKFullMatrixExt<Number> &                                  M)
  : HMatrix<spacedim, Number>(bc_node,
                              M,
                              HMatrixSupport::Property::symmetric,
                              HMatrixSupport::BlockType::diagonal_block)
{
  Assert(M.get_property() == LAPACKSupport::Property::symmetric,
         ExcInternalError());
}


template <int spacedim, typename Number>
HMatrixSymm<spacedim, Number>::HMatrixSymm(
  const HMatrixSymm<spacedim, Number> &H)
  : HMatrix<spacedim, Number>(H)
{}


template <int spacedim, typename Number>
HMatrixSymm<spacedim, Number>::HMatrixSymm(
  HMatrixSymm<spacedim, Number> &&H) noexcept
  : HMatrix<spacedim, Number>(std::move(H))
{}


template <int spacedim, typename Number>
HMatrixSymm<spacedim, Number> &
HMatrixSymm<spacedim, Number>::operator=(const HMatrixSymm<spacedim, Number> &H)
{
  HMatrix<spacedim, Number>::operator=(H);

  return (*this);
}


template <int spacedim, typename Number>
HMatrixSymm<spacedim, Number> &
HMatrixSymm<spacedim, Number>::
operator=(HMatrixSymm<spacedim, Number> &&H) noexcept
{
  HMatrix<spacedim, Number>::operator=(std::move(H));

  return (*this);
}


template <int spacedim, typename Number>
void
HMatrixSymm<spacedim, Number>::vmult(Vector<Number> &      y,
                                     const Vector<Number> &x) const
{
  /**
   * Call the member function @p vmult in the parent class.
   */
  this->HMatrix<spacedim, Number>::vmult(y,
                                         x,
                                         HMatrixSupport::Property::symmetric);
}

#endif /* INCLUDE_HMATRIX_SYMM_H_ */
