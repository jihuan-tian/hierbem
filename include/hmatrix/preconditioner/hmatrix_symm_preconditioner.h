// Copyright (C) 2022-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file hmatrix_symm_preconditioner.h
 * @brief Symmetric \hmat by inheriting from @p HMatrix, which is used as the preconditioner for SPD \hmatrices.
 *
 * At the moment, its purpose is to wrap some member functions of @p HMatrix,
 * such as @p HMatrix::vmult, which will be used as preconditioner in
 * @p SolverCGGeneral.
 *
 * @date 2022-10-27
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_HMATRIX_PRECONDITIONER_HMATRIX_SYMM_PRECONDITIONER_H_
#define HIERBEM_INCLUDE_HMATRIX_PRECONDITIONER_HMATRIX_SYMM_PRECONDITIONER_H_

#include <deal.II/base/numbers.h>

#include "config.h"
#include "hmatrix/hmatrix.h"

HBEM_NS_OPEN

template <int spacedim, typename Number = double>
class HMatrixSymmPreconditioner : public HMatrix<spacedim, Number>
{
public:
  using real_type = typename numbers::NumberTraits<Number>::real_type;

  /**
   * Default constructor
   */
  HMatrixSymmPreconditioner();

  /**
   * Construct the hierarchical structure without data from the root node of a
   * BlockClusterTree.
   */
  HMatrixSymmPreconditioner(const BlockClusterTree<spacedim, real_type> &bct,
                            const unsigned int fixed_rank_k = 1);

  /**
   * Construct the hierarchical structure without data from a TreeNode in a
   * BlockClusterTree.
   */
  HMatrixSymmPreconditioner(
    typename BlockClusterTree<spacedim, real_type>::node_const_pointer_type
                       bc_node,
    const unsigned int fixed_rank_k = 1);

  /**
   * Construct from the root node of a BlockClusterTree while copying the data
   * of a <strong>global</strong> full matrix, which is created on the
   * complete block cluster \f$I \times J\f$.
   *
   * \mynote{Since this \hmatrix is the global matrix, i.e. on the same level
   * as the global full matrix @p M in the \hmatrix hierarchy, its block type
   * is set to @p HMatrixSupport::diagonal_block by default.}
   */
  HMatrixSymmPreconditioner(const BlockClusterTree<spacedim, real_type> &bct,
                            const LAPACKFullMatrixExt<Number>           &M,
                            const unsigned int fixed_rank_k);

  /**
   * Construct from the root node of a BlockClusterTree while copying the data
   * of a <strong>global</strong> full matrix, which is created on the
   * complete block cluster \f$I \times J\f$.
   *
   * This version has no rank truncation.
   *
   * \mynote{Since this \hmatrix is the global matrix, i.e. on the same level
   * as the global full matrix @p M in the \hmatrix hierarchy, its block type
   * is set to @p HMatrixSupport::diagonal_block by default.}
   */
  HMatrixSymmPreconditioner(const BlockClusterTree<spacedim, real_type> &bct,
                            const LAPACKFullMatrixExt<Number>           &M);

  /**
   * Construct from a TreeNode in a BlockClusterTree while copying the data of
   * a global full matrix, which is created on the complete block cluster \f$I
   * \times J\f$.
   *
   * \mynote{The current \hmatrix to be built may only be a matrix block in
   * the
   * global matrix, while the full matrix @p M is global, i.e., the \hmatrix
   * and the full matrix @p M are not on a same level in the \hmatrix
   * hierarchy. But because the current \hmatrix is symmetric, its block type
   * is still diagonal and its property should be symmetric.}
   */
  HMatrixSymmPreconditioner(
    typename BlockClusterTree<spacedim, real_type>::node_const_pointer_type
                                       bc_node,
    const LAPACKFullMatrixExt<Number> &M,
    const unsigned int                 fixed_rank_k);

  /**
   * Construct from a TreeNode in a BlockClusterTree while copying the data of
   * a global full matrix, which is created on the complete block cluster \f$I
   * \times J\f$.
   *
   * This version has no rank truncation.
   *
   * \mynote{The current \hmatrix to be built may only be a matrix block in
   * the
   * global matrix, while the full matrix @p M is global, i.e., the \hmatrix
   * and the full matrix @p M are not on a same level in the \hmatrix
   * hierarchy. But because the current \hmatrix is symmetric, its block type
   * is still diagonal and its property should be symmetric.}
   */
  HMatrixSymmPreconditioner(
    typename BlockClusterTree<spacedim, real_type>::node_const_pointer_type
                                       bc_node,
    const LAPACKFullMatrixExt<Number> &M);

  /**
   * Deep copy constructor
   *
   * @param H
   */
  HMatrixSymmPreconditioner(
    const HMatrixSymmPreconditioner<spacedim, Number> &H);

  /**
   * Shallow copy constructor
   *
   * @param H
   */
  HMatrixSymmPreconditioner(
    HMatrixSymmPreconditioner<spacedim, Number> &&H) noexcept;

  /**
   * Deep assignment operator
   *
   * @param
   * @return
   */
  HMatrixSymmPreconditioner<spacedim, Number> &
  operator=(const HMatrixSymmPreconditioner<spacedim, Number> &H);

  /**
   * Shallow assignment operator
   * @param
   * @return
   */
  HMatrixSymmPreconditioner<spacedim, Number> &
  operator=(HMatrixSymmPreconditioner<spacedim, Number> &&H) noexcept;

  /**
   * Deep copy constructor, copy from a @p HMatrix.
   *
   * @param H
   */
  HMatrixSymmPreconditioner(const HMatrix<spacedim, Number> &H);

  /**
   * Shallow copy constructor, copy from a @p HMatrix.
   *
   * @param H
   */
  HMatrixSymmPreconditioner(HMatrix<spacedim, Number> &&H) noexcept;

  /**
   * Deep assignment operator, assign from a @p HMatrix.
   *
   * @param H
   * @return
   */
  HMatrixSymmPreconditioner<spacedim, Number> &
  operator=(const HMatrix<spacedim, Number> &H);

  /**
   * Shallow assignment operator, assign from a @p HMatrix.
   *
   * @param H
   * @return
   */
  HMatrixSymmPreconditioner<spacedim, Number> &
  operator=(HMatrix<spacedim, Number> &&H) noexcept;

  /**
   * Calculate matrix-vector multiplication as \f$y = M^{-1} \cdot x\f$, which
   * is actually applying the preconditioner to the input vector.
   *
   * @param y
   * @param x
   */
  virtual void
  vmult(Vector<Number> &y, const Vector<Number> &x) const;
};


template <int spacedim, typename Number>
HMatrixSymmPreconditioner<spacedim, Number>::HMatrixSymmPreconditioner()
  : HMatrix<spacedim, Number>()
{}


template <int spacedim, typename Number>
HMatrixSymmPreconditioner<spacedim, Number>::HMatrixSymmPreconditioner(
  const BlockClusterTree<spacedim, real_type> &bct,
  const unsigned int                           fixed_rank_k)
  : HMatrix<spacedim, Number>(bct,
                              fixed_rank_k,
                              HMatrixSupport::Property::symmetric,
                              HMatrixSupport::BlockType::diagonal_block)
{}


template <int spacedim, typename Number>
HMatrixSymmPreconditioner<spacedim, Number>::HMatrixSymmPreconditioner(
  typename BlockClusterTree<spacedim, real_type>::node_const_pointer_type
                     bc_node,
  const unsigned int fixed_rank_k)
  : HMatrix<spacedim, Number>(bc_node,
                              fixed_rank_k,
                              HMatrixSupport::Property::symmetric,
                              HMatrixSupport::BlockType::diagonal_block)
{}


template <int spacedim, typename Number>
HMatrixSymmPreconditioner<spacedim, Number>::HMatrixSymmPreconditioner(
  const BlockClusterTree<spacedim, real_type> &bct,
  const LAPACKFullMatrixExt<Number>           &M,
  const unsigned int                           fixed_rank_k)
  : HMatrix<spacedim, Number>(bct,
                              M,
                              fixed_rank_k,
                              HMatrixSupport::BlockType::diagonal_block)
{
  Assert(M.get_property() == LAPACKSupport::Property::symmetric,
         ExcInternalError());
}


template <int spacedim, typename Number>
HMatrixSymmPreconditioner<spacedim, Number>::HMatrixSymmPreconditioner(
  const BlockClusterTree<spacedim, real_type> &bct,
  const LAPACKFullMatrixExt<Number>           &M)
  : HMatrix<spacedim, Number>(bct, M, HMatrixSupport::BlockType::diagonal_block)
{
  Assert(M.get_property() == LAPACKSupport::Property::symmetric,
         ExcInternalError());
}


template <int spacedim, typename Number>
HMatrixSymmPreconditioner<spacedim, Number>::HMatrixSymmPreconditioner(
  typename BlockClusterTree<spacedim, real_type>::node_const_pointer_type
                                     bc_node,
  const LAPACKFullMatrixExt<Number> &M,
  const unsigned int                 fixed_rank_k)
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
HMatrixSymmPreconditioner<spacedim, Number>::HMatrixSymmPreconditioner(
  typename BlockClusterTree<spacedim, real_type>::node_const_pointer_type
                                     bc_node,
  const LAPACKFullMatrixExt<Number> &M)
  : HMatrix<spacedim, Number>(bc_node,
                              M,
                              HMatrixSupport::Property::symmetric,
                              HMatrixSupport::BlockType::diagonal_block)
{
  Assert(M.get_property() == LAPACKSupport::Property::symmetric,
         ExcInternalError());
}


template <int spacedim, typename Number>
HMatrixSymmPreconditioner<spacedim, Number>::HMatrixSymmPreconditioner(
  const HMatrixSymmPreconditioner<spacedim, Number> &H)
  : HMatrix<spacedim, Number>(H)
{}


template <int spacedim, typename Number>
HMatrixSymmPreconditioner<spacedim, Number>::HMatrixSymmPreconditioner(
  HMatrixSymmPreconditioner<spacedim, Number> &&H) noexcept
  : HMatrix<spacedim, Number>(std::move(H))
{}


template <int spacedim, typename Number>
HMatrixSymmPreconditioner<spacedim, Number> &
HMatrixSymmPreconditioner<spacedim, Number>::operator=(
  const HMatrixSymmPreconditioner<spacedim, Number> &H)
{
  HMatrix<spacedim, Number>::operator=(H);

  return (*this);
}


template <int spacedim, typename Number>
HMatrixSymmPreconditioner<spacedim, Number> &
HMatrixSymmPreconditioner<spacedim, Number>::operator=(
  HMatrixSymmPreconditioner<spacedim, Number> &&H) noexcept
{
  HMatrix<spacedim, Number>::operator=(std::move(H));

  return (*this);
}


template <int spacedim, typename Number>
HMatrixSymmPreconditioner<spacedim, Number>::HMatrixSymmPreconditioner(
  const HMatrix<spacedim, Number> &H)
  : HMatrix<spacedim, Number>(H)
{}


template <int spacedim, typename Number>
HMatrixSymmPreconditioner<spacedim, Number>::HMatrixSymmPreconditioner(
  HMatrix<spacedim, Number> &&H) noexcept
  : HMatrix<spacedim, Number>(std::move(H))
{}


template <int spacedim, typename Number>
HMatrixSymmPreconditioner<spacedim, Number> &
HMatrixSymmPreconditioner<spacedim, Number>::operator=(
  const HMatrix<spacedim, Number> &H)
{
  HMatrix<spacedim, Number>::operator=(H);

  return (*this);
}


template <int spacedim, typename Number>
HMatrixSymmPreconditioner<spacedim, Number> &
HMatrixSymmPreconditioner<spacedim, Number>::operator=(
  HMatrix<spacedim, Number> &&H) noexcept
{
  HMatrix<spacedim, Number>::operator=(std::move(H));

  return (*this);
}


template <int spacedim, typename Number>
void
HMatrixSymmPreconditioner<spacedim, Number>::vmult(
  Vector<Number>       &y,
  const Vector<Number> &x) const
{
  this->solve_cholesky(y, x);
}

HBEM_NS_CLOSE

#endif /* HIERBEM_INCLUDE_HMATRIX_PRECONDITIONER_HMATRIX_SYMM_PRECONDITIONER_H_ \
        */
