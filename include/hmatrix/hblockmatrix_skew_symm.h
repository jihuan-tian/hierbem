/**
 * @file hblockmatrix_skew_symm.h
 * @brief Introduction of hblockmatrix_skew_symm.h
 *
 * @date 2022-11-25
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_HMATRIX_HBLOCKMATRIX_SKEW_SYMM_H_
#define HIERBEM_INCLUDE_HMATRIX_HBLOCKMATRIX_SKEW_SYMM_H_

#include "config.h"
#include "hmatrix/hmatrix.h"
#include "linalg.h"

HBEM_NS_OPEN

/**
 * Class for the skew symmetric block \hmat.
 *
 * The block \hmat has the structure
 * \f[
 * \begin{pmatrix}
 * M_{11}   & M_{12} \\
 * (M_{21}) & M_{22}
 * \end{pmatrix},
 * \f]
 * where \f$M_{11}\f$ and \f$M_{22}\f$ are symmetric and positive definite,
 * the matrix block \f$M_{21}=-M_{12}^T\f$ is not stored.
 */
template <int spacedim, typename Number = double>
class HBlockMatrixSkewSymm
{
public:
  using size_type = const typename Vector<Number>::size_type;

  /**
   * Default constructor
   */
  HBlockMatrixSkewSymm()
    : M11(nullptr)
    , M12(nullptr)
    , M22(nullptr)
  {}

  /**
   * Constructor for wrapping \hmat pointers.
   *
   * @param M11
   * @param M12
   * @param M22
   */
  HBlockMatrixSkewSymm(HMatrix<spacedim, Number> *M11,
                       HMatrix<spacedim, Number> *M12,
                       HMatrix<spacedim, Number> *M22)
    : M11(M11)
    , M12(M12)
    , M22(M22)
  {}

  /**
   * Copy constructor
   */
  HBlockMatrixSkewSymm(const HBlockMatrixSkewSymm<spacedim, Number> &block_mat);

  /**
   * Assignment operator
   */
  HBlockMatrixSkewSymm<spacedim, Number> &
  operator=(const HBlockMatrixSkewSymm<spacedim, Number> &block_mat);

  /**
   * Matrix-vector multiplication
   *
   * @param y
   * @param x
   */
  void
  vmult(Vector<Number> &y, const Vector<Number> &x) const;

private:
  HMatrix<spacedim, Number> *M11;
  HMatrix<spacedim, Number> *M12;
  HMatrix<spacedim, Number> *M22;
};


template <int spacedim, typename Number>
HBlockMatrixSkewSymm<spacedim, Number>::HBlockMatrixSkewSymm(
  const HBlockMatrixSkewSymm<spacedim, Number> &block_mat)
  : M11(block_mat.M11)
  , M12(block_mat.M12)
  , M22(block_mat.M22)
{}


template <int spacedim, typename Number>
HBlockMatrixSkewSymm<spacedim, Number> &
HBlockMatrixSkewSymm<spacedim, Number>::operator=(
  const HBlockMatrixSkewSymm<spacedim, Number> &block_mat)
{
  M11 = block_mat.M11;
  M12 = block_mat.M12;
  M22 = block_mat.M22;

  return (*this);
}


template <int spacedim, typename Number>
void
HBlockMatrixSkewSymm<spacedim, Number>::vmult(Vector<Number>       &y,
                                              const Vector<Number> &x) const
{
  // Verify the dimensions of matrices and vectors should match.
  AssertDimension(y.size(), M11->get_m() + M22->get_m());
  AssertDimension(x.size(), M11->get_n() + M12->get_n());

  // Verify the dimensions of matrix blocks should match.
  AssertDimension(M11->get_m(), M12->get_m());
  AssertDimension(M12->get_n(), M22->get_n());

  // Verify the whole block matrix should be square.
  AssertDimension(y.size(), x.size());

  // The diagonal blocks should be symmetric.
  Assert(M11->get_property() == HMatrixSupport::Property::symmetric,
         ExcInvalidHMatrixProperty(M11->get_property()));
  Assert(M22->get_property() == HMatrixSupport::Property::symmetric,
         ExcInvalidHMatrixProperty(M22->get_property()));

  // Split the input vector @p x.
  const size_type n1 = M11->get_n();
  const size_type n2 = M12->get_n();
  Vector<Number>  x1(n1);
  Vector<Number>  x2(n2);

  copy_vector(x1, 0, x, 0, n1);
  copy_vector(x2, 0, x, n1, n2);

  // Split the result vector.
  Vector<Number> y1(n1);
  Vector<Number> y2(n2);

  M11->vmult(y1, x1, HMatrixSupport::Property::symmetric);
  M12->vmult(y1, x2);
  M12->Tvmult(y2, -1.0, x1);
  M22->vmult(y2, x2, HMatrixSupport::Property::symmetric);

  // Merge @p y1 and @p y2 into the result vector @p y.
  copy_vector(y, 0, y1, 0, n1);
  copy_vector(y, n1, y2, 0, n2);
}

HBEM_NS_CLOSE

#endif /* HIERBEM_INCLUDE_HMATRIX_HBLOCKMATRIX_SKEW_SYMM_H_ */
