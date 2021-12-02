/**
 * \file hmatrix_support.h
 * \brief Introduction of hmatrix_support.h
 * \date 2021-10-20
 * \author Jihuan Tian
 */
#ifndef INCLUDE_HMATRIX_SUPPORT_H_
#define INCLUDE_HMATRIX_SUPPORT_H_


namespace HMatrixSupport
{
  enum Property
  {
    /// No special properties
    general = 0,
    /// Matrix is symmetric
    symmetric = 1,
    /// Matrix is upper triangular
    upper_triangular = 2,
    /// Matrix is lower triangular
    lower_triangular = 4,
    /// Matrix is diagonal
    diagonal = 6,
  };
}


#endif /* INCLUDE_HMATRIX_SUPPORT_H_ */
