// Copyright (C) 2023-2026 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file full_matrix_wrapper.h
 * @brief Definition of the class @p FullMatrixWrapper
 *
 * @date 2023-02-09
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_LINEAR_ALGEBRA_FULL_MATRIX_WRAPPER_H_
#define HIERBEM_INCLUDE_LINEAR_ALGEBRA_FULL_MATRIX_WRAPPER_H_

#include <stdio.h>
#include <stdlib.h>

#include <cassert>

#include "config.h"
#include "vector_wrapper.h"

HBEM_NS_OPEN

namespace PlatformShared
{
  /**
   * Simple full matrix class wrapping a pointer to an array of values, which
   * can be used both on the host and the device.
   */
  template <typename T = double>
  class FullMatrixWrapper
  {
  public:
    using pointer         = T *;
    using const_pointer   = const T *;
    using reference       = T &;
    using const_reference = const T &;
    using size_type       = std::size_t;

    HBEM_ATTR_HOST HBEM_ATTR_DEV
    FullMatrixWrapper();

    /**
     * Constructor from a pointer allocated with memory.
     *
     * @param p Pointer to the array of values
     * @param _rows Number of rows
     * @param _cols Number of columns
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV
    FullMatrixWrapper(pointer p, const size_type _rows, const size_type _cols);

    /**
     * Reinitialize the current matrix wrapper with a new pointer and dimension.
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV void
    reinit(pointer p, const size_type _rows, const size_type _cols);

    // Copy constructor is deleted, because this wrapper class does not allocate
    // memory by itself.
    HBEM_ATTR_HOST HBEM_ATTR_DEV
    FullMatrixWrapper(const FullMatrixWrapper<T> &mat) = delete;

    HBEM_ATTR_HOST HBEM_ATTR_DEV
    FullMatrixWrapper(FullMatrixWrapper<T> &&mat);

    HBEM_ATTR_HOST HBEM_ATTR_DEV FullMatrixWrapper<T>                              &
    operator=(const FullMatrixWrapper<T> &mat);

    HBEM_ATTR_HOST HBEM_ATTR_DEV FullMatrixWrapper<T>                              &
    operator=(FullMatrixWrapper<T> &&mat);

    /**
     * Fill the matrix from a pointer to source data.
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV void
    fill(const_pointer data);

    /**
     * Get the number of rows.
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV size_type
    m() const;

    /**
     * Get the number of columns.
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV size_type
    n() const;

    /**
     * Get the total number of matrix entries.
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV size_type
    n_elements() const;

    /**
     * Get the size of the i'th dimension.
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV size_type
    size(const size_type i) const;

    /**
     * Get the internal data pointer.
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV pointer
    data();

    /**
     * Get the internal data pointer (const version).
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV const_pointer
    data() const;

    /**
     * Get the reference to the matrix entry \f$(i,j)\f$.
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV reference
    operator()(const size_type i,
               const size_type j,
               const bool      is_column_major = true);

    /**
     * Get the const reference to the matrix entry \f$(i,j)\f$.
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV const_reference
    operator()(const size_type i,
               const size_type j,
               const bool      is_column_major = true) const;

    /**
     * Perform matrix-vector multiplication \f$w = Av\f$. When @p adding is
     * true, compute \f$w = Av + w\f$.
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV void
    vmult(VectorWrapper<T>       &w,
          const VectorWrapper<T> &v,
          const bool              adding = false) const;

    /**
     * Perform matrix-vector multiplication \f$w = A^Tv\f$. When @p adding is
     * true, compute \f$w = A^Tv + w\f$.
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV void
    Tvmult(VectorWrapper<T>       &w,
           const VectorWrapper<T> &v,
           const bool              adding = false) const;

    /**
     * Perform matrix-matrix multiplication \f$C = A B\f$. When @p adding is
     * true, compute \f$C = C + A B\f$.
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV void
    mmult(FullMatrixWrapper<T>       &C,
          const FullMatrixWrapper<T> &B,
          const bool                  adding = false) const;

    /**
     * Perform matrix-matrix multiplication \f$C = A^T B\f$. When @p adding is
     * true, compute \f$C = C + A^T B\f$.
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV void
    Tmmult(FullMatrixWrapper<T>       &C,
           const FullMatrixWrapper<T> &B,
           const bool                  adding = false) const;

    /**
     * Perform matrix-matrix multiplication \f$C = A B^T\f$. When @p adding is
     * true, compute \f$C = C + A B^T\f$.
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV void
    mTmult(FullMatrixWrapper<T>       &C,
           const FullMatrixWrapper<T> &B,
           const bool                  adding = false) const;

    /**
     * Compute the inverse of the matrix using Gauss elimination.
     *
     * \alert{Before calling this function, the output matrix @p M_inv should
     * be reinitialized to zero.}
     *
     * @param M_inv [out] Inverse matrix
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV void
    invert_by_gauss_elim(FullMatrixWrapper<T> &M_inv);

    /**
     * Assign the inverse of the given @p 2x2 matrix @p M to @p *this.
     *
     * Let
     * \f[
     * M = \begin{pmatrix}
     * a & b \\
     * c & d
     * \end{pmatrix}
     * \f]
     *
     * \f[
     * M^{-1} = \frac{1}{\abs{M}} \begin{pmatrix}
     * d & -b \\
     * -c & a
     * \end{pmatrix}
     * \f]
     *
     * @param M [in] Input matrix
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV void
    invert2x2(const FullMatrixWrapper<T> &M);

    /**
     * Scale all matrix entries by a factor.
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV FullMatrixWrapper<T>                              &
    operator*=(const T factor);

    /**
     * Divide all matrix entires by the factor.
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV FullMatrixWrapper<T>                              &
    operator/=(const T factor);

    /**
     * Calculate the determinant of the current \f$2\times 2\f$ matrix.
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV T
    determinant2x2() const;

    /**
     * Calculate the determinant of the current \f$3\times 3\f$ matrix.
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV T
    determinant3x3() const;

    /**
     * Print all the values in the matrix using the specified C or Fortran
     * index style.
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV void
    print(const bool C_index_style = false, const bool scientific = true) const;

    /**
     * Set all matrix entries to zero.
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV void
    reinit();

  private:
    pointer   values;
    size_type rows;
    size_type cols;
  };


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV
  FullMatrixWrapper<T>::FullMatrixWrapper()
    : values(nullptr)
    , rows(0)
    , cols(0)
  {}


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV
  FullMatrixWrapper<T>::FullMatrixWrapper(pointer         p,
                                          const size_type _rows,
                                          const size_type _cols)
    : values(p)
    , rows(_rows)
    , cols(_cols)
  {}


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV void
  FullMatrixWrapper<T>::reinit(pointer         p,
                               const size_type _rows,
                               const size_type _cols)
  {
    values = p;
    rows   = _rows;
    cols   = _cols;
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV
  FullMatrixWrapper<T>::FullMatrixWrapper(FullMatrixWrapper<T> &&mat)
    : values(mat.values)
    , rows(mat.rows)
    , cols(mat.cols)
  {}


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV FullMatrixWrapper<T> &
  FullMatrixWrapper<T>::operator=(const FullMatrixWrapper<T> &mat)
  {
    assert(rows == mat.rows);
    assert(cols == mat.cols);

    for (size_type i = 0; i < n_elements(); i++)
      values[i] = mat.values[i];

    return *this;
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV FullMatrixWrapper<T> &
  FullMatrixWrapper<T>::operator=(FullMatrixWrapper<T> &&mat)
  {
    values = mat.values;
    rows   = mat.rows;
    cols   = mat.cols;

    return *this;
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV void
  FullMatrixWrapper<T>::fill(const_pointer data)
  {
    for (size_type i = 0; i < n_elements(); i++)
      values[i] = data[i];
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV inline typename FullMatrixWrapper<T>::size_type
                 FullMatrixWrapper<T>::m() const
  {
    return rows;
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV inline typename FullMatrixWrapper<T>::size_type
                 FullMatrixWrapper<T>::n() const
  {
    return cols;
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV inline typename FullMatrixWrapper<T>::size_type
                 FullMatrixWrapper<T>::n_elements() const
  {
    return rows * cols;
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV typename FullMatrixWrapper<T>::size_type
                 FullMatrixWrapper<T>::size(const size_type i) const
  {
    switch (i)
      {
        case 0:
          return rows;
        case 1:
          return cols;
        default:
          assert(false);
          return 0;
      }
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV inline typename FullMatrixWrapper<T>::pointer
                 FullMatrixWrapper<T>::data()
  {
    return values;
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV inline
    typename FullMatrixWrapper<T>::const_pointer
    FullMatrixWrapper<T>::data() const
  {
    return values;
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV inline typename FullMatrixWrapper<T>::reference
  FullMatrixWrapper<T>::operator()(const size_type i,
                                   const size_type j,
                                   const bool      is_column_major)
  {
    return is_column_major ? values[rows * j + i] : values[cols * i + j];
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV inline
    typename FullMatrixWrapper<T>::const_reference
    FullMatrixWrapper<T>::operator()(const size_type i,
                                     const size_type j,
                                     const bool      is_column_major) const
  {
    return is_column_major ? values[rows * j + i] : values[cols * i + j];
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV void
  FullMatrixWrapper<T>::vmult(VectorWrapper<T>       &w,
                              const VectorWrapper<T> &v,
                              const bool              adding) const
  {
    assert(cols == v.size());
    assert(rows == w.size());

    for (size_type i = 0; i < rows; i++)
      {
        if (!adding)
          w(i) = 0.;

        for (size_type j = 0; j < cols; j++)
          w(i) += (*this)(i, j) * v(j);
      }
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV void
  FullMatrixWrapper<T>::Tvmult(VectorWrapper<T>       &w,
                               const VectorWrapper<T> &v,
                               const bool              adding) const
  {
    assert(rows == v.size());
    assert(cols == w.size());

    for (size_type i = 0; i < cols; i++)
      {
        if (!adding)
          w(i) = 0.;

        for (size_type j = 0; j < rows; j++)
          w(i) += (*this)(j, i) * v(j);
      }
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV void
  FullMatrixWrapper<T>::mmult(FullMatrixWrapper<T>       &C,
                              const FullMatrixWrapper<T> &B,
                              const bool                  adding) const
  {
    assert(cols == B.m());
    assert(rows == C.m());
    assert(B.n() == C.n());

    for (size_type i = 0; i < C.m(); i++)
      {
        for (size_type j = 0; j < C.n(); j++)
          {
            if (!adding)
              C(i, j) = 0.;

            for (size_type k = 0; k < cols; k++)
              C(i, j) += (*this)(i, k) * B(k, j);
          }
      }
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV void
  FullMatrixWrapper<T>::Tmmult(FullMatrixWrapper<T>       &C,
                               const FullMatrixWrapper<T> &B,
                               const bool                  adding) const
  {
    assert(rows == B.m());
    assert(cols == C.m());
    assert(B.n() == C.n());

    for (size_type i = 0; i < C.m(); i++)
      {
        for (size_type j = 0; j < C.n(); j++)
          {
            if (!adding)
              C(i, j) = 0.;

            for (size_type k = 0; k < rows; k++)
              C(i, j) += (*this)(k, i) * B(k, j);
          }
      }
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV void
  FullMatrixWrapper<T>::mTmult(FullMatrixWrapper<T>       &C,
                               const FullMatrixWrapper<T> &B,
                               const bool                  adding) const
  {
    assert(cols == B.n());
    assert(rows == C.m());
    assert(B.m() == C.n());

    for (size_type i = 0; i < C.m(); i++)
      {
        for (size_type j = 0; j < C.n(); j++)
          {
            if (!adding)
              C(i, j) = 0.;

            for (size_type k = 0; k < cols; k++)
              C(i, j) += (*this)(i, k) * B(j, k);
          }
      }
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV void
  FullMatrixWrapper<T>::invert_by_gauss_elim(FullMatrixWrapper<T> &M_inv)
  {
    assert(rows == cols);
    assert(rows == M_inv.m());
    assert(cols == M_inv.n());

    M_inv.reinit();

    // Eliminate the lower triangular part of the matrix.
    for (size_type l = 0; l < rows; l++)
      {
        // Scale the current row by the factor \f$\frac{1}{a_{ll}}\f$ and
        // this value is directly filled into the result matrix.
        M_inv(l, l) = 1.0 / (*this)(l, l);

        // For the matrix \f$M\f$, after this scaling, the element
        // \f$M_{ll}\f$ will become 1, there is no need to compute the
        // scaling of this element. Meanwhile, the elements \f$M_{l1},
        // \cdots, M_{l,l-1}\f$ are already zeros after previous
        // eliminations, neither need we compute the scaling of these
        // elements. Therefore, the actual computation to be performed is
        // for columns \f$j = l + 1, \cdots, n\f$.
        for (size_type j = l + 1; j < rows; j++)
          (*this)(l, j) = M_inv(l, l) * (*this)(l, j);

        // For the matrix \f$M^{-1}\f$, after this scaling, the element at
        // \f$(l, l)\f$ will be \f$\frac{1}{M_{ll}}\f$, so there is no need
        // to compute this scaling. Meanwhile, only those elements
        // \f$M^{-1}_{l,1}, \cdots, M^{-1}_{l-1}\f$ may be non-zeros, we
        // only loop over \f$j = 1, \cdots, l
        // - 1\f$.
        for (size_type j = 0; j < l; j++)
          M_inv(l, j) = M_inv(l, l) * M_inv(l, j);

        // Then we eliminate the elements \f$M_{l+1,l}, \cdots, M_{n,l}\f$
        // by iterating over the rows \f$l + 1, \cdots, n\f$.
        for (size_type i = l + 1; i < rows; i++)
          {
            // This transformation will only influence the columns \f$1,
            // \cdots, l\f$ in \f$M^{-1}\f$.
            for (size_type j = 0; j <= l; j++)
              M_inv(i, j) = M_inv(i, j) - (*this)(i, l) * M_inv(l, j);

            // This transformation will only influence the columns \f$l+1,
            // \cdots, n\f$ in \f$M\f$.
            for (size_type j = l + 1; j < rows; j++)
              (*this)(i, j) = (*this)(i, j) - (*this)(i, l) * (*this)(l, j);
          }
      }

    // Eliminate the upper triangular part. Now the row transformation is
    // only related to the result matrix \f$M^{-1}\f$.
    for (size_type l = rows - 1; l > 0; l--)
      {
        // Eliminate the elements \f$M_{l-1,l}, \cdots, M_{1,l}\f$.
        size_type i = l - 1;

        // N.B. When the loop counter \p i of \p unsigned type decreased to be
        // zero, further decrement will not produce a value smaller than
        // zero but a very large integer. Hence, we do not use a typical
        // for-loop here as below.
        //
        // <code> for (size_type i = l - 1; i >= 0; i--)
        // {
        //   ...
        // }
        // </code>
        //
        // Instead, we use a while-loop with the \p true condition. Inside this
        // loop, when we detect the counter is zero after loop execution, we
        // jump out the loop.
        while (true)
          {
            // Because the elements in the current l'th row of \f$M^{-1}\f$
            // are generally non-zeros, the computation involves columns
            // \f$1, \cdots, n\f$.
            for (size_type j = 0; j < rows; j++)
              M_inv(i, j) = M_inv(i, j) - (*this)(i, l) * M_inv(l, j);

            if (i == 0)
              break;
            else
              i--;
          }
      }
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV void
  FullMatrixWrapper<T>::invert2x2(const FullMatrixWrapper<T> &M)
  {
    assert(rows == cols);
    assert(rows == M.m());
    assert(cols == M.n());

    (*this)(0, 0) = M(1, 1);
    (*this)(0, 1) = -M(0, 1);
    (*this)(1, 0) = -M(1, 0);
    (*this)(1, 1) = M(0, 0);

    (*this) /= M.determinant2x2();
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV FullMatrixWrapper<T> &
  FullMatrixWrapper<T>::operator*=(const T factor)
  {
    for (size_type i = 0; i < n_elements(); i++)
      values[i] *= factor;

    return *this;
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV FullMatrixWrapper<T> &
  FullMatrixWrapper<T>::operator/=(const T factor)
  {
    for (size_type i = 0; i < n_elements(); i++)
      values[i] /= factor;

    return *this;
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV T
  FullMatrixWrapper<T>::determinant2x2() const
  {
    return (*this)(0, 0) * (*this)(1, 1) - (*this)(0, 1) * (*this)(1, 0);
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV T
  FullMatrixWrapper<T>::determinant3x3() const
  {
    return (*this)(0, 0) * (*this)(1, 1) * (*this)(2, 2) +
           (*this)(0, 1) * (*this)(1, 2) * (*this)(2, 0) +
           (*this)(0, 2) * (*this)(1, 0) * (*this)(2, 1) -
           (*this)(0, 2) * (*this)(1, 1) * (*this)(2, 0) -
           (*this)(0, 1) * (*this)(1, 0) * (*this)(2, 2) -
           (*this)(0, 0) * (*this)(1, 2) * (*this)(2, 1);
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV void
  FullMatrixWrapper<T>::print(const bool C_index_style,
                              const bool scientific) const
  {
    for (size_type i = 0; i < rows; i++)
      {
        for (size_type j = 0; j < cols; j++)
          {
            if (scientific)
              printf("%20.8e",
                     values[C_index_style ? i * cols + j : j * rows + i]);
            else
              printf("%20.8f",
                     values[C_index_style ? i * cols + j : j * rows + i]);
          }

        printf("\n");
      }
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV void
  FullMatrixWrapper<T>::reinit()
  {
    for (size_type i = 0; i < n_elements(); i++)
      values[i] = 0.;
  }
} // namespace PlatformShared

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_LINEAR_ALGEBRA_FULL_MATRIX_WRAPPER_H_
