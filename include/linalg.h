/*
 * linalg.h
 *
 *  Created on: 2020年11月30日
 *      Author: jihuan
 */

#ifndef INCLUDE_LINALG_H_
#define INCLUDE_LINALG_H_

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

namespace LinAlg
{
  using namespace dealii;

  /**
   * Calculate the determinant of a $4 \times 4$ matrix.
   */
  template <typename number>
  number
  determinant4x4(const FullMatrix<number> &matrix)
  {
    Assert(matrix.m() == 4, ExcDimensionMismatch(matrix.m(), 4));
    Assert(matrix.n() == 4, ExcDimensionMismatch(matrix.n(), 4));

    return matrix(0, 0) *
             (matrix(1, 1) *
                (matrix(2, 2) * matrix(3, 3) - matrix(2, 3) * matrix(3, 2)) -
              matrix(1, 2) *
                (matrix(2, 1) * matrix(3, 3) - matrix(2, 3) * matrix(3, 1)) +
              matrix(1, 3) *
                (matrix(2, 1) * matrix(3, 2) - matrix(2, 2) * matrix(3, 1))) -
           matrix(0, 1) *
             (matrix(1, 0) *
                (matrix(2, 2) * matrix(3, 3) - matrix(2, 3) * matrix(3, 2)) -
              matrix(1, 2) *
                (matrix(2, 0) * matrix(3, 3) - matrix(2, 3) * matrix(3, 0)) +
              matrix(1, 3) *
                (matrix(2, 0) * matrix(3, 2) - matrix(2, 2) * matrix(3, 0))) +
           matrix(0, 2) *
             (matrix(1, 0) *
                (matrix(2, 1) * matrix(3, 3) - matrix(2, 3) * matrix(3, 1)) -
              matrix(1, 1) *
                (matrix(2, 0) * matrix(3, 3) - matrix(2, 3) * matrix(3, 0)) +
              matrix(1, 3) *
                (matrix(2, 0) * matrix(3, 1) - matrix(2, 1) * matrix(3, 0))) -
           matrix(0, 3) *
             (matrix(1, 0) *
                (matrix(2, 1) * matrix(3, 2) - matrix(2, 2) * matrix(3, 1)) -
              matrix(1, 1) *
                (matrix(2, 0) * matrix(3, 2) - matrix(2, 2) * matrix(3, 0)) +
              matrix(1, 2) *
                (matrix(2, 0) * matrix(3, 1) - matrix(2, 1) * matrix(3, 0)));
  }
} // namespace LinAlg

#endif /* INCLUDE_LINALG_H_ */
