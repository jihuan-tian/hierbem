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

#include <limits>

#include "lapack_full_matrix_ext.h"

namespace IdeoBEM
{
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


    template <typename number>
    number
    determinant4x4(const LAPACKFullMatrixExt<number> &matrix)
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


    /**
     * Check if the vector is zero-valued by calculating its L1 norm.
     *
     * @param vec
     * @return
     */
    template <typename number>
    bool
    is_all_zero(const Vector<number> &vec)
    {
      if (vec.l1_norm() > std::numeric_limits<number>::epsilon())
        {
          return false;
        }
      else
        {
          return true;
        }
    }


    /**
     * Copy a segment of a vector into another vector at the specified location.
     *
     * @param dst_vec
     * @param dst_start_index
     * @param src_vec
     * @param start_index
     * @param number_of_data
     */
    template <typename number>
    void
    copy_vector(Vector<number>                          &dst_vec,
                const typename Vector<number>::size_type dst_start_index,
                const Vector<number>                    &src_vec,
                const typename Vector<number>::size_type src_start_index,
                const typename Vector<number>::size_type number_of_data)
    {
      std::memcpy(dst_vec.data() + dst_start_index,
                  src_vec.data() + src_start_index,
                  number_of_data * sizeof(number));
    }
  } // namespace LinAlg
} // namespace IdeoBEM

#endif /* INCLUDE_LINALG_H_ */
