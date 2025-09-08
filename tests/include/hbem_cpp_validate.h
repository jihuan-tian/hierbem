// Copyright (C) 2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file hbem_cpp_validate.h
 * @brief Validate C++ computation.
 *
 * @date 2025-03-08
 * @author Jihuan Tian
 */

#ifndef HIERBEM_TESTS_INCLUDE_HBEM_CPP_VALIDATE_H_
#define HIERBEM_TESTS_INCLUDE_HBEM_CPP_VALIDATE_H_

#include <deal.II/base/numbers.h>

#include <catch2/catch_all.hpp>

#include <functional>
#include <string>

#include "config.h"
#include "linear_algebra/lapack_full_matrix_ext.h"
#include "utilities/debug_tools.h"

HBEM_NS_OPEN

using namespace Catch::Matchers;
using namespace dealii;

template <typename Func>
void
compare_two_files(const std::string &file1,
                  const std::string &file2,
                  const Func        &check_equality)
{
  std::vector<std::string> file1_lines, file2_lines;
  read_file_lines(file1, file1_lines);
  read_file_lines(file2, file2_lines);

  check_equality(file1_lines.size(), file2_lines.size());

  for (size_t i = 0; i < file1_lines.size(); i++)
    check_equality(file1_lines[i], file2_lines[i]);
}


template <typename Number>
void
compare_lapack_matrices(
  const LAPACKFullMatrixExt<Number>                      &mat_ref,
  const LAPACKFullMatrixExt<Number>                      &mat,
  const typename numbers::NumberTraits<Number>::real_type abs_error = 1e-15,
  const typename numbers::NumberTraits<Number>::real_type rel_error = 1e-15)
{
  const size_t m = mat.m();
  const size_t n = mat.n();
  REQUIRE(mat_ref.m() == m);
  REQUIRE(mat_ref.n() == n);

  for (size_t j = 0; j < n; j++)
    for (size_t i = 0; i < m; i++)
      {
        if constexpr (numbers::NumberTraits<Number>::is_complex)
          {
            REQUIRE_THAT(mat(i, j).real(),
                         WithinAbs(mat_ref(i, j).real(), abs_error) ||
                           WithinRel(mat_ref(i, j).real(), rel_error));
            REQUIRE_THAT(mat(i, j).imag(),
                         WithinAbs(mat_ref(i, j).imag(), abs_error) ||
                           WithinRel(mat_ref(i, j).imag(), rel_error));
          }
        else
          {
            REQUIRE_THAT(mat(i, j),
                         WithinAbs(mat_ref(i, j), abs_error) ||
                           WithinRel(mat_ref(i, j), rel_error));
          }
      }
}


template <typename Number>
void
check_svd_self_consistency(
  const LAPACKFullMatrixExt<Number>                                    &A,
  const LAPACKFullMatrixExt<Number>                                    &U,
  const LAPACKFullMatrixExt<Number>                                    &VT,
  const std::vector<typename numbers::NumberTraits<Number>::real_type> &Sigma_r,
  const typename numbers::NumberTraits<Number>::real_type abs_error = 1e-15,
  const typename numbers::NumberTraits<Number>::real_type rel_error = 1e-15)
{
  // Check <code>A == U*Sigma_r*VT</code>. Assume there are @p k singular values in
  // the vector @p Sigma_r, before the computation, we need to keep only the
  // first @p k columns in @p U and the first @p k rows in @p VT.
  const size_t                n = Sigma_r.size();
  LAPACKFullMatrixExt<Number> U_copy(U), VT_copy(VT);
  U_copy.keep_first_n_columns(n);
  VT_copy.keep_first_n_rows(n);
  // The multiplication of @p Sigma_r*VT is computed by scaling the rows of @p VT
  // using @p Sigma_r .
  VT_copy.scale_rows(Sigma_r);

  LAPACKFullMatrixExt<Number> A_tmp;
  U_copy.mmult(A_tmp, VT_copy);

  compare_lapack_matrices(A, A_tmp, abs_error, rel_error);

  // Check <code>U*U^T == I</code> or <code>U*U^H == I</code>.
  LAPACKFullMatrixExt<Number> Iu;
  LAPACKFullMatrixExt<Number>::IdentityMatrix(U.m(), Iu);
  LAPACKFullMatrixExt<Number> Iu_tmp;

  if constexpr (numbers::NumberTraits<Number>::is_complex)
    U.mHmult(Iu_tmp, U);
  else
    U.mTmult(Iu_tmp, U);
  compare_lapack_matrices(Iu, Iu_tmp, abs_error, rel_error);

  // Check <code>V*V^T == I</code> or <code>V*V^H == I</code>.
  LAPACKFullMatrixExt<Number> Iv;
  LAPACKFullMatrixExt<Number>::IdentityMatrix(VT.n(), Iv);
  LAPACKFullMatrixExt<Number> Iv_tmp;
  if constexpr (numbers::NumberTraits<Number>::is_complex)
    VT.Hmmult(Iv_tmp, VT);
  else
    VT.Tmmult(Iv_tmp, VT);
  compare_lapack_matrices(Iv, Iv_tmp, abs_error, rel_error);
}

HBEM_NS_CLOSE

#endif
