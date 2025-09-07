/**
 * @file lapack-matrix-norm.cc
 * @brief Verify norm computation for LAPACKFullMatrixExt
 * @ingroup linalg
 *
 * @date 2025-03-28
 * @author Jihuan Tian
 */

#include <deal.II/base/exceptions.h>

#include <deal.II/lac/lapack_support.h>

#include <catch2/catch_all.hpp>

#include <cmath>
#include <complex>

#include "hbem_julia_cpp_compare.h"
#include "linear_algebra/lapack_full_matrix_ext.h"

using namespace Catch::Matchers;
using namespace dealii;
using namespace HierBEM;

TEST_CASE("Verify norm computation for LAPACKFullMatrixExt", "[linalg]")
{
  INFO("*** test start");
  HBEMJuliaWrapper &inst = HBEMJuliaWrapper::get_instance();
  inst.source_file("process.jl");

  const unsigned int n = 6;

  LAPACKFullMatrixExt<double> A(n, n);
  LAPACKFullMatrixExt<double> A_symm(n, n);

  unsigned int counter = 1;
  for (unsigned int i = 0; i < n; i++)
    {
      for (unsigned int j = 0; j < n; j++)
        {
          A(i, j) = (double)counter;
          counter++;
        }
    }

  A.mTmult(A_symm, A);
  Assert(A_symm.get_property() == LAPACKSupport::Property::symmetric,
         ExcInternalError());

  LAPACKFullMatrixExt<std::complex<double>> A_complex(n, n);
  LAPACKFullMatrixExt<std::complex<double>> A_complex_symm(n, n);

  for (unsigned int i = 0; i < n; i++)
    {
      for (unsigned int j = 0; j < n; j++)
        {
          A_complex(i, j) = std::complex<double>(std::sin((double)(i + 1)),
                                                 std::cos((double)(j + 1)));
        }
    }

  A_complex.mTmult(A_complex_symm, A_complex);
  Assert(A_complex_symm.get_property() == LAPACKSupport::Property::symmetric,
         ExcInternalError());

  // @p opnorm is used for computing matrix norm, while @p norm is used for
  // vector norm and the Frobenius norm of a matrix.
  compare_with_jl_scalar(A.l1_norm(), "opnorm(A, 1)", 1e-15, 1e-15);
  compare_with_jl_scalar(A.linfty_norm(), "opnorm(A, Inf)", 1e-15, 1e-15);
  compare_with_jl_scalar(A.frobenius_norm(), "norm(A, 2)", 1e-15, 1e-15);

  compare_with_jl_scalar(A_symm.l1_norm(), "opnorm(A_symm, 1)", 1e-15, 1e-15);
  compare_with_jl_scalar(A_symm.linfty_norm(),
                         "opnorm(A_symm, Inf)",
                         1e-15,
                         1e-15);
  compare_with_jl_scalar(A_symm.frobenius_norm(),
                         "norm(A_symm, 2)",
                         1e-15,
                         1e-15);

  compare_with_jl_scalar(A_complex.l1_norm(),
                         "opnorm(A_complex, 1)",
                         1e-15,
                         1e-15);
  compare_with_jl_scalar(A_complex.linfty_norm(),
                         "opnorm(A_complex, Inf)",
                         1e-15,
                         1e-15);
  compare_with_jl_scalar(A_complex.frobenius_norm(),
                         "norm(A_complex, 2)",
                         1e-15,
                         1e-15);

  compare_with_jl_scalar(A_complex_symm.l1_norm(),
                         "opnorm(A_complex_symm, 1)",
                         1e-15,
                         1e-15);
  compare_with_jl_scalar(A_complex_symm.linfty_norm(),
                         "opnorm(A_complex_symm, Inf)",
                         1e-15,
                         1e-15);
  compare_with_jl_scalar(A_complex_symm.frobenius_norm(),
                         "norm(A_complex_symm, 2)",
                         1e-15,
                         1e-15);

  INFO("*** test end");
}
