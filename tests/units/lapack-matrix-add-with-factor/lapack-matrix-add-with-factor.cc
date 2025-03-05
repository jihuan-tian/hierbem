/**
 * \file lapack-matrix-add-with-factor.cc
 * \brief Verify matrix addition \f$C = A + b B\f$
 *
 * \ingroup linalg
 * \author Jihuan Tian
 * \date 2021-10-05
 */

#include <catch2/catch_all.hpp>
#include <julia.h>

#include <cmath>
#include <complex>

#include "lapack_full_matrix_ext.h"

using namespace Catch::Matchers;
using namespace dealii;
using namespace HierBEM;

void
matrix_double_add()
{
  std::vector<double> A_data{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> B_data{3, 5, 7, 4, 6, 8, 5, 7, 9};

  const size_t                m = 3;
  const size_t                n = 3;
  LAPACKFullMatrixExt<double> A, B;
  LAPACKFullMatrixExt<double>::Reshape(m, n, A_data, A);
  LAPACKFullMatrixExt<double>::Reshape(m, n, B_data, B);

  LAPACKFullMatrixExt<double> C;
  double                      b = 3.5;
  A.add(C, b, B);
  A.add(b, B);

  // Get the result from Julia.
  jl_array_t *C_jl      = (jl_array_t *)jl_eval_string("C");
  double     *C_jl_data = (double *)jl_array_data(C_jl);
  REQUIRE(jl_array_dim(C_jl, 0) == m);
  REQUIRE(jl_array_dim(C_jl, 1) == n);

  for (size_t j = 0; j < n; j++)
    for (size_t i = 0; i < m; i++)
      {
        REQUIRE(C_jl_data[i + j * m] == Catch::Approx(C(i, j)).epsilon(1e-15));
        REQUIRE(C_jl_data[i + j * m] == Catch::Approx(A(i, j)).epsilon(1e-15));
      }
}

void
matrix_complex_add()
{
  using namespace std::complex_literals;

  std::vector<std::complex<double>> A_data{1.0 + 0.1i,
                                           2.0 + 0.2i,
                                           3.0 + 0.3i,
                                           4.0 + 0.4i,
                                           5.0 + 0.5i,
                                           6.0 + 0.6i,
                                           7.0 + 0.7i,
                                           8.0 + 0.8i,
                                           9.0 + 0.9i};
  std::vector<std::complex<double>> B_data{3.0 + 0.1i,
                                           5.0 + 0.2i,
                                           7.0 + 0.3i,
                                           4.0 + 0.4i,
                                           6.0 + 0.5i,
                                           8.0 + 0.6i,
                                           5.0 + 0.7i,
                                           7.0 + 0.8i,
                                           9.0 + 0.9i};

  const size_t                              m = 3;
  const size_t                              n = 3;
  LAPACKFullMatrixExt<std::complex<double>> A, B;
  LAPACKFullMatrixExt<std::complex<double>>::Reshape(m, n, A_data, A);
  LAPACKFullMatrixExt<std::complex<double>>::Reshape(m, n, B_data, B);

  LAPACKFullMatrixExt<std::complex<double>> C;
  double                                    b = 3.5;
  A.add(C, b, B);
  A.add(b, B);

  // Get the result from Julia.
  jl_array_t           *C_jl      = (jl_array_t *)jl_eval_string("C_complex");
  std::complex<double> *C_jl_data = (std::complex<double> *)jl_array_data(C_jl);
  REQUIRE(jl_array_dim(C_jl, 0) == m);
  REQUIRE(jl_array_dim(C_jl, 1) == n);

  for (size_t j = 0; j < n; j++)
    for (size_t i = 0; i < m; i++)
      {
        REQUIRE(C_jl_data[i + j * m].real() ==
                Catch::Approx(C(i, j).real()).epsilon(1e-15));
        REQUIRE(C_jl_data[i + j * m].imag() ==
                Catch::Approx(C(i, j).imag()).epsilon(1e-15));
        REQUIRE(C_jl_data[i + j * m].real() ==
                Catch::Approx(A(i, j).real()).epsilon(1e-15));
        REQUIRE(C_jl_data[i + j * m].imag() ==
                Catch::Approx(A(i, j).imag()).epsilon(1e-15));
      }
}

TEST_CASE("Verify matrix addition for LAPACKFullMatrixExt", "[linalg]")
{
  INFO("*** test start");
  jl_init();
  (void)jl_eval_string("include(\"process.jl\")");

  matrix_double_add();
  matrix_complex_add();

  jl_atexit_hook(0);
  INFO("*** test end");
}
