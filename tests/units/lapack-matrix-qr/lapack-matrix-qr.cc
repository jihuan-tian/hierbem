/**
 * \file lapack-matrix-qr.cc
 * \brief Verify QR decomposition and reduced QR decomposition.
 *
 * \ingroup linalg
 * \author Jihuan Tian
 * \date 2021-07-02
 */

#include <deal.II/base/numbers.h>

#include <catch2/catch_all.hpp>
#include <julia.h>

#include <complex>
#include <iostream>
#include <type_traits>
#include <vector>

#include "hbem_julia_cpp_compare.h"
#include "hbem_julia_wrapper.h"
#include "linear_algebra/lapack_full_matrix_ext.h"
#include "utilities/debug_tools.h"

using namespace Catch::Matchers;
using namespace dealii;
using namespace HierBEM;

TEST_CASE("Verify matrix QR decomposition for LAPACKFullMatrixExt", "[linalg]")
{
  INFO("*** test start");
  HBEMJuliaWrapper &inst = HBEMJuliaWrapper::get_instance();
  inst.source_file("process.jl");

  using namespace std::complex_literals;

  std::vector<double> values{
    3., 8., 10., 7., 1., 9., 7., 6., 12., 4., 5., 8., 8., 9., 20.};
  std::vector<std::complex<double>> values_c{3. + 2.i,
                                             8. + 1.i,
                                             10. + 3.i,
                                             7. + 5.i,
                                             1. + 0.3i,
                                             9. + 7.i,
                                             7. + 10.i,
                                             6. + 1.i,
                                             12. + 0.8i,
                                             4. + 2.i,
                                             5. + 7.i,
                                             8. + 10.i,
                                             8. + 3.3i,
                                             9. + 7.i,
                                             20. + 12.i};

  {
    /**
     * QR decomposition of a matrix with more columns than rows.
     */
    LAPACKFullMatrixExt<double>               M, Q, R;
    LAPACKFullMatrixExt<std::complex<double>> Mc, Qc, Rc;

    LAPACKFullMatrixExt<double>::Reshape(3, 5, values, M);
    M.qr(Q, R);
    LAPACKFullMatrixExt<std::complex<double>>::Reshape(3, 5, values_c, Mc);
    Mc.qr(Qc, Rc);

    Q.print_formatted(std::cout, 5, true, 25, "0");
    std::cout << "\n";
    R.print_formatted(std::cout, 5, true, 25, "0");
    std::cout << "\n";
    Qc.print_formatted(std::cout, 5, true, 30, "0");
    std::cout << "\n";
    Rc.print_formatted(std::cout, 5, true, 30, "0");
    std::cout << std::endl;

    compare_with_jl_matrix(Q, "Q1", 1e-14, 1e-14);
    compare_with_jl_matrix(R, "R1", 1e-14, 1e-14);
    compare_with_jl_matrix(Qc, "Qc1", 1e-14, 1e-14);
    compare_with_jl_matrix(Rc, "Rc1", 1e-14, 1e-14);
  }

  {
    /**
     * QR decomposition of a matrix with more rows than columns.
     */
    LAPACKFullMatrixExt<double>               M, Q, R;
    LAPACKFullMatrixExt<std::complex<double>> Mc, Qc, Rc;

    LAPACKFullMatrixExt<double>::Reshape(5, 3, values, M);
    M.qr(Q, R);
    LAPACKFullMatrixExt<std::complex<double>>::Reshape(5, 3, values_c, Mc);
    Mc.qr(Qc, Rc);

    Q.print_formatted(std::cout, 5, true, 25, "0");
    std::cout << "\n";
    R.print_formatted(std::cout, 5, true, 25, "0");
    std::cout << "\n";
    Qc.print_formatted(std::cout, 5, true, 30, "0");
    std::cout << "\n";
    Rc.print_formatted(std::cout, 5, true, 30, "0");
    std::cout << std::endl;

    compare_with_jl_matrix(Q, "Q2", 1e-14, 1e-14);
    compare_with_jl_matrix(R, "R2", 1e-14, 1e-14);
    compare_with_jl_matrix(Qc, "Qc2", 1e-14, 1e-14);
    compare_with_jl_matrix(Rc, "Rc2", 1e-14, 1e-14);
  }

  {
    /**
     * Reduced QR decomposition of a matrix with more rows than columns.
     */
    LAPACKFullMatrixExt<double>               M, Q, R;
    LAPACKFullMatrixExt<std::complex<double>> Mc, Qc, Rc;

    LAPACKFullMatrixExt<double>::Reshape(5, 3, values, M);
    M.reduced_qr(Q, R);
    LAPACKFullMatrixExt<std::complex<double>>::Reshape(5, 3, values_c, Mc);
    Mc.reduced_qr(Qc, Rc);

    Q.print_formatted(std::cout, 5, true, 25, "0");
    std::cout << "\n";
    R.print_formatted(std::cout, 5, true, 25, "0");
    std::cout << "\n";
    Qc.print_formatted(std::cout, 5, true, 30, "0");
    std::cout << "\n";
    Rc.print_formatted(std::cout, 5, true, 30, "0");
    std::cout << std::endl;

    compare_with_jl_matrix(Q, "Q3", 1e-14, 1e-14);
    compare_with_jl_matrix(R, "R3", 1e-14, 1e-14);
    compare_with_jl_matrix(Qc, "Qc3", 1e-14, 1e-14);
    compare_with_jl_matrix(Rc, "Rc3", 1e-14, 1e-14);
  }

  INFO("*** test end");
}
