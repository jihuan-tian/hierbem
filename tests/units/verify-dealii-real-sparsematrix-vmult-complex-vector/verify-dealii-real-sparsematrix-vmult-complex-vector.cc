/**
 * @file verify-dealii-real-sparsematrix-vmult-complex-vector.cc
 * @brief Verify the multiplication of a real valued dealii::SparseMatrix with a
 * complex valued dealii::Vector.
 *
 * This function is used in the operator preconditioner, where the mass matrix
 * triple \f$C_d M_r C_p^{\mathrm{T}}\f$ or its transpose (real valued)
 * multiplies a complex valued vector.
 *
 * @ingroup linalg
 *
 * @date 2025-07-02
 * @author Jihuan Tian
 */

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_matrix.templates.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <complex>
#include <iostream>

#include "hbem_julia_cpp_compare.h"

using namespace Catch::Matchers;
using namespace dealii;
using namespace HierBEM;

TEST_CASE("Verify real valued SparseMatrix multiplies a complex valued vector",
          "[linalg]")
{
  INFO("*** test start");
  HBEMJuliaWrapper &inst = HBEMJuliaWrapper::get_instance();
  inst.source_file("process.jl");

  int    *row_indices = inst.get_int_array_var("rows");
  int    *col_indices = inst.get_int_array_var("cols");
  double *mat_vals    = inst.get_double_array_var("vals");

  HBEMJuliaValue     jl_value         = inst.eval_string("rows");
  const unsigned int number_of_values = jl_value.nrows();
  const unsigned int m                = 5;
  const unsigned int n                = 5;

  // Create the sparsity pattern.
  DynamicSparsityPattern dsp(m, n);
  for (unsigned int i = 0; i < number_of_values; i++)
    {
      dsp.add(row_indices[i] - 1, col_indices[i] - 1);
    }

  // Create the sparse matrix from the sparsity pattern.
  SparsityPattern sp;
  sp.copy_from(dsp);
  SparseMatrix<double> mat(sp);

  // Assign values to the sparse matrix.
  for (unsigned int i = 0; i < number_of_values; i++)
    mat.set(row_indices[i] - 1, col_indices[i] - 1, mat_vals[i]);

  // Create input and output vectors.
  std::complex<double>        *x_jl = inst.get_complex_double_array_var("x");
  Vector<std::complex<double>> x(n);
  for (unsigned int i = 0; i < n; i++)
    x(i) = x_jl[i];
  Vector<std::complex<double>> y_cpp(m);

  // Performance matrix/vector multiplication.
  mat.vmult(y_cpp, x);

  std::cout << "A*x=\n";
  for (unsigned int i = 0; i < m; i++)
    std::cout << y_cpp(i) << "\n";
  std::cout << std::endl;

  // Compare C++ and Julia results.
  compare_with_jl_array(y_cpp, "y");

  INFO("*** test end");
}
