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
 * @file solver-dqgmres.cc
 * @brief Verify DQGMRES
 *
 * @ingroup linalg
 * @author Jihuan Tian
 * @date 2025-07-25
 */
#include <deal.II/base/types.h>

#include <deal.II/lac/lapack_support.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>

#include <catch2/catch_all.hpp>

#include <complex>
#include <type_traits>

#include "hbem_julia_cpp_compare.h"
#include "hbem_julia_wrapper.h"
#include "linear_algebra/lapack_full_matrix_ext.h"
#include "solvers/solver_gmres_general.h"

using namespace Catch::Matchers;
using namespace dealii;
using namespace HierBEM;

TEST_CASE("Verify DQGMRES", "[linalg]")
{
  // Write run-time logs to file
  std::ofstream ofs("solver-dqgmres.log");
  deallog.pop();
  deallog.depth_console(0);
  deallog.depth_file(5);
  deallog.attach(ofs);

  using size_type = std::make_unsigned<dealii::types::blas_int>::type;

  INFO("*** test start");
  HBEMJuliaWrapper &inst = HBEMJuliaWrapper::get_instance();
  inst.source_file(SOURCE_DIR "/process.jl");

  {
    // Verify the real valued case.

    // Get matrix and vector data from Julia.
    HBEMJuliaValue jl_value = inst.eval_string("b1");
    const size_t   n        = jl_value.length();
    double        *A_data   = inst.get_double_array_var("A1");
    double        *b_data   = inst.get_double_array_var("b1");

    LAPACKFullMatrixExt<double> A;
    LAPACKFullMatrixExt<double>::Reshape(n, n, A_data, A);

    Vector<double> b(n);
    for (size_type i = 0; i < n; i++)
      b(i) = b_data[i];

    // Configure the solver.
    const size_type krylov_dim = 1000;
    // We use full orthogonalization.
    const size_type ortho_hist_len = krylov_dim;
    const size_type max_iter       = 1;
    const double    abs_tol        = 1e-3;

    {
      deallog << "*** Real valued, without preconditioner" << std::endl;
      Vector<double>                x(n);
      SolverControl                 control(max_iter, abs_tol, true, true);
      SolverDQGMRES<Vector<double>> solver(control);
      solver.set_krylov_dim(krylov_dim);
      solver.set_orthogonal_history_length(ortho_hist_len);
      solver.solve(A, x, b, PreconditionIdentity());
      compare_with_jl_array(x, "x1", 1e-3, 1e-3);
    }

    {
      deallog << "*** Real valued, without preconditioner, using deal.ii solver"
              << std::endl;
      Vector<double> x(n);
      SolverControl  control(max_iter * krylov_dim, abs_tol, true, true);
      SolverGMRES<Vector<double>> solver(
        control,
        typename SolverGMRES<Vector<double>>::AdditionalData(krylov_dim + 2));
      solver.solve(A, x, b, PreconditionIdentity());
      compare_with_jl_array(x, "x1", 1e-3, 1e-3);
    }

    // Initialize the preconditioner.
    PreconditionJacobi<LAPACKFullMatrixExt<double>> precond;
    precond.initialize(
      A,
      typename PreconditionJacobi<LAPACKFullMatrixExt<double>>::AdditionalData(
        1.0));

    {
      deallog << "*** Real valued, left precondition" << std::endl;
      Vector<double>                x(n);
      SolverControl                 control(max_iter, abs_tol, true, true);
      SolverDQGMRES<Vector<double>> solver(control);
      solver.set_krylov_dim(krylov_dim);
      solver.set_orthogonal_history_length(ortho_hist_len);
      solver.set_left_precondition(true);
      solver.solve(A, x, b, precond);
      compare_with_jl_array(x, "x1_jacobi_left_precond", 1e-3, 1e-3);
    }

    {
      deallog << "*** Real valued, left precondition, using deal.ii solver"
              << std::endl;
      Vector<double> x(n);
      SolverControl  control(max_iter * krylov_dim, abs_tol, true, true);
      SolverGMRES<Vector<double>> solver(
        control,
        typename SolverGMRES<Vector<double>>::AdditionalData(krylov_dim + 2,
                                                             false,
                                                             true));
      solver.solve(A, x, b, precond);
      compare_with_jl_array(x, "x1_jacobi_left_precond", 1e-3, 1e-3);
    }

    {
      deallog << "*** Real valued, right precondition" << std::endl;
      Vector<double>                x(n);
      SolverControl                 control(max_iter, abs_tol, true, true);
      SolverDQGMRES<Vector<double>> solver(control);
      solver.set_krylov_dim(krylov_dim);
      solver.set_orthogonal_history_length(ortho_hist_len);
      solver.set_left_precondition(false);
      solver.solve(A, x, b, precond);
      compare_with_jl_array(x, "x1_jacobi_right_precond", 1e-3, 1e-3);
    }

    {
      deallog << "*** Real valued, right precondition, using deal.ii solver"
              << std::endl;
      Vector<double> x(n);
      SolverControl  control(max_iter * krylov_dim, abs_tol, true, true);
      SolverGMRES<Vector<double>> solver(
        control,
        typename SolverGMRES<Vector<double>>::AdditionalData(krylov_dim + 2,
                                                             true,
                                                             true));
      solver.solve(A, x, b, precond);
      compare_with_jl_array(x, "x1_jacobi_right_precond", 1e-3, 1e-3);
    }
  }

  {
    // Verify the complex valued case.

    // Get matrix and vector data from Julia.
    HBEMJuliaValue        jl_value = inst.eval_string("b2");
    const size_t          n        = jl_value.length();
    std::complex<double> *A_data   = inst.get_complex_double_array_var("A2");
    std::complex<double> *b_data   = inst.get_complex_double_array_var("b2");

    LAPACKFullMatrixExt<std::complex<double>> A;
    LAPACKFullMatrixExt<std::complex<double>>::Reshape(n, n, A_data, A);

    Vector<std::complex<double>> b(n);
    for (size_type i = 0; i < n; i++)
      b(i) = b_data[i];

    // Configure the solver.
    const size_type krylov_dim = 1000;
    // We use full orthogonalization.
    const size_type ortho_hist_len = krylov_dim;
    const size_type max_iter       = 1;
    const double    abs_tol        = 1e-3;

    {
      deallog << "*** complex valued, without preconditioner" << std::endl;
      Vector<std::complex<double>> x(n);
      SolverControl                control(max_iter, abs_tol, true, true);
      SolverDQGMRES<Vector<std::complex<double>>> solver(control);
      solver.set_krylov_dim(krylov_dim);
      solver.set_orthogonal_history_length(ortho_hist_len);
      solver.solve(A, x, b, PreconditionIdentity());
      compare_with_jl_array(x, "x2", 1e-3, 1e-3);
    }

    // Initialize the preconditioner.
    PreconditionJacobi<LAPACKFullMatrixExt<std::complex<double>>> precond;
    precond.initialize(
      A,
      typename PreconditionJacobi<
        LAPACKFullMatrixExt<std::complex<double>>>::AdditionalData(1.0));

    {
      deallog << "*** Complex valued, left precondition" << std::endl;
      Vector<std::complex<double>> x(n);
      SolverControl                control(max_iter, abs_tol, true, true);
      SolverDQGMRES<Vector<std::complex<double>>> solver(control);
      solver.set_krylov_dim(krylov_dim);
      solver.set_orthogonal_history_length(ortho_hist_len);
      solver.set_left_precondition(true);
      solver.solve(A, x, b, precond);
      compare_with_jl_array(x, "x2_jacobi_left_precond", 1e-3, 1e-3);
    }

    {
      deallog << "*** Complex valued, right precondition" << std::endl;
      Vector<std::complex<double>> x(n);
      SolverControl                control(max_iter, abs_tol, true, true);
      SolverDQGMRES<Vector<std::complex<double>>> solver(control);
      solver.set_krylov_dim(krylov_dim);
      solver.set_orthogonal_history_length(ortho_hist_len);
      solver.set_left_precondition(false);
      solver.solve(A, x, b, precond);
      compare_with_jl_array(x, "x2_jacobi_right_precond", 1e-3, 1e-3);
    }
  }

  INFO("*** test end");
}
