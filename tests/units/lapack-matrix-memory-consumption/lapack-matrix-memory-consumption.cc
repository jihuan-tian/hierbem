/**
 * \file lapack-matrix-memory-consumption.cc
 * \brief Verify the memory consumption calculation for a @p LAPACKFullMatrixExt.
 *
 * \ingroup testers linalg
 * \author Jihuan Tian
 * \date 2022-05-06
 */

#include <iostream>

#include "lapack_full_matrix_ext.h"
#include "unary_template_arg_containers.h"

using namespace HierBEM;

int
main()
{
  std::cout << "# Matrix dimension,Memory consumption\n";

  {
    unsigned int n = 10;

    LAPACKFullMatrixExt<double> M;
    std::vector<double>         values(n * n);
    gen_linear_indices<vector_uta, double>(values, 1, 1.5);
    LAPACKFullMatrixExt<double>::Reshape(n, n, values, M);

    std::cout << n << "," << M.memory_consumption() << std::endl;
  }

  {
    unsigned int n = 100;

    LAPACKFullMatrixExt<double> M;
    std::vector<double>         values(n * n);
    gen_linear_indices<vector_uta, double>(values, 1, 1.5);
    LAPACKFullMatrixExt<double>::Reshape(n, n, values, M);

    std::cout << n << "," << M.memory_consumption() << std::endl;
  }

  {
    unsigned int n = 1000;

    LAPACKFullMatrixExt<double> M;
    std::vector<double>         values(n * n);
    gen_linear_indices<vector_uta, double>(values, 1, 1.5);
    LAPACKFullMatrixExt<double>::Reshape(n, n, values, M);

    std::cout << n << "," << M.memory_consumption() << std::endl;
  }


  {
    unsigned int n = 10000;

    LAPACKFullMatrixExt<double> M;
    std::vector<double>         values(n * n);
    gen_linear_indices<vector_uta, double>(values, 1, 1.5);
    LAPACKFullMatrixExt<double>::Reshape(n, n, values, M);

    std::cout << n << "," << M.memory_consumption() << std::endl;
  }

  return 0;
}
