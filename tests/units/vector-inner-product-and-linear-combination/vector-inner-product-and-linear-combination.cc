/**
 * @file vector-inner-product-and-linear-combination.cc
 * @brief Verify inner product and linear combination of two vectors
 * @ingroup linalg
 *
 * @date 2025-04-23
 * @author Jihuan Tian
 */

#include <deal.II/base/types.h>

#include <deal.II/lac/vector.h>

#include <catch2/catch_all.hpp>

#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>

#include "hbem_julia_cpp_compare.h"
#include "hbem_julia_wrapper.h"
#include "linalg.h"

using namespace Catch::Matchers;
using namespace dealii;
using namespace HierBEM;

template <typename Number>
void
copy_data_to_vector(const Number *data, Vector<Number> &vec)
{
  for (unsigned int i = 0; i < vec.size(); i++)
    vec[i] = data[i];
}

TEST_CASE("Inner product and linear combination of two vectors", "[linalg]")
{
  INFO("*** test start");
  HBEMJuliaWrapper &inst = HBEMJuliaWrapper::get_instance();
  inst.source_file("process.jl");

  // Read vectors from Julia.
  double               *v1_array = inst.get_double_array_var("v1");
  double               *v2_array = inst.get_double_array_var("v2");
  std::complex<double> *v1_complex_array =
    inst.get_complex_double_array_var("v1_complex");
  std::complex<double> *v2_complex_array =
    inst.get_complex_double_array_var("v2_complex");

  const unsigned int           n = 16 * 1024;
  Vector<double>               v1(n);
  Vector<double>               v2(n);
  Vector<std::complex<double>> v1_complex(n);
  Vector<std::complex<double>> v2_complex(n);

  copy_data_to_vector(v1_array, v1);
  copy_data_to_vector(v2_array, v2);
  copy_data_to_vector(v1_complex_array, v1_complex);
  copy_data_to_vector(v2_complex_array, v2_complex);

  double               result;
  std::complex<double> result_complex;

  result = LinAlg::inner_product(v1, v2);
  std::cout << std::defaultfloat << std::setprecision(16) << std::showpoint
            << result << std::endl;
  compare_with_jl_scalar(result, "inner_product1", 1e-10, 1e-10);

  result = LinAlg::inner_product_tbb(v1, v2);
  std::cout << std::defaultfloat << std::setprecision(16) << std::showpoint
            << result << std::endl;
  compare_with_jl_scalar(result, "inner_product1", 1e-10, 1e-10);

  result_complex = LinAlg::inner_product(v1_complex, v2_complex);
  std::cout << std::defaultfloat << std::setprecision(16) << std::showpoint
            << result_complex << std::endl;
  compare_with_jl_complex(result_complex, "inner_product2", 1e-10, 1e-10);

  result_complex = LinAlg::inner_product_tbb(v1_complex, v2_complex);
  std::cout << std::defaultfloat << std::setprecision(16) << std::showpoint
            << result_complex << std::endl;
  compare_with_jl_complex(result_complex, "inner_product2", 1e-10, 1e-10);

  result_complex = LinAlg::inner_product(v1_complex, v2);
  std::cout << std::defaultfloat << std::setprecision(16) << std::showpoint
            << result_complex << std::endl;
  compare_with_jl_complex(result_complex, "inner_product3", 1e-10, 1e-10);

  result_complex = LinAlg::inner_product_tbb(v1_complex, v2);
  std::cout << std::defaultfloat << std::setprecision(16) << std::showpoint
            << result_complex << std::endl;
  compare_with_jl_complex(result_complex, "inner_product3", 1e-10, 1e-10);

  result = LinAlg::linear_combination(v1, v2);
  std::cout << std::defaultfloat << std::setprecision(16) << std::showpoint
            << result << std::endl;
  compare_with_jl_scalar(result, "linear_combination1", 1e-10, 1e-10);

  result = LinAlg::linear_combination_tbb(v1, v2);
  std::cout << std::defaultfloat << std::setprecision(16) << std::showpoint
            << result << std::endl;
  compare_with_jl_scalar(result, "linear_combination1", 1e-10, 1e-10);

  result_complex = LinAlg::linear_combination(v1_complex, v2_complex);
  std::cout << std::defaultfloat << std::setprecision(16) << std::showpoint
            << result_complex << std::endl;
  compare_with_jl_complex(result_complex, "linear_combination2", 1e-10, 1e-10);

  result_complex = LinAlg::linear_combination_tbb(v1_complex, v2_complex);
  std::cout << std::defaultfloat << std::setprecision(16) << std::showpoint
            << result_complex << std::endl;
  compare_with_jl_complex(result_complex, "linear_combination2", 1e-10, 1e-10);

  result_complex = LinAlg::linear_combination(v1_complex, v2);
  std::cout << std::defaultfloat << std::setprecision(16) << std::showpoint
            << result_complex << std::endl;
  compare_with_jl_complex(result_complex, "linear_combination3", 1e-10, 1e-10);

  result_complex = LinAlg::linear_combination_tbb(v1_complex, v2);
  std::cout << std::defaultfloat << std::setprecision(16) << std::showpoint
            << result_complex << std::endl;
  compare_with_jl_complex(result_complex, "linear_combination3", 1e-10, 1e-10);

  INFO("*** test end");
}
