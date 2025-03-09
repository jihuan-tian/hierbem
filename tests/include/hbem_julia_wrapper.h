/**
 * @file hbem_julia_wrapper.h
 * @brief Interface class for calling Julia API.
 *
 * @author Jihuan Tian
 * @date 2025-03-07
 */

#ifndef HIERBEM_TESTS_INCLUDE_HBEM_JULIA_WRAPPER_H_
#define HIERBEM_TESTS_INCLUDE_HBEM_JULIA_WRAPPER_H_

#include <julia.h>

#include <complex>
#include <mutex>

#include "config.h"

HBEM_NS_OPEN

/**
 * Class wrapping the pointer to Julia value.
 */
class HBEMJuliaValue
{
public:
  HBEMJuliaValue();
  HBEMJuliaValue(jl_value_t *val);
  HBEMJuliaValue(const HBEMJuliaValue &other);
  HBEMJuliaValue &
  operator=(const HBEMJuliaValue &other);

  unsigned int
  uint_value() const;

  int
  int_value() const;

  float
  float_value() const;

  double
  double_value() const;

  // N.B. At the moment, there is no @p complex_float_value or
  // @p complex_double_value, because @p std::complex is neither a simple basic
  // type or an array type, which cannot be directly obtained.

  float *
  float_array() const;

  double *
  double_array() const;

  std::complex<float> *
  complex_float_array() const;

  std::complex<double> *
  complex_double_array() const;

  template <typename Number>
  Number *
  array() const;

  /**
   * Get the number of rows in an array. When the array is 1D, this is the
   * number of elements.
   */
  size_t
  nrows() const;

  /**
   * Get the number of dimensions in the array.
   */
  size_t
  ndims() const;

  /**
   * Get the array size in the @p dim-th dimension. @p dim starts from 0.
   */
  size_t
  size(const size_t dim) const;

  /**
   * Get the total number of elements in the multi-dimensional array.
   */
  size_t
  length() const;

private:
  jl_value_t *value;
};


class HBEMJuliaWrapper
{
public:
  static HBEMJuliaWrapper &
  get_instance();

  // Delete default copy constructor and assignment operator to make this class
  // a singleton.
  HBEMJuliaWrapper(const HBEMJuliaWrapper &) = delete;
  HBEMJuliaWrapper &
  operator=(const HBEMJuliaWrapper &) = delete;

  /**
   * Evalute a Julia expression which has a return value.
   */
  HBEMJuliaValue
  eval_string(const std::string &eval_str) const;

  unsigned int
  get_uint_var(const std::string &var_name) const;

  int
  get_int_var(const std::string &var_name) const;

  float
  get_float_var(const std::string &var_name) const;

  double
  get_double_var(const std::string &var_name) const;

  std::complex<float>
  get_complex_float_var(const std::string &var_name) const;

  std::complex<double>
  get_complex_double_var(const std::string &var_name) const;

  float *
  get_float_array_var(const std::string &var_name) const;

  double *
  get_double_array_var(const std::string &var_name) const;

  std::complex<float> *
  get_complex_float_array_var(const std::string &var_name) const;

  std::complex<double> *
  get_complex_double_array_var(const std::string &var_name) const;

  template <typename Number>
  Number *
  get_array_var(const std::string &var_name) const;

  /**
   * Source a Julia script file.
   */
  void
  source_file(const std::string &file_name) const;

private:
  /**
   * Mutex for protecting Julia session from multithreads.
   */
  static std::mutex julia_mutex;

  HBEMJuliaWrapper();

  ~HBEMJuliaWrapper();
};


template <typename Number>
Number *
HBEMJuliaValue::array() const
{
  return (Number *)jl_array_data((jl_array_t *)value);
}


template <typename Number>
Number *
HBEMJuliaWrapper::get_array_var(const std::string &var_name) const
{
  HBEMJuliaValue val = eval_string(var_name);
  return val.array<Number>();
}

HBEM_NS_CLOSE

#endif /* HIERBEM_TESTS_INCLUDE_HBEM_JULIA_WRAPPER_H_ */
