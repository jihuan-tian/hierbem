/**
 * @file hbem_julia_wrapper.cc
 * @brief Interface class for calling Julia API.
 *
 * @author Jihuan Tian
 * @date 2025-03-07
 */

#include "hbem_julia_wrapper.h"

#include <complex>
#include <mutex>

HBEM_NS_OPEN

HBEMJuliaValue::HBEMJuliaValue()
  : value(nullptr)
{}

HBEMJuliaValue::HBEMJuliaValue(jl_value_t *val)
  : value(val)
{}

HBEMJuliaValue::HBEMJuliaValue(const HBEMJuliaValue &other)
{
  value = other.value;
}

HBEMJuliaValue &
HBEMJuliaValue::operator=(const HBEMJuliaValue &other)
{
  value = other.value;
  return *this;
}

unsigned int
HBEMJuliaValue::uint_value() const
{
  return jl_unbox_uint32(value);
}

int
HBEMJuliaValue::int_value() const
{
  return jl_unbox_int32(value);
}

float
HBEMJuliaValue::float_value() const
{
  return jl_unbox_float32(value);
}

double
HBEMJuliaValue::double_value() const
{
  return jl_unbox_float64(value);
}

unsigned int *
HBEMJuliaValue::uint_array() const
{
  return (unsigned int *)jl_array_data((jl_array_t *)value);
}

int *
HBEMJuliaValue::int_array() const
{
  return (int *)jl_array_data((jl_array_t *)value);
}

float *
HBEMJuliaValue::float_array() const
{
  return (float *)jl_array_data((jl_array_t *)value);
}

double *
HBEMJuliaValue::double_array() const
{
  return (double *)jl_array_data((jl_array_t *)value);
}

std::complex<float> *
HBEMJuliaValue::complex_float_array() const
{
  return (std::complex<float> *)jl_array_data((jl_array_t *)value);
}

std::complex<double> *
HBEMJuliaValue::complex_double_array() const
{
  return (std::complex<double> *)jl_array_data((jl_array_t *)value);
}

size_t
HBEMJuliaValue::nrows() const
{
  return jl_array_nrows((jl_array_t *)value);
}

size_t
HBEMJuliaValue::ndims() const
{
  return jl_array_ndims((jl_array_t *)value);
}

size_t
HBEMJuliaValue::size(const size_t dim) const
{
  return jl_array_dim((jl_array_t *)value, dim);
}

size_t
HBEMJuliaValue::length() const
{
  return jl_array_len((jl_array_t *)value);
}

std::mutex HBEMJuliaWrapper::julia_mutex;

HBEMJuliaWrapper &
HBEMJuliaWrapper::get_instance()
{
  static HBEMJuliaWrapper instance;
  return instance;
}

HBEMJuliaWrapper::HBEMJuliaWrapper()
{
  jl_init();
}

HBEMJuliaWrapper::~HBEMJuliaWrapper()
{
  jl_atexit_hook(0);
}

HBEMJuliaValue
HBEMJuliaWrapper::eval_string(const std::string &eval_str) const
{
  std::lock_guard<std::mutex> lock(julia_mutex);
  return HBEMJuliaValue(jl_eval_string(eval_str.c_str()));
}

unsigned int
HBEMJuliaWrapper::get_uint_var(const std::string &var_name) const
{
  HBEMJuliaValue val = eval_string(var_name);
  return val.uint_value();
}

int
HBEMJuliaWrapper::get_int_var(const std::string &var_name) const
{
  HBEMJuliaValue val = eval_string(var_name);
  return val.int_value();
}

float
HBEMJuliaWrapper::get_float_var(const std::string &var_name) const
{
  HBEMJuliaValue val = eval_string(var_name);
  return val.float_value();
}

double
HBEMJuliaWrapper::get_double_var(const std::string &var_name) const
{
  HBEMJuliaValue val = eval_string(var_name);
  return val.double_value();
}

std::complex<float>
HBEMJuliaWrapper::get_complex_float_var(const std::string &var_name) const
{
  // We need to evaluate two times: one for the real part, the other for the
  // imaginary part.
  HBEMJuliaValue real_part =
    eval_string(std::string("real(") + var_name + std::string(")"));
  HBEMJuliaValue imag_part =
    eval_string(std::string("imag(") + var_name + std::string(")"));

  return std::complex<float>(real_part.float_value(), imag_part.float_value());
}

std::complex<double>
HBEMJuliaWrapper::get_complex_double_var(const std::string &var_name) const
{
  // We need to evaluate two times: one for the real part, the other for the
  // imaginary part.
  HBEMJuliaValue real_part =
    eval_string(std::string("real(") + var_name + std::string(")"));
  HBEMJuliaValue imag_part =
    eval_string(std::string("imag(") + var_name + std::string(")"));

  return std::complex<double>(real_part.double_value(),
                              imag_part.double_value());
}

unsigned int *
HBEMJuliaWrapper::get_uint_array_var(const std::string &var_name) const
{
  HBEMJuliaValue val = eval_string(var_name);
  return val.uint_array();
}

int *
HBEMJuliaWrapper::get_int_array_var(const std::string &var_name) const
{
  HBEMJuliaValue val = eval_string(var_name);
  return val.int_array();
}

float *
HBEMJuliaWrapper::get_float_array_var(const std::string &var_name) const
{
  HBEMJuliaValue val = eval_string(var_name);
  return val.float_array();
}

double *
HBEMJuliaWrapper::get_double_array_var(const std::string &var_name) const
{
  HBEMJuliaValue val = eval_string(var_name);
  return val.double_array();
}

std::complex<float> *
HBEMJuliaWrapper::get_complex_float_array_var(const std::string &var_name) const
{
  HBEMJuliaValue val = eval_string(var_name);
  return val.complex_float_array();
}

std::complex<double> *
HBEMJuliaWrapper::get_complex_double_array_var(
  const std::string &var_name) const
{
  HBEMJuliaValue val = eval_string(var_name);
  return val.complex_double_array();
}

void
HBEMJuliaWrapper::source_file(const std::string &file_name) const
{
  std::lock_guard<std::mutex> lock(julia_mutex);
  (void)jl_eval_string(
    (std::string("include(\"") + file_name + std::string("\")")).c_str());
}

HBEM_NS_CLOSE
