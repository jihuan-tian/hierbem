// Copyright (C) 2025-2026 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

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
  , value_ref(nullptr)
{}

HBEMJuliaValue::HBEMJuliaValue(const std::string &eval_str)
{
  value = jl_eval_string(eval_str.c_str());
  add_value_to_dict();
}

HBEMJuliaValue::HBEMJuliaValue(jl_value_t *val)
  : value(val)
{
  add_value_to_dict();
}

HBEMJuliaValue::~HBEMJuliaValue()
{
  remove_value_from_dict();
}

void
HBEMJuliaValue::add_value_to_dict()
{
  // Protect the Julia value until we wrap its reference to a container.
  JL_GC_PUSH1(&value);
  value_ref = jl_new_struct(HBEMJuliaWrapper::container_type, value);
  JL_GC_POP();

  // Add the container to the global dictionary in @p HBEMJuliaWrapper, so that the
  // Julia value will not be destroyed by GC.
  jl_call3(HBEMJuliaWrapper::setindex,
           HBEMJuliaWrapper::dict,
           value_ref,
           value_ref);
}

void
HBEMJuliaValue::remove_value_from_dict()
{
  jl_call2(HBEMJuliaWrapper::delete_func, HBEMJuliaWrapper::dict, value_ref);
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
  return (unsigned int *)jl_array_data((jl_array_t *)value, unsigned int);
}

int *
HBEMJuliaValue::int_array() const
{
  return (int *)jl_array_data((jl_array_t *)value, int);
}

float *
HBEMJuliaValue::float_array() const
{
  return (float *)jl_array_data((jl_array_t *)value, float);
}

double *
HBEMJuliaValue::double_array() const
{
  return (double *)jl_array_data((jl_array_t *)value, double);
}

std::complex<float> *
HBEMJuliaValue::complex_float_array() const
{
  return (std::complex<float> *)jl_array_data((jl_array_t *)value,
                                              std::complex<float>);
}

std::complex<double> *
HBEMJuliaValue::complex_double_array() const
{
  return (std::complex<double> *)jl_array_data((jl_array_t *)value,
                                               std::complex<double>);
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

std::mutex     HBEMJuliaWrapper::julia_mutex;
jl_value_t    *HBEMJuliaWrapper::dict;
jl_function_t *HBEMJuliaWrapper::setindex;
jl_function_t *HBEMJuliaWrapper::delete_func;
jl_datatype_t *HBEMJuliaWrapper::container_type;

HBEMJuliaWrapper &
HBEMJuliaWrapper::get_instance()
{
  static HBEMJuliaWrapper instance;
  return instance;
}

HBEMJuliaWrapper::HBEMJuliaWrapper()
{
  // Initialize a Julia session.
  jl_init();

  HBEMJuliaWrapper::dict        = jl_eval_string("refs = IdDict()");
  HBEMJuliaWrapper::setindex    = jl_get_function(jl_base_module, "setindex!");
  HBEMJuliaWrapper::delete_func = jl_get_function(jl_base_module, "delete!");
  HBEMJuliaWrapper::container_type =
    (jl_datatype_t *)jl_eval_string("Base.RefValue{Any}");
}

HBEMJuliaWrapper::~HBEMJuliaWrapper()
{
  // Exit from the Julia session.
  jl_atexit_hook(0);
}

jl_value_t *
HBEMJuliaWrapper::eval_string(const std::string &eval_str) const
{
  std::lock_guard<std::mutex> lock(julia_mutex);
  return jl_eval_string(eval_str.c_str());
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

void
HBEMJuliaWrapper::source_file(const std::string &file_name) const
{
  std::lock_guard<std::mutex> lock(julia_mutex);
  (void)jl_eval_string(
    (std::string("include(\"") + file_name + std::string("\")")).c_str());
}

HBEM_NS_CLOSE
