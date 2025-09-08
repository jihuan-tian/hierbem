// Copyright (C) 2023 Xiaozhe Wang <chaoslawful@gmail.com>
// Copyright (C) 2023-2024 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * \file hbem_octave_wrapper.h
 * \brief Simple singleton wrapper around minimal Octave APIs to workaround
 * naming conflictions between Octave and CUDA and ease interacting
 * with Octave in testcases.
 *
 * \warning This header includes `octave/ov.h` for `octave_value` definition,
 * but it contains much more Octave types (Matrix and SparseMatrix eg) which
 * may conflict with HierBEM. If it is  the case, you have to extract all
 * HierBEM related logics into another source file and wrap them into a function
 * to be called by Octave testbase.
 *
 * \author Xiaozhe Wang
 * \date 2023-10-12
 */
#ifndef HBEM_OCTAVE_WRAPPER_H_
#define HBEM_OCTAVE_WRAPPER_H_

#include <string>
#include <vector>

class octave_value;
namespace HierBEM
{
  // Hide Octave related details
  struct HBEMOctaveValueImpl;
  class HBEMOctaveValue
  {
  public:
    HBEMOctaveValue();
    HBEMOctaveValue(octave_value const &ov);
    HBEMOctaveValue(HBEMOctaveValue const &other);
    void
    operator=(HBEMOctaveValue const &other);
    ~HBEMOctaveValue();

    // Old-fashioned proxy methods
    int
    int_value() const;
    double
    double_value() const;
    void
    matrix_value(std::vector<double> &mat_data,
                 unsigned int        &m,
                 unsigned int        &n);

  private:
    HBEMOctaveValueImpl *m_impl;
  };

  using HBEMOctaveValueList = std::vector<HBEMOctaveValue>;

  struct HBEMOctaveWrapperImpl;
  class HBEMOctaveWrapper
  {
  public:
    static HBEMOctaveWrapper &
    get_instance();

    // Deleting default copy methods to keep instance uncopyable
    HBEMOctaveWrapper(HBEMOctaveWrapper const &) = delete;
    void
    operator=(HBEMOctaveWrapper const &) = delete;

    ~HBEMOctaveWrapper();

    // Evaluate a string, which can be either an expression or a function with
    // return value(s).
    HBEMOctaveValue
    eval_string(const std::string &eval_str);
    // Evaluate a function which has no return value.
    void
    eval_function_void(const std::string &eval_str);
    // Evaluate a function which return a scalar value.
    HBEMOctaveValue
    eval_function_scalar(const std::string &eval_str);
    // Evaluate a function which can have any number of value.
    void
    eval_function(const std::string   &eval_str,
                  int                  nargout,
                  HBEMOctaveValueList &value_list);
    void
    source_file(const std::string &file_name);
    void
    add_path(const std::string &path);

  private:
    HBEMOctaveWrapper();

    HBEMOctaveWrapperImpl *m_impl;
  };
} // namespace HierBEM

#endif
