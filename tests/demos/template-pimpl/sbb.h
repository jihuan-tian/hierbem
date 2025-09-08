// Copyright (C) 2024 Xiaozhe Wang <chaoslawful@gmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

#ifndef SBB_H__
#define SBB_H__

#include <iostream>
#include <memory>

#include <experimental/propagate_const>

template <int dim1, int dim2>
class Mapping
{};

template <int dim, typename T>
class SBB
{
  struct impl;

  // use `propagate_const` to ensure that object constness is propagated to
  // implementations in order to prevent calling non-const implementation
  // functions on const interface objects
  std::experimental::propagate_const<std::unique_ptr<impl>> impl_;

  template <int dim1, typename T1>
  friend std::ostream &
  operator<<(std::ostream &out, const SBB<dim1, T1> &bbox);

public:
  static T svar;

  SBB();  // default ctor
  ~SBB(); // dtor should be defined in the implementation file
          // for std::unique_ptr to work properly

  SBB(const SBB<dim, T> &other); // copy ctor
  SBB<dim, T> &
  operator=(const SBB<dim, T> &other); // copy assignment

  SBB(SBB<dim, T> &&other); // move ctor
  SBB<dim, T> &
  operator=(SBB<dim, T> &&other); // move assignment

  SBB(const T &&value); // direct-init ctor

  template <int dim1>
  SBB(const Mapping<dim1, dim> &mapping); // template ctor

  void
  print() const;
  T
  get_value() const;
  void
  set_value(T value);
  size_t
  get_size() const;
  std::string
  to_string() const;
  T
  inline_func(T x) const
  {
    return T(10) * x;
  }
};

// ostream support for SBB, not confidential so it can be defined here
template <int dim, typename T>
std::ostream &
operator<<(std::ostream &out, const SBB<dim, T> &bbox)
{
  out << bbox.to_string();
  return out;
}

#endif // SBB_H__
