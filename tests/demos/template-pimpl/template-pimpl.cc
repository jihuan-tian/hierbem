// Copyright (C) 2024 Xiaozhe Wang <chaoslawful@gmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

#include <catch2/catch_all.hpp>
#include <fmt/format.h>

#include "cu_sbb.h"
#include "sbb.h"

using namespace Catch::Matchers;

TEST_CASE("Construct pre-instantiated objects", "[pimpl][demo]")
{
  {
    SBB<2, float> o;
    o.set_value(2.0f);
    o.print();
    o.svar = 1.234;
    std::cout << o.svar << std::endl;
  }

  {
    SBB<3, double> o;
    o.set_value(3.0);
    o.print();
    REQUIRE(o.svar != SBB<2, float>::svar);
  }
}

TEST_CASE("Pre-instantiate ostream support", "[pimpl][demo]")
{
  SBB<2, float> o;
  o.set_value(2.0f);
  std::cout << o << std::endl;
}

TEST_CASE("Pre-instantiate templated ctor", "[pimpl][demo]")
{
  {
    Mapping<3, 2> m;
    SBB<2, float> o(m);
    o.print();
  }

  {
    Mapping<1, 2>  m;
    SBB<2, double> o(m);
    o.print();
  }
}

TEST_CASE("Test inline func", "[pimpl][demo]")
{
  SBB<2, float> o;
  REQUIRE(o.inline_func(2) == 20);
}

TEST_CASE("Test CUDA kernel", "[pimpl][demo]")
{
  CuSbb<float> cu;
  int          n = 100;
  float       *a, *b, *c;
  a = new float[n];
  b = new float[n];
  c = new float[n];

  for (int i = 0; i < n; ++i)
    {
      a[i] = 1.0f;
      b[i] = 2.0f;
    }

  cu.vec_add(a, b, c, n);

  for (int i = 0; i < n; ++i)
    {
      REQUIRE(c[i] == 3.0f);
    }

  delete[] a;
  delete[] b;
  delete[] c;
}
