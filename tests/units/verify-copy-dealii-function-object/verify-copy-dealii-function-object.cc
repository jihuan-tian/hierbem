// Copyright (C) 2022-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * \file verify-copy-dealii-function-object.cc
 * \brief Verify the evaluation of a inherited deal.ii function after being
 * copied to the parent class @p Function.
 *
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2022-06-22
 */

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <iostream>

using namespace dealii;

class DirichletBC : public Function<3>
{
public:
  DirichletBC()
    : Function<3>()
    , x0(0.25, 0.25, 0.25)
  {}

  DirichletBC(const Point<3> &x0)
    : Function<3>()
    , x0(x0)
  {}

  double
  value(const Point<3> &p, const unsigned int component = 0) const
  {
    (void)component;
    return 1.0 / 4.0 / numbers::PI / (p - x0).norm();
  }

private:
  Point<3> x0;
};

int
main()
{
  DirichletBC dirichlet_bc;

  /**
   * Evaluate the inherited function object.
   */
  std::cout << "DirichletBC(0,0,0)=" << dirichlet_bc.value(Point<3>(0, 0, 0))
            << std::endl;

  /**
   * @p Function is an abstract class, which cannot be directly created.
   */
  // Function<3> func(dirichlet_bc);

  /**
   * Then we create a reference type of @p Function instead. Because the member
   * function @p value is virtual, polymorphism is enabled. Hence, the function
   * can be correctly evaluated.
   */
  Function<3> &func_ref = dirichlet_bc;
  std::cout << "Func_ref(0,0,0)=" << func_ref.value(Point<3>(0, 0, 0))
            << std::endl;

  /**
   * We further create a pointer type of @p Function, which should also work.
   */
  Function<3> *func_ptr = &dirichlet_bc;
  std::cout << "Func_ptr(0,0,0)=" << func_ptr->value(Point<3>(0, 0, 0))
            << std::endl;

  return 0;
}
