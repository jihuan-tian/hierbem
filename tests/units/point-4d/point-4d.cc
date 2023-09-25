// File: point-4d.cc
// Description:
// Author: Jihuan Tian
// Date: 2020-11-16
// Copyright (C) 2020 Jihuan Tian <jihuan_tian@hotmail.com>

#include <deal.II/base/logstream.h>
#include <deal.II/base/point.h>

using namespace dealii;

int main()
{
  deallog.depth_console(2);
  deallog.pop();

  const unsigned int dim = 4;
  Point<dim> pt;

  pt(0) = 1.;
  pt(1) = 2.;
  pt(2) = 3.;
  pt(3) = 4.;

  deallog << pt;

  return 0;
}
