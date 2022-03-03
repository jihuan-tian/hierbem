// File: erichsen1996efficient-example2.cc
// Description: This program solves Laplace equation with Neumann boundary
// condition using boundary element method.
// @author: Jihuan Tian
// @date: 2020-11-26
// Copyright (C) 2020 Jihuan Tian <jihuan_tian@hotmail.com>

#include <deal.II/base/logstream.h>

#include <erichsen1996efficient_example2.h>

using namespace dealii;

int
main(int argc, char *argv[])
{
  (void)argc;

  deallog.depth_console(2);
  deallog.pop();

  std::string        mesh_file_name(argv[1]);
  std::string        fe_order_str(argv[2]);
  std::string        proc_num_str(argv[3]);
  const unsigned int fe_order = std::stoi(fe_order_str);
  const unsigned int proc_num = std::stoi(proc_num_str);
  LaplaceBEM::Erichsen1996Efficient::Example2 testcase(mesh_file_name,
                                                       fe_order,
                                                       proc_num);
  testcase.run();

  return 0;
}
