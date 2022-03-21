// File: erichsen1996efficient-example2.cc
// Description: This program solves Laplace equation with Neumann boundary
// condition using boundary element method.
// @author: Jihuan Tian
// @date: 2020-11-26
// Copyright (C) 2020 Jihuan Tian <jihuan_tian@hotmail.com>

#include <deal.II/base/logstream.h>

#include <boost/program_options.hpp>

#include <erichsen1996efficient_example2.h>

using namespace dealii;
using namespace boost::program_options;

int
main(int argc, char *argv[])
{
  (void)argc;

  deallog.depth_console(2);
  deallog.pop();

  std::string  mesh_file_name;
  unsigned int fe_order;
  unsigned int thread_num;
  unsigned int n_min;
  double       eta;

  options_description opts("erichsen1996efficient-example2 options");
  opts.add_options()("help,h", "Display this help")("input,i",
                                                    value<std::string>(),
                                                    "Path to the mesh file")(
    "fe_order,o", value<unsigned int>(), "Finite element order")(
    "threads,t", value<unsigned int>(), "Number of threads")(
    "nmin,n", value<unsigned int>(), "Minimum cluster size/cardinality")(
    "adm,a", value<double>(), "Admissibility condition constant");

  variables_map vm;
  store(parse_command_line(argc, argv, opts), vm);
  notify(vm);

  if (vm.empty())
    {
      std::cout << "Please provide command line options!" << std::endl;
      std::cout << opts << std::endl;
      return 0;
    }

  if (vm.count("help"))
    {
      std::cout << opts << std::endl;
      return 0;
    }

  if (vm.count("input"))
    {
      mesh_file_name = vm["input"].as<std::string>();
    }
  else
    {
      std::cout << "Please provide the mesh file name!" << std::endl;
      return 0;
    }

  if (vm.count("fe_order"))
    {
      fe_order = vm["fe_order"].as<unsigned int>();
    }
  else
    {
      fe_order = 1;
    }

  if (vm.count("threads"))
    {
      thread_num = vm["threads"].as<unsigned int>();
    }
  else
    {
      thread_num = 4;
    }

  if (vm.count("nmin"))
    {
      n_min = vm["nmin"].as<unsigned int>();
    }
  else
    {
      n_min = 2;
      std::cout
        << "Minimum cluster size/cardinality has been set to the default value: 2"
        << std::endl;
    }

  if (vm.count("adm"))
    {
      eta = vm["adm"].as<double>();
    }
  else
    {
      eta = 1.0;
      std::cout
        << "Admissibility constant eta has been set to the default value: 1.0"
        << std::endl;
    }

  IdeoBEM::Erichsen1996Efficient::Example2 testcase(
    mesh_file_name, fe_order, thread_num, n_min, eta);
  testcase.run();

  return 0;
}
