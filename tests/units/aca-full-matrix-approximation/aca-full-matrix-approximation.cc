/**
 * \file aca-full-matrix-approximation.cc
 * \brief
 * \ingroup testers
 * \author Jihuan Tian
 * \date 2022-03-16
 */

#include <boost/program_options.hpp>

#include <fstream>
#include <iostream>

#include "hmatrix/aca_plus/aca_plus.hcu"
#include "read_octave_data.h"

using namespace HierBEM;
using namespace boost::program_options;

int
main(int argc, char *argv[])
{
  (void)argc;

  std::string  matrix_file_name;
  std::string  matrix_var_name;
  unsigned int max_iter;
  double       epsilon;
  double       eta;

  options_description opts("aca-full-matrix-approximation options");
  opts.add_options()("help,h", "Display this help")(
    "input,i", value<std::string>(), "Input file for the matrix data")(
    "var,v", value<std::string>(), "Variable name of the matrix")(
    "max_iter,n", value<unsigned int>(), "Maximum ACA iteration")(
    "epsilon,e", value<double>(), "Relative approximation error")(
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
      matrix_file_name = vm["input"].as<std::string>();
    }
  else
    {
      std::cout << "Please provide the input file name!" << std::endl;
    }

  if (vm.count("var"))
    {
      matrix_var_name = vm["var"].as<std::string>();
    }
  else
    {
      matrix_var_name = std::string("A");
      std::cout << "Matrix name is set to the default: A" << std::endl;
    }

  if (vm.count("max_iter"))
    {
      max_iter = vm["max_iter"].as<unsigned int>();
    }
  else
    {
      max_iter = 4;
      std::cout << "Maximum ACA iteration is set to " << max_iter << std::endl;
    }

  if (vm.count("epsilon"))
    {
      epsilon = vm["epsilon"].as<double>();
    }
  else
    {
      epsilon = 1e-2;
      std::cout << "Relative approximation error is set to " << epsilon
                << std::endl;
    }

  if (vm.count("adm"))
    {
      eta = vm["adm"].as<double>();
    }
  else
    {
      eta = 2.0;
      std::cout << "Admissibility condition constant is set to " << eta
                << std::endl;
    }

  LAPACKFullMatrixExt<double> A;
  std::ifstream               in(matrix_file_name);
  A.read_from_mat(in, matrix_var_name);
  in.close();

  ACAConfig aca_conf(max_iter, epsilon, eta);

  const unsigned int m = A.m();
  const unsigned int n = A.n();

  RkMatrix<double> rkmat(m, n, max_iter);

  aca_plus(rkmat, aca_conf, A);

  rkmat.print_formatted_to_mat(std::cout, "R", 15, false, 25, "0");

  return 0;
}
