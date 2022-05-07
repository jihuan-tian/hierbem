/**
 * \file lapack-matrix-memory-consumption.cc
 * \brief Verify the memory consumption calculation for a @p LAPACKFullMatrixExt.
 *
 * \ingroup testers linalg
 * \author Jihuan Tian
 * \date 2022-05-06
 */

#include <boost/program_options.hpp>

#include <iostream>

#include "lapack_full_matrix_ext.h"
#include "unary_template_arg_containers.h"

using namespace boost::program_options;

int
main(int argc, char *argv[])
{
  unsigned int n;

  options_description opts("lapack-matrix-memory-consumption options");
  opts.add_options()("help,h", "Display this help")("dim,d",
                                                    value<unsigned int>(),
                                                    "Matrix dimension");
  variables_map vm;
  store(parse_command_line(argc, argv, opts), vm);
  notify(vm);

  if (vm.empty())
    {
      std::cout << opts << std::endl;
      return 0;
    }

  if (vm.count("help"))
    {
      std::cout << opts << std::endl;
      return 0;
    }

  if (vm.count("dim"))
    {
      n = vm["dim"].as<unsigned int>();
    }
  else
    {
      std::cout << "Please specify the matrix dimension!" << std::endl;
      return 0;
    }

  LAPACKFullMatrixExt<double> M;
  std::vector<double>         values(n * n);
  gen_linear_indices<vector_uta, double>(values, 1, 1.5);
  LAPACKFullMatrixExt<double>::Reshape(n, n, values, M);

  std::cout
    << "# Matrix dimension,Memory consumption,Coarse memory consumption\n";
  std::cout << n << "," << M.memory_consumption() << ","
            << M.memory_consumption_for_core_data() << std::endl;

  return 0;
}
