/**
 * \file erichsen1996efficient-example2-hmat-neumann-bc.cc
 * \brief
 * \ingroup testers
 * \author Jihuan Tian
 * \date 2022-05-14
 */

#include <deal.II/base/multithread_info.h>

#include <boost/program_options.hpp>

#include "erichsen1996efficient_example2.h"

using namespace dealii;
using namespace boost::program_options;

int
main(int argc, char *argv[])
{
  (void)argc;

  std::string  mesh_file_name;
  unsigned int fe_order;
  unsigned int thread_num;
  char         problem_type_string;
  unsigned int n_min_for_ct;
  unsigned int n_min_for_bct;
  double       eta;
  unsigned int max_hmat_rank;
  double       aca_relative_error;

  options_description opts(
    "erichsen1996efficient-example2-hmat-neumann-bc options");
  opts.add_options()("help,h", "Display this help")("input,i",
                                                    value<std::string>(),
                                                    "Path to the mesh file")(
    "fe_order,o", value<unsigned int>(), "Finite element order")(
    "threads,t", value<unsigned int>(), "Number of threads")(
    "problem,p",
    value<char>(),
    "Problem type: 'd' for Dirichlet problem, 'n' for Neumann problem")(
    "nmin_ct,n", value<unsigned int>(), "Minimum cluster size/cardinality")(
    "nmin_bct,N",
    value<unsigned int>(),
    "Minimum block cluster size/cardinality")(
    "adm,a", value<double>(), "Admissibility condition constant")(
    "rank,r", value<unsigned int>(), "Maximum rank for the H-matrix")(
    "epsilon,e", value<double>(), "ACA+ relative error");

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
      std::cout << "Mesh file: " << mesh_file_name << std::endl;
    }
  else
    {
      std::cout << "Please provide the mesh file name!" << std::endl;
      return 0;
    }

  if (vm.count("fe_order"))
    {
      fe_order = vm["fe_order"].as<unsigned int>();
      std::cout << "Finite element order: " << fe_order << std::endl;
    }
  else
    {
      fe_order = 1;
      std::cout << "Finite element order has been set to the default value: 1"
                << std::endl;
    }

  if (vm.count("threads"))
    {
      thread_num = vm["threads"].as<unsigned int>();
      std::cout << "Number of threads: " << thread_num << std::endl;
    }
  else
    {
      thread_num = MultithreadInfo::n_cores();
      std::cout << "Number of threads has been set to the default value: "
                << thread_num << std::endl;
    }

  if (vm.count("problem"))
    {
      problem_type_string = vm["problem"].as<char>();
    }
  else
    {
      problem_type_string = 'n';
    }

  std::cout << "Problem type is: "
            << (problem_type_string == 'n' ? "Neumann" : "Dirichlet")
            << std::endl;

  if (vm.count("nmin_ct"))
    {
      n_min_for_ct = vm["nmin_ct"].as<unsigned int>();
      std::cout << "Minimum cluster size: " << n_min_for_ct << std::endl;
    }
  else
    {
      n_min_for_ct = 2;
      std::cout
        << "Minimum cluster size/cardinality has been set to the default value: 2"
        << std::endl;
    }

  if (vm.count("nmin_bct"))
    {
      n_min_for_bct = vm["nmin_bct"].as<unsigned int>();
      std::cout << "Minimum block cluster size: " << n_min_for_bct << std::endl;
    }
  else
    {
      n_min_for_bct = 2;
      std::cout
        << "Minimum block cluster size/cardinality has been set to the default value: 2"
        << std::endl;
    }

  if (vm.count("adm"))
    {
      eta = vm["adm"].as<double>();
      std::cout << "Admissibility constant: " << eta << std::endl;
    }
  else
    {
      eta = 1.0;
      std::cout
        << "Admissibility constant eta has been set to the default value: 1.0"
        << std::endl;
    }

  if (vm.count("rank"))
    {
      max_hmat_rank = vm["rank"].as<unsigned int>();
      std::cout << "Maximum H-matrix rank: " << max_hmat_rank << std::endl;
    }
  else
    {
      max_hmat_rank = 2;
      std::cout
        << "Maximum rank for the H-matrix has been set to the default value: 2"
        << std::endl;
    }

  if (vm.count("epsilon"))
    {
      aca_relative_error = vm["epsilon"].as<double>();
      std::cout << "Relative error for ACA+: " << aca_relative_error
                << std::endl;
    }
  else
    {
      aca_relative_error = 1e-2;
      std::cout
        << "The relative error for ACA+ has been set to the default value: 1e-2"
        << std::endl;
    }

  IdeoBEM::Erichsen1996Efficient::Example2::ProblemType problem_type;

  switch (problem_type_string)
    {
      case 'n':
        {
          problem_type =
            IdeoBEM::Erichsen1996Efficient::Example2::NeumannBCProblem;

          break;
        }
      case 'd':
        {
          problem_type =
            IdeoBEM::Erichsen1996Efficient::Example2::DirichletBCProblem;

          break;
        }
      default:
        {
          Assert(false, ExcInternalError());

          problem_type =
            IdeoBEM::Erichsen1996Efficient::Example2::NeumannBCProblem;
          ;
        }
    }

  IdeoBEM::Erichsen1996Efficient::Example2 testcase(mesh_file_name,
                                                    fe_order,
                                                    problem_type,
                                                    thread_num,
                                                    n_min_for_ct,
                                                    n_min_for_bct,
                                                    eta,
                                                    max_hmat_rank,
                                                    aca_relative_error);

  testcase.run_using_hmat();

  return 0;
}
