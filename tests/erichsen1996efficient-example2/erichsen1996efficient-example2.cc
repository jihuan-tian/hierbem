/**
 * \file erichsen1996efficient-example2.cc
 * \brief Verify solving the Laplace problem with pure Neumann boundary
 * condition using BEM. The matrices are constructed as full matrices.
 *
 * \ingroup testers
 * \author Jihuan Tian
 * \date 2020-11-26
 */

#include <deal.II/base/logstream.h>

#include <boost/program_options.hpp>

#include <erichsen1996efficient_example2.h>

using namespace dealii;
using namespace boost::program_options;

int
main(int argc, char *argv[])
{
  deallog.depth_console(2);
  deallog.pop();

  std::string  mesh_file_name;
  unsigned int fe_order;
  unsigned int thread_num;
  char         problem_type_string;
  unsigned int n_min_for_ct       = 2;
  unsigned int n_min_for_bct      = 2;
  double       eta                = 1.0;
  unsigned int max_hmat_rank      = 4;
  double       aca_relative_error = 0.01;

  options_description opts("erichsen1996efficient-example2 options");
  opts.add_options()("help,h", "Display this help")("input,i",
                                                    value<std::string>(),
                                                    "Path to the mesh file")(
    "fe_order,o", value<unsigned int>(), "Finite element order")(
    "threads,t", value<unsigned int>(), "Number of threads")(
    "problem,p",
    value<char>(),
    "Problem type: 'd' for Dirichlet problem, 'n' for Neumann problem");

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
      thread_num = 4;
      std::cout << "Number of threads has been set to the default value: 4"
                << std::endl;
    }

  if (vm.count("problem"))
    {
      problem_type_string = vm["problem"].as<char>();
    }
  else
    {
      problem_type_string = 'n';
    }

  HierBEM::Erichsen1996Efficient::Example2::ProblemType problem_type;

  switch (problem_type_string)
    {
      case 'n':
        {
          problem_type =
            HierBEM::Erichsen1996Efficient::Example2::NeumannBCProblem;

          break;
        }
      case 'd':
        {
          problem_type =
            HierBEM::Erichsen1996Efficient::Example2::DirichletBCProblem;

          break;
        }
      default:
        {
          Assert(false, ExcInternalError());

          problem_type =
            HierBEM::Erichsen1996Efficient::Example2::NeumannBCProblem;
          ;
        }
    }

  HierBEM::Erichsen1996Efficient::Example2 testcase(mesh_file_name,
                                                    fe_order,
                                                    problem_type,
                                                    thread_num,
                                                    n_min_for_ct,
                                                    n_min_for_bct,
                                                    eta,
                                                    max_hmat_rank,
                                                    aca_relative_error);
  testcase.run();

  return 0;
}
