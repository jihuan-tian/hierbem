/**
 * \file hmatrix-build-from-aca.cc
 * \brief Build \hmatrices for SLP and DLP kernels based on the mesh data from
 * @p Erichsen1996Efficient Example 2.
 *
 * \ingroup testers hierarchical_matrices
 * \author Jihuan Tian
 * \date 2022-03-22
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

  std::string  slp_full_matrix_var_name;
  std::string  dlp_full_matrix_var_name;
  std::string  mesh_file_name;
  unsigned int fe_order;
  unsigned int thread_num;
  unsigned int n_min_for_ct;
  unsigned int n_min_for_bct;
  double       eta;
  unsigned int max_hmat_rank;
  double       aca_relative_error;
  bool         enable_build_symmetric_hmat;

  options_description opts("hmatrix-build-from-aca options");
  opts.add_options()("help,h", "Display this help")(
    "var_slp,s", value<std::string>(), "Variable name for the SLP full matrix")(
    "var_dlp,d", value<std::string>(), "Variable name for the DLP full matrix")(
    "input,i", value<std::string>(), "Path to the mesh file")(
    "fe_order,o", value<unsigned int>(), "Finite element order")(
    "threads,t", value<unsigned int>(), "Number of threads")(
    "nmin_ct,n", value<unsigned int>(), "Minimum cluster size/cardinality")(
    "nmin_bct,N",
    value<unsigned int>(),
    "Minimum block cluster size/cardinality")(
    "adm,a", value<double>(), "Admissibility condition constant")(
    "rank,r", value<unsigned int>(), "Maximum rank for the H-matrix")(
    "epsilon,e",
    value<double>(),
    "ACA+ relative error")("symm", "Build symmetric H-matrix");

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

  if (vm.count("var_slp"))
    {
      slp_full_matrix_var_name = vm["var_slp"].as<std::string>();
      std::cout << "SLP full matrix variable name: " << slp_full_matrix_var_name
                << std::endl;
    }
  else
    {
      std::cout << "Please provide the variable name for the SLP full matrix!"
                << std::endl;
      return 0;
    }

  if (vm.count("var_dlp"))
    {
      dlp_full_matrix_var_name = vm["var_dlp"].as<std::string>();
      std::cout << "DLP full matrix variable name: " << dlp_full_matrix_var_name
                << std::endl;
    }
  else
    {
      std::cout << "Please provide the variable name for the DLP full matrix!"
                << std::endl;
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

  if (vm.count("symm"))
    {
      enable_build_symmetric_hmat = true;
    }
  else
    {
      enable_build_symmetric_hmat = false;
    }
  std::cout << "Is symmetric H-matrix enabled: "
            << (enable_build_symmetric_hmat ? "true" : "false") << std::endl;

  HierBEM::Erichsen1996Efficient::Example2 testcase(mesh_file_name,
                                                    fe_order,
                                                    thread_num,
                                                    n_min_for_ct,
                                                    n_min_for_bct,
                                                    eta,
                                                    max_hmat_rank,
                                                    aca_relative_error);

  testcase.read_mesh();
  testcase.setup_system();
  testcase.assemble_system_as_hmatrices(enable_build_symmetric_hmat);

  /**
   * Convert the assembled \hmatrices to full matrices and write the data into
   * file.
   */
  LAPACKFullMatrixExt<double> slp_full_matrix, dlp_full_matrix;

  testcase.get_slp_hmat().convertToFullMatrix(slp_full_matrix);
  testcase.get_dlp_hmat().convertToFullMatrix(dlp_full_matrix);

  slp_full_matrix.print_formatted_to_mat(
    std::cout, slp_full_matrix_var_name, 15, false, 25, "0");
  dlp_full_matrix.print_formatted_to_mat(
    std::cout, dlp_full_matrix_var_name, 15, false, 25, "0");

  return 0;
}
