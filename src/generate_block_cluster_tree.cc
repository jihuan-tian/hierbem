/**
 * \file generate_block_cluster_tree.cc
 * \brief Generate the block cluster tree for a mesh and save the hierarchical
 * structure of the block cluster tree for visualization using @p dot in
 * Graphviz.
 *
 * \date 2022-03-16
 * \author Jihuan Tian
 */

#include <deal.II/base/logstream.h>
#include <deal.II/base/point.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

// Grid input and output
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>

#include <boost/program_options.hpp>

#include <fstream>
#include <regex>

#include "block_cluster_tree.h"
#include "cluster_tree.h"
#include "debug_tools.h"
#include "hmatrix/hmatrix.h"
#include "regex_tools.h"
#include "simple_bounding_box.h"
#include "unary_template_arg_containers.h"

using namespace dealii;
using namespace boost::program_options;

int
main(int argc, char *argv[])
{
  (void)argc;

  /**
   * Initialize deal.ii log stream.
   */
  deallog.pop();
  deallog.depth_console(2);

  std::string  mesh_file_name;
  std::string  bct_file_name;
  unsigned int fe_order;
  unsigned int n_min;
  double       eta;

  options_description opts("generate_block_cluster_tree options");
  opts.add_options()("help,h", "Display this help")("input,i",
                                                    value<std::string>(),
                                                    "Path to the mesh file")(
    "bct,b", value<std::string>(), "Output file for block cluster tree")(
    "fe_order,o", value<unsigned int>(), "Finite element order")(
    "nmin,n", value<unsigned int>(), "Minimum cluster size/cardinality")(
    "adm,a", value<double>(), "Admissibility constant");

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

  if (vm.count("bct"))
    {
      bct_file_name = vm["bct"].as<std::string>();
    }
  else
    {
      bct_file_name =
        RegexTools::file_basename(mesh_file_name) + std::string("-bct.puml");
      std::cout << "Block cluster tree will be saved into " << bct_file_name
                << std::endl;
    }

  if (vm.count("fe_order"))
    {
      fe_order = vm["fe_order"].as<unsigned int>();
    }
  else
    {
      fe_order = 2;
      std::cout << "Finite element order has been set to the default value: 2"
                << std::endl;
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

  const unsigned int           spacedim = 3;
  const unsigned int           dim      = 2;
  Triangulation<dim, spacedim> triangulation;

  /**
   * Read the mesh from a file.
   */
  GridIn<dim, spacedim> grid_in;
  grid_in.attach_triangulation(triangulation);
  std::fstream mesh_file(mesh_file_name);
  grid_in.read_msh(mesh_file);

  /**
   * Create the Lagrangian finite element.
   */
  FE_Q<dim, spacedim> fe(fe_order);

  /**
   * Create a DoFHandler, which is associated with the triangulation and
   * distributed with the finite element.
   */
  DoFHandler<dim, spacedim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  /**
   * Create the mapping object, which is required in generating the map from
   * DoF indices to support points.
   */
  const MappingQ<dim, spacedim> mapping(fe_order);

  /**
   * Generate a list of all DoF indices.
   */
  std::vector<types::global_dof_index> dof_indices(dof_handler.n_dofs());
  gen_linear_indices<vector_uta, types::global_dof_index>(dof_indices);

  /**
   * Get the spatial coordinates of the support points associated with DoF
   * indices.
   */
  std::vector<Point<spacedim>> all_support_points(dof_handler.n_dofs());
  DoFTools::map_dofs_to_support_points(mapping,
                                       dof_handler,
                                       all_support_points);


  /**
   * Calculate the average mesh cell size at each support point.
   */
  std::vector<double> dof_average_cell_size(dof_handler.n_dofs(), 0);
  map_dofs_to_average_cell_size(dof_handler, dof_average_cell_size);

  /**
   * Initialize the cluster tree \f$T(I)\f$ and \f$T(J)\f$ for all the DoF
   * indices.
   */
  ClusterTree<spacedim> TI(dof_indices,
                           all_support_points,
                           dof_average_cell_size,
                           n_min);

  /**
   * Partition the cluster tree.
   */
  TI.partition(all_support_points, dof_average_cell_size);

  /**
   * Create the block cluster tree.
   */
  BlockClusterTree<spacedim> block_cluster_tree(TI, TI, eta, n_min);

  /**
   * Perform admissible partition on the block cluster tree.
   */
  block_cluster_tree.partition(all_support_points);

  /**
   * Write the hierarchical structure of the \bct into a file for visualization
   * in @p PlantUML.
   */
  std::ofstream bct_digraph(bct_file_name);
  block_cluster_tree.print_bct_info_as_dot(bct_digraph);
  bct_digraph.close();
}
