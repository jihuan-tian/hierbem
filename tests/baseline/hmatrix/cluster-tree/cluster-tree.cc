// Copyright (C) 2026 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file cluster-tree.cc
 * @brief Evaluate the performance of building cluster trees.
 *
 * @author Jihuan Tian
 * @date 2026-07-19
 */

#include <deal.II/base/point.h>
#include <deal.II/base/timer.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <boost/program_options.hpp>

#include <cluster_tree/simple_bounding_box.h>

#include <fstream>
#include <iostream>
#include <string>

#include "cluster_tree/cluster_tree.h"
#include "hbem_cpp_validate.h"
#include "utilities/debug_tools.h"
#include "utilities/generic_functors.h"
#include "utilities/unary_template_arg_containers.h"

using namespace HierBEM;
using namespace dealii;
namespace po = boost::program_options;

struct CmdOpts
{
  std::string  outfile;
  unsigned int mapping_order;
  unsigned int refinement;
  unsigned int n_min;
  bool         enable_parallel;
  unsigned int cutoff_level;
};

CmdOpts
parse_cmdline(int argc, char *argv[])
{
  CmdOpts                 opts;
  po::options_description desc("Allowed options");

  // clang-format off
  desc.add_options()
    ("help,h", "show help message")
    ("output,o", po::value<std::string>()->default_value("cluster-tree.log"), "Output file name")
    ("mapping-order,m", po::value<unsigned int>()->default_value(2), "Mapping order for the mesh")
    ("refinement,r", po::value<unsigned int>()->default_value(3), "Number of global mesh refinement")
    ("n-min,n", po::value<unsigned int>()->default_value(4), "n_min criteria for small cluster")
    ("enable-parallel,p", po::bool_switch(&opts.enable_parallel), "Enable parallel tree building")
    ("cutoff-level,c", po::value<unsigned int>()->default_value(8), "Cutoff level for building cluster tree in parallel");
  // clang-format on

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help"))
    {
      std::cout << desc << std::endl;
      std::exit(EXIT_SUCCESS);
    }

  opts.outfile       = vm["output"].as<std::string>();
  opts.mapping_order = vm["mapping-order"].as<unsigned int>();
  opts.refinement    = vm["refinement"].as<unsigned int>();
  opts.n_min         = vm["n-min"].as<unsigned int>();
  opts.cutoff_level  = vm["cutoff-level"].as<unsigned int>();

  return opts;
}

int
main(int argc, char *argv[])
{
  CmdOpts       opts = parse_cmdline(argc, argv);
  std::ofstream ofs(opts.outfile);

  // Generate the grid for a 3D sphere.
  const unsigned int      dim = 3;
  Triangulation<dim, dim> triangulation;
  // N.B. Use type cast for triangulation to suppress Eclipse editor error
  // prompt.
  GridGenerator::hyper_ball((Triangulation<dim> &)triangulation,
                            Point<3>(0., 0., 0.),
                            2.0,
                            true);
  triangulation.refine_global(opts.refinement);

  // Create a Lagrangian finite element.
  FE_Q<dim, dim> fe(1);

  // Create a DoFHandler, which is associated with the triangulation and
  // distributed with the finite element.
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  ofs << "Number of DoFs: " << dof_handler.n_dofs() << std::endl;

  // Create a mapping object, which is required in generating the map from
  // DoF indices to support points.
  const MappingQ<dim, dim> mapping(opts.mapping_order);

  // Generate a list of all DoF indices.
  std::vector<types::global_dof_index> dof_indices(dof_handler.n_dofs());
  gen_linear_indices<vector_uta, types::global_dof_index>(dof_indices);

  // Get the spatial coordinates of the support points associated with DoF
  // indices.
  std::vector<Point<dim>> all_support_points(dof_handler.n_dofs());
  DoFTools::map_dofs_to_support_points(mapping,
                                       dof_handler,
                                       all_support_points);

  Timer            timer;
  ClusterTree<dim> cluster_tree(dof_indices, all_support_points, opts.n_min);
  timer.stop();
  print_wall_time(ofs, timer, "create root node");

  timer.start();
  if (opts.enable_parallel)
    cluster_tree.partition(all_support_points, opts.cutoff_level);
  else
    cluster_tree.partition(all_support_points);
  timer.stop();
  print_wall_time(ofs, timer, "partition cluster tree");

  ofs << "=== Support point coordinates ===\n";
  for (auto &point : all_support_points)
    ofs << point << "\n";

  ofs << "=== Cluster tree ===\n";
  ofs << cluster_tree << std::endl;

  ofs << "Memory consumption of all clusters: "
      << cluster_tree.memory_consumption_of_all_clusters() << "\n";
  ofs << "Memory consumption: " << cluster_tree.memory_consumption()
      << std::endl;
  ofs.close();

  // Export the cluster tree as a directional graph.
  std::ofstream graph("cluster-tree.puml");
  cluster_tree.print_tree_info_as_dot(graph);
  graph.close();

  return 0;
}
