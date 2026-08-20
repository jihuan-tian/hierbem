// Copyright (C) 2023-2026 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file hmatrix-related-memory-consumption.cu
 * @brief Estimate the memory consumption of H-matrix, cluster tree and block
 * cluster tree, etc.
 *
 * @ingroup test_cases
 * @author Jihuan Tian
 * @date 2024-02-20
 */

#include <deal.II/base/memory_consumption.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/point.h>
#include <deal.II/base/table.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/types.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_dgq.h>

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <boost/program_options.hpp>

#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "bem/bem_bilinear_form.h"
#include "bem/bem_function_space.h"
#include "bem/bem_tools.h"
#include "cad_mesh/gmsh_manipulation.h"
#include "cad_mesh/subdomain_topology.h"
#include "config_file/config_structs.h"
#include "config_file/cu_related.h"
#include "grid/grid_in_ext.h"
#include "hbem_test_config.h"
#include "linear_algebra/cu_table.hcu"
#include "mapping/mapping_info.h"
#include "platform_shared/laplace_kernels.h"
#include "quadrature/sauter_quadrature_tools.h"
#include "utilities/unary_template_arg_containers.h"

using namespace dealii;
using namespace HierBEM;
namespace po = boost::program_options;

struct CmdOpts
{
  unsigned int mapping_order;
  unsigned int refinement;
  unsigned int n_min;
  double       eta;
  unsigned int initial_rank;
  unsigned int max_rank;
  double       epsilon;
};

CmdOpts
parse_cmdline(int argc, char *argv[])
{
  CmdOpts                 opts;
  po::options_description desc("Allowed options");

  // clang-format off
  desc.add_options()
    ("help,h", "show help message")
    ("mapping-order,o", po::value<unsigned int>()->default_value(2), "Mapping order for the sphere")
    ("refinement,r", po::value<unsigned int>()->default_value(5), "Number of global mesh refinement")
    ("n-min,n", po::value<unsigned int>()->default_value(32), "n_min criteria for small cluster")
    ("eta,e", po::value<double>()->default_value(0.8), "Admissibility constant eta")
    ("initial-rank,i", po::value<unsigned int>()->default_value(5), "Initial rank for a rank-k matrix")
    ("max-rank,m", po::value<unsigned int>()->default_value(0), "Max rank for a rank-k matrix; when it is 0, no limit")
    ("epsilon,E", po::value<double>()->default_value(0.01), "Relative error of ACA for building a rank-k matrix");
  // clang-format on

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help"))
    {
      std::cout << desc << std::endl;
      std::exit(EXIT_SUCCESS);
    }

  opts.mapping_order = vm["mapping-order"].as<unsigned int>();
  opts.refinement    = vm["refinement"].as<unsigned int>();
  opts.n_min         = vm["n-min"].as<unsigned int>();
  opts.eta           = vm["eta"].as<double>();
  opts.initial_rank  = vm["initial-rank"].as<unsigned int>();
  opts.max_rank      = vm["max-rank"].as<unsigned int>();
  opts.epsilon       = vm["epsilon"].as<double>();

  return opts;
}

int
main(int argc, char *argv[])
{
  CmdOpts opts = parse_cmdline(argc, argv);

  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  using SearchableMaterialIdContainer = std::set<EntityTag>;

  const double inter_distance = 8.0;

  /**
   * Surface-to-volume and volume-to-surface relationship.
   */
  SubdomainTopology<dim, spacedim> subdomain_topology;

  Triangulation<dim, spacedim> tria;
  read_msh(HBEM_TEST_MODEL_DIR "two-spheres-fine.msh", tria);
  subdomain_topology.generate_topology(HBEM_TEST_MODEL_DIR "two-spheres.brep",
                                       HBEM_TEST_MODEL_DIR "two-spheres.msh");

  // Define manifolds
  std::map<types::manifold_id, Manifold<dim, spacedim> *> manifolds;
  Manifold<dim, spacedim>                                *left_sphere_manifold =
    new SphericalManifold<dim, spacedim>(
      Point<spacedim>(-inter_distance / 2.0, 0, 0));
  Manifold<dim, spacedim> *right_sphere_manifold =
    new SphericalManifold<dim, spacedim>(
      Point<spacedim>(inter_distance / 2.0, 0, 0));
  manifolds[0] = left_sphere_manifold;
  manifolds[1] = right_sphere_manifold;

  // Define the mapping order adopted for each manifold.
  std::map<types::manifold_id, unsigned int> manifold_id_to_mapping_order;
  manifold_id_to_mapping_order[0] = opts.mapping_order;
  manifold_id_to_mapping_order[1] = opts.mapping_order;

  // Assign manifolds to surfaces.
  std::map<EntityTag, types::manifold_id> manifold_description;
  manifold_description[1] = 0;
  manifold_description[2] = 1;

  // Define mappings of different orders.
  std::vector<MappingInfo<dim, spacedim> *> mappings(3);
  for (unsigned int i = 1; i <= 3; i++)
    mappings[i - 1] = new MappingInfo<dim, spacedim>(i);

  // Construct the map from material ids to mapping indices.
  std::map<types::material_id, unsigned int> material_id_to_mapping_index;
  for (const auto &m : manifold_description)
    {
      material_id_to_mapping_index[m.first] =
        manifold_id_to_mapping_order[m.second] - 1;
    }

  FE_DGQ<dim, spacedim>     fe(0);
  DoFHandler<dim, spacedim> dof_handler(tria);

  HierBEM::PlatformShared::LaplaceKernel::SingleLayerKernel<spacedim, double>
    single_layer_kernel;

  // Parameters for building H-matrices.
  ConfHMatrix             hmat_params{opts.n_min,
                          opts.n_min,
                          1,
                          1,
                          opts.eta,
                          opts.initial_rank,
                          opts.max_rank,
                          opts.epsilon,
                          false,
                          10};
  ConfSauterQuadNearField sauter_quad_near_field_params;
  ConfSauterQuadFarField  sauter_quad_far_field_params;
  ConfParallelization     parallel_params;

  // Set TBB thread num.
  if (parallel_params.tbb_thread_num == -1)
    MultithreadInfo::set_thread_limit(MultithreadInfo::n_threads());
  else
    MultithreadInfo::set_thread_limit(parallel_params.tbb_thread_num);

  // Initialize CUDA stack size and device properties.
  initCudaRuntime(parallel_params);

  TableHandler table;
  for (unsigned int i = 0; i <= opts.refinement; i++)
    {
      table.add_value("Refinement", i);
      table.add_value("Object", "Surface triangulation");
      table.add_value("Memory", tria.memory_consumption());

      Table<2, Point<spacedim>> tria_mapping_support_points_cpu;
      HierBEM::CUDAWrappers::CUDATable<2, Point<spacedim>>
                                tria_mapping_support_points_gpu;
      std::vector<unsigned int> tria_mapping_indices_cpu;
      HierBEM::CUDAWrappers::CUDATable<1, unsigned int>
        tria_mapping_indices_gpu;

      BEMTools::compute_mapping_support_points_and_indices_for_tria(
        tria,
        mappings,
        material_id_to_mapping_index,
        tria_mapping_support_points_cpu,
        tria_mapping_indices_cpu);

      const types::global_cell_index n_cells = tria.n_active_cells();
      tria_mapping_support_points_gpu.allocate(
        TableIndices<2>(n_cells,
                        mappings.back()->get_data()->n_shape_functions));
      tria_mapping_support_points_gpu.assign_from_host(
        tria_mapping_support_points_cpu);

      tria_mapping_indices_gpu.allocate(TableIndices<1>(n_cells));
      tria_mapping_indices_gpu.assign_from_host(tria_mapping_indices_cpu);

      dof_handler.distribute_dofs(fe);

      BEMFunctionSpace<dim, spacedim, SearchableMaterialIdContainer, double>
        H_minus_half(dof_handler,
                     static_cast<unsigned int>(hmat_params.n_min_for_ct),
                     static_cast<unsigned int>(hmat_params.cutoff_level_ct));
      BEMBilinearForm<dim,
                      spacedim,
                      SearchableMaterialIdContainer,
                      HierBEM::PlatformShared::LaplaceKernel::SingleLayerKernel,
                      double,
                      double>
        bV(H_minus_half, H_minus_half);

      bV.build_block_cluster_tree(
        hmat_params.eta,
        static_cast<unsigned int>(hmat_params.n_min_for_bct),
        static_cast<unsigned int>(hmat_params.cutoff_level_bct));
      std::unique_ptr<HMatrix<spacedim, double>> V =
        bV.build_hmatrix(hmat_params,
                         sauter_quad_near_field_params,
                         sauter_quad_far_field_params,
                         parallel_params,
                         1.0,
                         SauterQuadratureRule<dim>(5, 4, 4, 3),
                         mappings,
                         material_id_to_mapping_index,
                         tria_mapping_support_points_cpu,
                         tria_mapping_support_points_gpu,
                         tria_mapping_indices_gpu,
                         subdomain_topology);

      table.start_new_row();
      table.add_value("Refinement", i);
      table.add_value("Object", "Cluster tree");
      table.add_value("Memory",
                      H_minus_half.get_cluster_tree().memory_consumption());

      table.start_new_row();
      table.add_value("Refinement", i);
      table.add_value("Object", "Block cluster tree");
      table.add_value("Memory",
                      bV.get_block_cluster_tree().memory_consumption());

      table.start_new_row();
      table.add_value("Refinement", i);
      table.add_value("Object", "SLP matrix V");
      table.add_value("Memory", V->memory_consumption());

      if (i < opts.refinement)
        // Refine the mesh.
        tria.refine_global(1);

      tria_mapping_support_points_gpu.release();
      tria_mapping_indices_gpu.release();
    }

  table.write_text(std::cout, TableHandler::TextOutputFormat::org_mode_table);

  // Delete manifolds and mappings.
  for (auto &m : manifolds)
    {
      if (m.second != nullptr)
        delete m.second;
    }

  for (auto &m : mappings)
    {
      if (m != nullptr)
        delete m;
    }

  return 0;
}
