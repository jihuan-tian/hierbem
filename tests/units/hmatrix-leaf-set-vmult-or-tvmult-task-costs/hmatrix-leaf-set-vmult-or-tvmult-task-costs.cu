// Copyright (C) 2024-2026 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file hmatrix-leaf-set-vmult-or-tvmult-task-costs.cu
 * @brief Verify estimating the task costs for \hmatrix/vector multiplication
 * for all \hmatrix nodes in the leaf set.
 *
 * @ingroup test_cases
 * @author Jihuan Tian
 * @date 2024-03-14
 */

#include <deal.II/base/multithread_info.h>
#include <deal.II/base/point.h>
#include <deal.II/base/table.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_dgq.h>

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <fstream>
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
#include "cluster_tree/block_cluster_tree.h"
#include "cluster_tree/cluster_tree.h"
#include "config_file/config_structs.h"
#include "config_file/cu_related.h"
#include "dofs/dof_to_cell_topology.h"
#include "dofs/dof_tools_ext.h"
#include "grid/grid_in_ext.h"
#include "hbem_test_config.h"
#include "hmatrix/aca_plus/aca_plus.hcu"
#include "hmatrix/hmatrix.h"
#include "hmatrix/hmatrix_support.h"
#include "linear_algebra/cu_table.hcu"
#include "mapping/mapping_info.h"
#include "platform_shared/laplace_kernels.h"
#include "quadrature/sauter_quadrature_tools.h"
#include "sequence_partition/sequence_partition.h"
#include "utilities/debug_tools.h"
#include "utilities/generic_functors.h"
#include "utilities/unary_template_arg_containers.h"

using namespace dealii;
using namespace HierBEM;

int
main()
{
  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  using SearchableMaterialIdContainer = std::set<EntityTag>;

  const double inter_distance = 8.0;

  /**
   * Surface-to-volume and volume-to-surface relationship.
   */
  SubdomainTopology<dim, spacedim> subdomain_topology;

  Triangulation<dim, spacedim> tria;
  std::ifstream mesh_in(HBEM_TEST_MODEL_DIR "two-spheres-fine.msh");
  read_msh(mesh_in, tria);
  mesh_in.close();
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
  manifold_id_to_mapping_order[0] = 2;
  manifold_id_to_mapping_order[1] = 2;

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

  PlatformShared::LaplaceKernel::SingleLayerKernel<spacedim>
    single_layer_kernel;

  // Parameters for building H-matrices.
  ConfHMatrix             hmat_params{32, 32, 0.8, 5, 0.01, false};
  ConfSauterQuadNearField sauter_quad_near_field_params;
  ConfSauterQuadFarField  sauter_quad_far_field_params;
  ConfParallelization     parallel_params;
  // Set TBB thread num to 1 to remove randomness in ACA.
  parallel_params.tbb_thread_num = 1;

  // Set TBB thread num.
  if (parallel_params.tbb_thread_num == -1)
    MultithreadInfo::set_thread_limit(MultithreadInfo::n_threads());
  else
    MultithreadInfo::set_thread_limit(parallel_params.tbb_thread_num);

  // Initialize CUDA stack size and device properties.
  initCudaRuntime(parallel_params);

  Table<2, Point<spacedim>>                   tria_mapping_support_points_cpu;
  CUDAWrappers::CUDATable<2, Point<spacedim>> tria_mapping_support_points_gpu;
  std::vector<unsigned int>                   tria_mapping_indices_cpu;
  CUDAWrappers::CUDATable<1, unsigned int>    tria_mapping_indices_gpu;

  BEMTools::compute_mapping_support_points_and_indices_for_tria(
    tria,
    mappings,
    material_id_to_mapping_index,
    tria_mapping_support_points_cpu,
    tria_mapping_indices_cpu);

  const types::global_cell_index n_cells = tria.n_active_cells();
  tria_mapping_support_points_gpu.allocate(
    TableIndices<2>(n_cells, mappings.back()->get_data()->n_shape_functions));
  tria_mapping_support_points_gpu.assign_from_host(
    tria_mapping_support_points_cpu);

  tria_mapping_indices_gpu.allocate(TableIndices<1>(n_cells));
  tria_mapping_indices_gpu.assign_from_host(tria_mapping_indices_cpu);

  dof_handler.distribute_dofs(fe);

  // Generate a list of cell iterators which will be used for constructing
  // the dof-to-cell topology.
  std::vector<typename DoFHandler<dim, spacedim>::cell_iterator> cell_iterators;
  for (const auto &cell : dof_handler.active_cell_iterators())
    cell_iterators.push_back(cell);

  DoFToCellTopology<dim, spacedim> dof_to_cell_topo;
  DoFToolsExt::build_dof_to_cell_topology(dof_to_cell_topo,
                                          cell_iterators,
                                          dof_handler);

  std::vector<const typename DoFHandler<dim, spacedim>::cell_iterator *>
    cell_iterator_ptrs;
  cell_iterator_ptrs.reserve(n_cells);
  for (auto &cell : cell_iterators)
    cell_iterator_ptrs.push_back(&cell);

  std::vector<types::global_cell_index> global_to_local_cell_index_map(n_cells);
  std::vector<types::global_cell_index> local_to_global_cell_index_map;
  gen_linear_indices<vector_uta, types::global_cell_index>(
    global_to_local_cell_index_map);
  local_to_global_cell_index_map = global_to_local_cell_index_map;

  // Generate lists of DoF indices.
  std::vector<types::global_dof_index> dof_indices(dof_handler.n_dofs());
  gen_linear_indices<vector_uta, types::global_dof_index>(dof_indices);
  // Get the spatial coordinates of the support points. Even though
  // different surfaces may be assigned a manifold which is further
  // associated with a high order mapping, here we only use the first order
  // mapping to generate the support points for finite element shape
  // functions. This is good enough for the partition of cluster trees.
  std::vector<Point<spacedim>> support_points(dof_handler.n_dofs());
  DoFTools::map_dofs_to_support_points(mappings[0]->get_mapping(),
                                       dof_handler,
                                       support_points);

  // Compute average cell size at each support points.
  std::vector<double> cell_size_at_support_points(dof_handler.n_dofs());
  cell_size_at_support_points.assign(dof_handler.n_dofs(), 0);
  DoFToolsExt::map_dofs_to_average_cell_size(dof_handler,
                                             cell_size_at_support_points);

  // Create and partition the cluster tree.
  ClusterTree<spacedim> ct(dof_indices,
                           support_points,
                           cell_size_at_support_points,
                           static_cast<unsigned int>(hmat_params.n_min_for_ct));
  ct.partition(support_points, cell_size_at_support_points);

  // Create and partition the block cluster tree.
  BlockClusterTree<spacedim> bct(ct,
                                 ct,
                                 hmat_params.eta,
                                 static_cast<unsigned int>(
                                   hmat_params.n_min_for_bct));
  bct.partition(ct.get_internal_to_external_dof_numbering(),
                support_points,
                cell_size_at_support_points);

  // Create a general \hmatrix with respect to the block cluster tree.
  HMatrix<spacedim> V(bct,
                      static_cast<unsigned int>(hmat_params.max_rank),
                      HMatrixSupport::Property::general,
                      HMatrixSupport::BlockType::diagonal_block);

  // Create a symmetric \hmatrix with respect to the block cluster tree.
  HMatrix<spacedim> V_symm(bct,
                           static_cast<unsigned int>(hmat_params.max_rank),
                           HMatrixSupport::Property::symmetric,
                           HMatrixSupport::BlockType::diagonal_block);

  // Estimate the storage before assembling the matrices. At this moment, the
  // rank values of all low rank matrices are set to @p max_rank.
  std::vector<double> V_near_field_set_storage(
    V.get_near_field_leaf_set().size());
  std::vector<double> V_far_field_set_storage(
    V.get_far_field_leaf_set().size());
  std::vector<double> V_symm_near_field_set_storage(
    V_symm.get_near_field_leaf_set().size());
  std::vector<double> V_symm_far_field_set_storage(
    V_symm.get_far_field_leaf_set().size());

  V.compute_near_field_leaf_set_assembly_task_costs(V_near_field_set_storage);
  V.compute_far_field_leaf_set_assembly_task_costs(V_far_field_set_storage);
  V_symm.compute_near_field_leaf_set_assembly_task_costs(
    V_symm_near_field_set_storage);
  V_symm.compute_far_field_leaf_set_assembly_task_costs(
    V_symm_far_field_set_storage);

  print_vector_to_mat(std::cout,
                      "V_near_field_set_storage_before_assembly",
                      V_near_field_set_storage);
  print_vector_to_mat(std::cout,
                      "V_far_field_set_storage_before_assembly",
                      V_far_field_set_storage);
  print_vector_to_mat(std::cout,
                      "V_symm_near_field_set_storage_before_assembly",
                      V_symm_near_field_set_storage);
  print_vector_to_mat(std::cout,
                      "V_symm_far_field_set_storage_before_assembly",
                      V_symm_far_field_set_storage);

  // Assemble the general \hmatrix using ACA.
  fill_hmatrix_with_aca_plus_smp<
    dim,
    spacedim,
    PlatformShared::LaplaceKernel::SingleLayerKernel,
    double,
    double,
    SurfaceNormalDetector<dim, spacedim>>(
    V,
    hmat_params,
    sauter_quad_near_field_params,
    sauter_quad_far_field_params,
    parallel_params,
    single_layer_kernel,
    1.0,
    dof_to_cell_topo,
    dof_to_cell_topo,
    SauterQuadratureRule<dim>(5, 4, 4, 3),
    dof_handler,
    dof_handler,
    nullptr,
    nullptr,
    ct.get_internal_to_external_dof_numbering(),
    ct.get_internal_to_external_dof_numbering(),
    mappings,
    material_id_to_mapping_index,
    global_to_local_cell_index_map,
    local_to_global_cell_index_map,
    cell_iterator_ptrs,
    tria_mapping_support_points_cpu,
    tria_mapping_support_points_gpu,
    tria_mapping_indices_gpu,
    SurfaceNormalDetector<dim, spacedim>(subdomain_topology),
    false);

  // Assemble the symmetric \hmatrix using ACA.
  fill_hmatrix_with_aca_plus_smp<
    dim,
    spacedim,
    HierBEM::PlatformShared::LaplaceKernel::SingleLayerKernel,
    double,
    double,
    SurfaceNormalDetector<dim, spacedim>>(
    V_symm,
    hmat_params,
    sauter_quad_near_field_params,
    sauter_quad_far_field_params,
    parallel_params,
    single_layer_kernel,
    1.0,
    dof_to_cell_topo,
    dof_to_cell_topo,
    SauterQuadratureRule<dim>(5, 4, 4, 3),
    dof_handler,
    dof_handler,
    nullptr,
    nullptr,
    ct.get_internal_to_external_dof_numbering(),
    ct.get_internal_to_external_dof_numbering(),
    mappings,
    material_id_to_mapping_index,
    global_to_local_cell_index_map,
    local_to_global_cell_index_map,
    cell_iterator_ptrs,
    tria_mapping_support_points_cpu,
    tria_mapping_support_points_gpu,
    tria_mapping_indices_gpu,
    SurfaceNormalDetector<dim, spacedim>(subdomain_topology),
    true);

  // Write out the leaf set information.
  std::ofstream bct_out("V-bct.dat");
  V.write_leaf_set_by_iteration(bct_out);
  bct_out.close();

  bct_out.open("V-symm-bct.dat");
  V_symm.write_leaf_set_by_iteration(bct_out);
  bct_out.close();

  tria_mapping_support_points_gpu.release();
  tria_mapping_indices_gpu.release();

  // Estimate the storage after assembling the matrices. Because ACA has been
  // adopted, the rank of each rank-k matrix block may now be different from
  // @p max_rank.
  V.compute_near_field_leaf_set_assembly_task_costs(V_near_field_set_storage);
  V.compute_far_field_leaf_set_assembly_task_costs(V_far_field_set_storage);
  V_symm.compute_near_field_leaf_set_assembly_task_costs(
    V_symm_near_field_set_storage);
  V_symm.compute_far_field_leaf_set_assembly_task_costs(
    V_symm_far_field_set_storage);

  print_vector_to_mat(std::cout,
                      "V_near_field_set_storage_after_assembly",
                      V_near_field_set_storage);
  print_vector_to_mat(std::cout,
                      "V_far_field_set_storage_after_assembly",
                      V_far_field_set_storage);
  print_vector_to_mat(std::cout,
                      "V_symm_near_field_set_storage_after_assembly",
                      V_symm_near_field_set_storage);
  print_vector_to_mat(std::cout,
                      "V_symm_far_field_set_storage_after_assembly",
                      V_symm_far_field_set_storage);

  // Estimate the \hmatrix/vector multiplication costs.
  std::vector<double> V_leaf_set_vmult_costs(V.get_leaf_set().size());
  std::vector<double> V_symm_leaf_set_vmult_costs(V_symm.get_leaf_set().size());

  V.compute_leaf_set_vmult_or_Tvmult_task_costs(V_leaf_set_vmult_costs);
  V_symm.compute_leaf_set_vmult_or_Tvmult_task_costs(
    V_symm_leaf_set_vmult_costs);

  print_vector_to_mat(std::cout,
                      "V_leaf_set_vmult_costs",
                      V_leaf_set_vmult_costs);
  print_vector_to_mat(std::cout,
                      "V_symm_leaf_set_vmult_costs",
                      V_symm_leaf_set_vmult_costs);

  // Generate the cost function for sequence partition.
  auto V_vmult_cost_func = [&V_leaf_set_vmult_costs](int i, int j) -> double {
    double sum = 0.0;
    for (int k = i; k <= j; k++)
      {
        sum += V_leaf_set_vmult_costs[k];
      }
    return sum;
  };

  const unsigned int                               thread_num = 8;
  SequencePartitioner<decltype(V_vmult_cost_func)> V_sp(
    V_leaf_set_vmult_costs.size(), thread_num, V_vmult_cost_func);
  V_sp.partition();

  double minmax_cost = V_sp.get_minmax_cost();
  std::cout << "Minimum maximum interval cost for V.vmult: " << minmax_cost
            << std::endl;

  std::vector<std::pair<int64_t, int64_t>> V_parts;
  V_sp.get_partitions(V_parts);

  for (const auto &part : V_parts)
    {
      double interval_cost = V_vmult_cost_func(part.first, part.second);
      std::cout << "[" << part.first << "," << part.second
                << "]: " << interval_cost << std::endl;
    }

  auto V_symm_vmult_cost_func =
    [&V_symm_leaf_set_vmult_costs](int i, int j) -> double {
    double sum = 0.0;
    for (int k = i; k <= j; k++)
      {
        sum += V_symm_leaf_set_vmult_costs[k];
      }
    return sum;
  };

  SequencePartitioner<decltype(V_symm_vmult_cost_func)> V_symm_sp(
    V_symm_leaf_set_vmult_costs.size(), thread_num, V_symm_vmult_cost_func);
  V_symm_sp.partition();

  minmax_cost = V_symm_sp.get_minmax_cost();
  std::cout << "Minimum maximum interval cost for V_symm.vmult: " << minmax_cost
            << std::endl;

  std::vector<std::pair<int64_t, int64_t>> V_symm_parts;
  V_symm_sp.get_partitions(V_symm_parts);

  for (const auto &part : V_symm_parts)
    {
      double interval_cost = V_symm_vmult_cost_func(part.first, part.second);
      std::cout << "[" << part.first << "," << part.second
                << "]: " << interval_cost << std::endl;
    }

  return 0;
}
