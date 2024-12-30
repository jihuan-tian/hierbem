/**
 * @file hmatrix-leaf-set-vmult-or-tvmult-task-costs.cu
 * @brief Verify estimating the task costs for \hmatrix/vector multiplication
 * for all \hmatrix nodes in the leaf set.
 *
 * @ingroup testers
 * @author Jihuan Tian
 * @date 2024-03-14
 */

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_dgq.h>

#include <deal.II/grid/manifold_lib.h>

#include <cuda_runtime.h>

#include <iostream>

#include "hmatrix/aca_plus/aca_plus.hcu"
#include "grid_in_ext.h"
#include "hbem_test_config.h"
#include "laplace_bem.h"
#include "mapping/mapping_info.h"
#include "sauter_quadrature.hcu"
#include "sequence_partition/sequence_partition.h"
#include "subdomain_topology.h"
#include "unary_template_arg_containers.h"

using namespace dealii;
using namespace HierBEM;

int
main()
{
  /**
   * Set TBB thread number as 1 to remove ACA randomness caused by multiple
   * threads, even though the random engine is initialized with a definite
   * state.
   */
  dealii::MultithreadInfo::set_thread_limit(1);

  /**
   * Initialize the CUDA device parameters.
   */
  const size_t stack_size = 1024 * 10;
  AssertCuda(cudaDeviceSetLimit(cudaLimitStackSize, stack_size));
  std::cout << "CUDA stack size has been set to " << stack_size << std::endl;

  /**
   * Get GPU device properties.
   */
  AssertCuda(
    cudaGetDeviceProperties(&HierBEM::CUDAWrappers::device_properties, 0));

  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  const double inter_distance = 8.0;

  /**
   * Surface-to-volume and volume-to-surface relationship.
   */
  SubdomainTopology<dim, spacedim> subdomain_topology;

  Triangulation<dim, spacedim> tria;
  read_skeleton_mesh(HBEM_TEST_MODEL_DIR "two-spheres-fine.msh", tria);
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
    {
      mappings[i - 1] = new MappingInfo<dim, spacedim>(i);
    }

  // Construct the map from material ids to mapping indices.
  std::map<types::material_id, unsigned int> material_id_to_mapping_index;
  for (const auto &m : manifold_description)
    {
      material_id_to_mapping_index[m.first] =
        manifold_id_to_mapping_order[m.second] - 1;
    }

  FE_DGQ<dim, spacedim>     fe(0);
  DoFHandler<dim, spacedim> dof_handler(tria);

  HierBEM::PlatformShared::LaplaceKernel::SingleLayerKernel<spacedim>
    single_layer_kernel;

  const unsigned int n_min    = 32;
  const double       eta      = 0.8;
  const double       max_rank = 5;
  const double       epsilon  = 0.01;

  dof_handler.distribute_dofs(fe);

  // Generate a list of cell iterators which will be used for constructing
  // the dof-to-cell topology.
  std::vector<typename DoFHandler<dim, spacedim>::cell_iterator> cell_iterators;
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_iterators.push_back(cell);
    }

  DofToCellTopology<dim, spacedim> dof_to_cell_topo;
  build_dof_to_cell_topology(dof_to_cell_topo, cell_iterators, dof_handler);

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
                           n_min);
  ct.partition(support_points, cell_size_at_support_points);

  // Create and partition the block cluster tree.
  BlockClusterTree<spacedim> bct(ct, ct, eta, n_min);
  bct.partition(ct.get_internal_to_external_dof_numbering(),
                support_points,
                cell_size_at_support_points);

  // Create a general \hmatrix with respect to the block cluster tree.
  HMatrix<spacedim> V(bct,
                      max_rank,
                      HMatrixSupport::Property::general,
                      HMatrixSupport::BlockType::diagonal_block);

  // Create a symmetric \hmatrix with respect to the block cluster tree.
  HMatrix<spacedim> V_symm(bct,
                           max_rank,
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
  fill_hmatrix_with_aca_plus_smp(
    MultithreadInfo::n_threads(),
    V,
    ACAConfig(max_rank, epsilon, eta),
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
    LaplaceBEM<dim, spacedim>::SurfaceNormalDetector(subdomain_topology),
    false);

  // Assemble the symmetric \hmatrix using ACA.
  fill_hmatrix_with_aca_plus_smp(
    MultithreadInfo::n_threads(),
    V_symm,
    ACAConfig(max_rank, epsilon, eta),
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
    LaplaceBEM<dim, spacedim>::SurfaceNormalDetector(subdomain_topology),
    true);

  // Write out the leaf set information.
  std::ofstream bct_out("V-bct.dat");
  V.write_leaf_set_by_iteration(bct_out);
  bct_out.close();

  bct_out.open("V-symm-bct.dat");
  V_symm.write_leaf_set_by_iteration(bct_out);
  bct_out.close();

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
