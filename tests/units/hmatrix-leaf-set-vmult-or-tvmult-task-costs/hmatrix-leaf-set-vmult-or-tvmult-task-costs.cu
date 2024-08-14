/**
 * @file hmatrix-leaf-set-vmult-or-tvmult-task-costs.cu
 * @brief Verify estimating the task costs for \hmatrix/vector multiplication
 * for all \hmatrix nodes in the leaf set.
 *
 * @ingroup testers
 * @author Jihuan Tian
 * @date 2024-03-14
 */

#include <cuda_runtime.h>

#include <iostream>

#include "debug_tools.hcu"
#include "laplace_bem.h"
#include "sequence_partition/sequence_partition.h"

using namespace dealii;
using namespace HierBEM;

template <int spacedim>
void
refine_boundary_mesh_for_two_spheres(Triangulation<spacedim> &tria,
                                     const double             inter_distance,
                                     const double             radius,
                                     const double             rtol = 1e-6)
{
  const auto left_ball_center  = Point<spacedim>(-inter_distance / 2.0, 0, 0);
  const auto right_ball_center = Point<spacedim>(inter_distance / 2.0, 0, 0);

  // Mark the cells near the boundary of two spheres and refine them only
  for (const auto &cell : tria.active_cell_iterators())
    {
      for (const auto &v : cell->vertex_indices())
        {
          const double distance_to_left_ball =
            left_ball_center.distance(cell->vertex(v));
          const double distance_to_right_ball =
            right_ball_center.distance(cell->vertex(v));
          if (fabs(distance_to_left_ball - radius) <= rtol * radius ||
              fabs(distance_to_right_ball - radius) <= rtol * radius)
            {
              cell->set_refine_flag();
              break;
            }
        }
    }
  tria.execute_coarsening_and_refinement();
}


template <int spacedim>
void
generate_coarse_mesh_for_two_spheres(Triangulation<spacedim> &tria,
                                     const double             inter_distance,
                                     const double             radius)
{
  Triangulation<spacedim> left_ball, right_ball;

  GridGenerator::hyper_ball(left_ball,
                            Point<spacedim>(-inter_distance / 2.0, 0, 0),
                            radius);
  GridGenerator::hyper_ball(right_ball,
                            Point<spacedim>(inter_distance / 2.0, 0, 0),
                            radius);

  /**
   * @internal Set different manifold ids and material ids to all the cells
   * in the two balls.
   */
  for (typename Triangulation<spacedim>::active_cell_iterator cell =
         left_ball.begin_active();
       cell != left_ball.end();
       cell++)
    {
      cell->set_all_manifold_ids(0);
      cell->set_material_id(0);
    }

  for (typename Triangulation<spacedim>::active_cell_iterator cell =
         right_ball.begin_active();
       cell != right_ball.end();
       cell++)
    {
      cell->set_all_manifold_ids(1);
      cell->set_material_id(1);
    }

  /**
   * @internal @p merge_triangulation can only operate on coarse mesh, i.e.
   * triangulations not refined. During the merging, the material ids are
   * copied. When the last argument is true, the manifold ids are copied.
   * Boundary ids will not be copied.
   */
  GridGenerator::merge_triangulations(left_ball, right_ball, tria, 1e-12, true);

  /**
   * @internal Assign manifold objects to the two balls in the merged mesh.
   */
  const SphericalManifold<spacedim> left_ball_manifold(
    Point<spacedim>(-inter_distance / 2.0, 0, 0));
  const SphericalManifold<spacedim> right_ball_manifold(
    Point<spacedim>(inter_distance / 2.0, 0, 0));

  tria.set_manifold(0, left_ball_manifold);
  tria.set_manifold(1, right_ball_manifold);
}


template <int dim, int spacedim>
void
extract_surface_mesh_for_two_spheres(
  Triangulation<spacedim>      &tria,
  Triangulation<dim, spacedim> &surface_tria,
  const double                  inter_distance,
  std::map<typename Triangulation<dim, spacedim>::cell_iterator,
           typename Triangulation<spacedim>::face_iterator>
    &map_from_surface_mesh_to_volume_mesh)
{
  // Generate an empty surface triangulation object with manifold ids
  // configured.
  const SphericalManifold<dim, spacedim> left_ball_surface_manifold(
    Point<spacedim>(-inter_distance / 2.0, 0, 0));
  const SphericalManifold<dim, spacedim> right_ball_surface_manifold(
    Point<spacedim>(inter_distance / 2.0, 0, 0));

  surface_tria.set_manifold(0, left_ball_surface_manifold);
  surface_tria.set_manifold(1, right_ball_surface_manifold);

  map_from_surface_mesh_to_volume_mesh =
    GridGenerator::extract_boundary_mesh(tria, surface_tria);
}


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
  const double radius         = 1.0;

  Triangulation<spacedim>      tria;
  Triangulation<dim, spacedim> surface_tria;
  generate_coarse_mesh_for_two_spheres(tria, inter_distance, radius);

  FE_DGQ<dim, spacedim>     fe(0);
  DoFHandler<dim, spacedim> dof_handler(surface_tria);

  // Define mapping objects and their internal data.
  MappingQGenericExt<dim, spacedim> kx_mapping(1);
  MappingQGenericExt<dim, spacedim> ky_mapping(1);

  std::unique_ptr<typename Mapping<dim, spacedim>::InternalDataBase>
    kx_mapping_database = kx_mapping.get_data(update_default, QGauss<dim>(1));
  std::unique_ptr<typename Mapping<dim, spacedim>::InternalDataBase>
    ky_mapping_database = ky_mapping.get_data(update_default, QGauss<dim>(1));
  std::unique_ptr<typename MappingQGeneric<dim, spacedim>::InternalData>
    kx_mapping_data =
      std::unique_ptr<typename MappingQGeneric<dim, spacedim>::InternalData>(
        static_cast<typename MappingQGeneric<dim, spacedim>::InternalData *>(
          kx_mapping_database.release()));
  std::unique_ptr<typename MappingQGeneric<dim, spacedim>::InternalData>
    ky_mapping_data =
      std::unique_ptr<typename MappingQGeneric<dim, spacedim>::InternalData>(
        static_cast<typename MappingQGeneric<dim, spacedim>::InternalData *>(
          ky_mapping_database.release()));

  HierBEM::CUDAWrappers::LaplaceKernel::SingleLayerKernel<spacedim>
    single_layer_kernel;

  const unsigned int n_min    = 32;
  const double       eta      = 0.8;
  const double       max_rank = 5;
  const double       epsilon  = 0.01;

  // Refine the volume mesh.
  for (unsigned int i = 0; i < 4; i++)
    {
      refine_boundary_mesh_for_two_spheres(tria, inter_distance, radius, 1e-6);
    }

  // Generate surface mesh from the volume mesh.
  std::map<typename Triangulation<dim, spacedim>::cell_iterator,
           typename Triangulation<spacedim>::face_iterator>
    map_from_surface_mesh_to_volume_mesh;
  extract_surface_mesh_for_two_spheres(tria,
                                       surface_tria,
                                       inter_distance,
                                       map_from_surface_mesh_to_volume_mesh);

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
  // Get the spatial coordinates of the support points.
  std::vector<Point<spacedim>> support_points(dof_handler.n_dofs());
  DoFTools::map_dofs_to_support_points(kx_mapping, dof_handler, support_points);

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
  fill_hmatrix_with_aca_plus_smp(MultithreadInfo::n_threads(),
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
                                 kx_mapping,
                                 ky_mapping,
                                 *kx_mapping_data,
                                 *ky_mapping_data,
                                 false);

  // Assemble the symmetric \hmatrix using ACA.
  fill_hmatrix_with_aca_plus_smp(MultithreadInfo::n_threads(),
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
                                 kx_mapping,
                                 ky_mapping,
                                 *kx_mapping_data,
                                 *ky_mapping_data,
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
