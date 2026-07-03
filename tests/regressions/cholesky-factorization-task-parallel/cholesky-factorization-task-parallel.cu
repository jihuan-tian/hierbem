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
 * @file cholesky-factorization-task-parallel.cu
 * @brief Repeatedly run H-Cholesky factorization.
 * Usage: ./cholesky-factorization-task-parallel-repeated <repeat-num>
 *
 * @ingroup test_cases
 * @author
 * @date 2024-01-10
 */

#include <deal.II/base/multithread_info.h>
#include <deal.II/base/point.h>
#include <deal.II/base/table.h>
#include <deal.II/base/types.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_dgq.h>

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <cstdlib> // atoi
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
#include "utilities/debug_tools.h"
#include "utilities/unary_template_arg_containers.h"

using namespace dealii;
using namespace HierBEM;

int
main(int argc, char *argv[])
{
  // Suppress compilation errors
  (void)argc;

  // Repetitions to run H-Cholesky.
  const unsigned int REPEATS = atoi(argv[1]);

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

  // Parameters for building H-matrices.
  ConfHMatrix             hmat_params{32, 32, 0.8, 5, 0.01, false};
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

  // Refine the volume mesh.
  tria.refine_global(1);

  Table<2, Point<spacedim>> tria_mapping_support_points_cpu;
  HierBEM::CUDAWrappers::CUDATable<2, Point<spacedim>>
                            tria_mapping_support_points_gpu;
  std::vector<unsigned int> tria_mapping_indices_cpu;
  HierBEM::CUDAWrappers::CUDATable<1, unsigned int> tria_mapping_indices_gpu;

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

  BEMFunctionSpace<dim, spacedim, SearchableMaterialIdContainer, double>
    H_minus_half(dof_handler,
                 static_cast<unsigned int>(hmat_params.n_min_for_ct));
  BEMBilinearForm<dim,
                  spacedim,
                  SearchableMaterialIdContainer,
                  HierBEM::PlatformShared::LaplaceKernel::SingleLayerKernel,
                  double,
                  double>
    bV(H_minus_half, H_minus_half);

  Timer timer;
  bV.build_block_cluster_tree(
    hmat_params.eta, static_cast<unsigned int>(hmat_params.n_min_for_bct));
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
  timer.stop();
  print_wall_time(std::cout, timer, "assemble H-matrix V");

  for (unsigned int i = 0; i < REPEATS; i++)
    {
      // Make a copy of the SLP matrix.
      HMatrix<spacedim, double> V_tmp(*V);

      // Perform Cholesky factorization.
      timer.start();
      V_tmp.compute_cholesky_factorization_task_parallel(
        static_cast<unsigned int>(hmat_params.max_rank));
      timer.stop();
      print_wall_time(std::cout, timer, "cholesky factorization");
    }

  tria_mapping_support_points_gpu.release();
  tria_mapping_indices_gpu.release();

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
