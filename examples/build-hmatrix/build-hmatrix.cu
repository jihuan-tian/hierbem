// Copyright (C) 2025-2026 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file build-hmatrix.cu
 * @brief Example for building an H-matrix.
 *
 * @ingroup examples
 * @author Jihuan Tian
 * @date 2025-10-23
 */

#include <deal.II/base/exceptions.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/table.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/manifold.h>
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
#include "utilities/number_traits.h"
#include "utilities/unary_template_arg_containers.h"

using namespace dealii;
using namespace HierBEM;
using namespace HierBEM::PlatformShared::LaplaceKernel;

int
main()
{
  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  using SearchableMaterialIdContainer = std::set<EntityTag>;

  // Read the triangulation.
  Triangulation<dim, spacedim> tria;
  std::ifstream mesh_in(HBEM_TEST_MODEL_DIR "two-spheres-fine.msh");
  read_msh(mesh_in, tria);
  // Generate surface-to-volume and volume-to-surface topology.
  SubdomainTopology<dim, spacedim> subdomain_topology;
  subdomain_topology.generate_topology(HBEM_TEST_MODEL_DIR "two-spheres.brep",
                                       HBEM_TEST_MODEL_DIR "two-spheres.msh");

  // Define manifolds for the two spheres.
  const double                                            inter_distance = 8.0;
  std::map<types::manifold_id, Manifold<dim, spacedim> *> manifolds;
  Manifold<dim, spacedim>                                *left_sphere_manifold =
    new SphericalManifold<dim, spacedim>(
      Point<spacedim>(-inter_distance / 2.0, 0, 0));
  Manifold<dim, spacedim> *right_sphere_manifold =
    new SphericalManifold<dim, spacedim>(
      Point<spacedim>(inter_distance / 2.0, 0, 0));
  manifolds[0] = left_sphere_manifold;
  manifolds[1] = right_sphere_manifold;

  // Assign manifold ids to surface entities in the CAD model.
  std::map<EntityTag, types::manifold_id> manifold_description;
  manifold_description[1] = 0;
  manifold_description[2] = 1;

  // Assign manifolds to the triangulation.
  for (auto &cell : tria.active_cell_iterators())
    cell->set_all_manifold_ids(manifold_description[cell->material_id()]);

  for (const auto &m : manifolds)
    tria.set_manifold(m.first, *m.second);

  // Define mappings up to the second order for describing the curved surface.
  std::vector<MappingInfo<dim, spacedim> *> mappings(2);
  for (unsigned int i = 1; i <= 2; i++)
    mappings[i - 1] = new MappingInfo<dim, spacedim>(i);

  // Construct a map from material ids to mapping indices.
  std::map<types::material_id, unsigned int> material_id_to_mapping_index;
  material_id_to_mapping_index[1] = 1;
  material_id_to_mapping_index[2] = 1;

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

  // Parameters for building H-matrices.
  ConfHMatrix             hmat_params{32, 32, 1, 1, 0.8, 5, 0.01, false};
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

  // Create a continuous Lagrangian finite element and a DoF handler for the
  // Sobolev space \f$H^{1/2}(\Gamma)\f$.
  FE_Q<dim, spacedim>       fe_H_half(1);
  DoFHandler<dim, spacedim> dof_handler_H_half(tria);
  dof_handler_H_half.distribute_dofs(fe_H_half);
  BEMFunctionSpace<dim, spacedim, SearchableMaterialIdContainer, double> H_half(
    dof_handler_H_half,
    static_cast<unsigned int>(hmat_params.n_min_for_ct),
    hmat_params.cutoff_level_ct);

  // Create a discontinuous Lagrangian finite element and a DoF handler for the
  // Sobolev space \f$H^{-1/2}(\Gamma)\f$ space.
  FE_DGQ<dim, spacedim>     fe_H_minus_half(0);
  DoFHandler<dim, spacedim> dof_handler_H_minus_half(tria);
  dof_handler_H_minus_half.distribute_dofs(fe_H_minus_half);
  BEMFunctionSpace<dim, spacedim, SearchableMaterialIdContainer, double>
    H_minus_half(dof_handler_H_minus_half,
                 static_cast<unsigned int>(hmat_params.n_min_for_ct),
                 hmat_params.cutoff_level_ct);

  // Create a bilinear form \f$b_V: H^{-1/2}(\Gamma)\times H^{-1/2}(\Gamma)
  // \rightarrow \mathbb{R}\f$ for the single layer potential operator \f$V\f$.
  BEMBilinearForm<dim,
                  spacedim,
                  SearchableMaterialIdContainer,
                  SingleLayerKernel>
    bV(H_minus_half, H_minus_half);
  bV.build_block_cluster_tree(
    hmat_params.eta, static_cast<unsigned int>(hmat_params.n_min_for_bct));
  // Create a bilinear form \f$b_{\frac{1}{2}I+K}: H^{1/2}{\Gamma}\times
  // H^{-1/2}{\Gamma} \rightarrow \mathbb{R}\f$ for the double layer potential
  // operator plus a scaled identity operator \f$\frac{1}{2}I+K\f$. This
  // bilinear form is needed to build the right hand side vector of the Laplace
  // equation with a Dirichlet boundary condition.
  BEMBilinearForm<dim,
                  spacedim,
                  SearchableMaterialIdContainer,
                  DoubleLayerKernel>
    bIK(H_half, H_minus_half);
  bIK.build_block_cluster_tree(
    hmat_params.eta, static_cast<unsigned int>(hmat_params.n_min_for_bct));

  // Build an H-matrix for bV.
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
  // Build an H-matrix for bIK.
  std::unique_ptr<HMatrix<spacedim, double>> IK =
    bIK.build_hmatrix_with_mass_matrix(hmat_params,
                                       sauter_quad_near_field_params,
                                       sauter_quad_far_field_params,
                                       parallel_params,
                                       1.0,
                                       0.5,
                                       SauterQuadratureRule<dim>(5, 4, 4, 3),
                                       QGauss<dim>(2),
                                       mappings,
                                       material_id_to_mapping_index,
                                       tria_mapping_support_points_cpu,
                                       tria_mapping_support_points_gpu,
                                       tria_mapping_indices_gpu,
                                       subdomain_topology);

  // Print out the leaf set information of H-matrices. For each leaf node,
  // the DoF index ranges in the block cluster, near field/far field flag and
  // matrix rank are printed.
  std::ofstream leaf_set("V-leaf-set.dat");
  V->write_leaf_set_by_iteration(leaf_set);
  leaf_set.close();

  leaf_set.open("IK-leaf-set.dat");
  IK->write_leaf_set_by_iteration(leaf_set);
  leaf_set.close();

  // Delete manifolds and mappings.
  for (auto &m : manifolds)
    if (m.second != nullptr)
      delete m.second;

  for (auto &m : mappings)
    if (m != nullptr)
      delete m;

  tria_mapping_support_points_gpu.release();
  tria_mapping_indices_gpu.release();

  return 0;
}
