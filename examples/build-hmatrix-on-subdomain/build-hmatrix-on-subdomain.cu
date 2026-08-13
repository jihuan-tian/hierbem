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
 * @file build-hmatrix-on-subdomain.cu
 * @brief Example for building an H-matrix on a subdomain which is specified by
 * a set of material ids.
 *
 * @ingroup examples
 * @author Jihuan Tian
 * @date 2025-11-04
 */

#include <deal.II/base/multithread_info.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h> // QGauss
#include <deal.II/base/table.h>
#include <deal.II/base/table_indices.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/manifold.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <string>
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
#include "hmatrix/hmatrix.h"
#include "linear_algebra/cu_table.hcu"
#include "mapping/mapping_info.h"
#include "platform_shared/laplace_kernels.h"
#include "quadrature/sauter_quadrature_tools.h"

using namespace dealii;
using namespace HierBEM;
using namespace HierBEM::PlatformShared::LaplaceKernel;

using SearchableMaterialIdContainer = std::set<EntityTag>;

template <int dim, int spacedim>
void
visualize_dofs_in_function_space(
  const std::string &file_basename,
  const BEMFunctionSpace<dim, spacedim, SearchableMaterialIdContainer, double>
    &space)
{
  const types::global_dof_index n_dofs = space.get_dof_handler().n_dofs();
  const std::vector<bool>      &dof_selectors = space.get_dof_selectors();
  Vector<double>                dof_markers(n_dofs);
  for (types::global_dof_index i = 0; i < n_dofs; i++)
    if (dof_selectors[i])
      dof_markers(i) = 1.0;
    else
      dof_markers(i) = 0;

  std::ofstream          vtk_output(file_basename + ".vtk");
  DataOut<dim, spacedim> data_out;
  data_out.add_data_vector(space.get_dof_handler(), dof_markers, "dof_support");
  data_out.build_patches();
  data_out.write_vtk(vtk_output);

  std::ofstream                       point_output(file_basename + ".txt");
  const std::vector<Point<spacedim>> &support_points =
    space.get_support_points();
  for (types::global_dof_index i = 0; i < support_points.size(); i++)
    point_output << support_points[i] << "\n";

  point_output.close();
}

int
main()
{
  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  // Read the triangulation.
  Triangulation<dim, spacedim> tria;
  std::ifstream                mesh_in(HBEM_TEST_MODEL_DIR "bar.msh");
  read_msh(mesh_in, tria);
  // Generate surface-to-volume and volume-to-surface topology.
  SubdomainTopology<dim, spacedim> subdomain_topology;
  subdomain_topology.generate_topology(HBEM_TEST_MODEL_DIR "bar.brep",
                                       HBEM_TEST_MODEL_DIR "bar.msh");

  // Define manifold for the bar.
  std::map<types::manifold_id, Manifold<dim, spacedim> *> manifolds;
  Manifold<dim, spacedim> *flat_manifold = new FlatManifold<dim, spacedim>();
  manifolds[0]                           = flat_manifold;

  // Assign manifold ids to surface entities in the CAD model.
  std::map<EntityTag, types::manifold_id> manifold_description;
  for (types::material_id i = 1; i <= 6; i++)
    manifold_description[i] = 0;

  // Assign manifolds to the triangulation.
  for (auto &cell : tria.active_cell_iterators())
    cell->set_all_manifold_ids(manifold_description[cell->material_id()]);

  for (const auto &m : manifolds)
    tria.set_manifold(m.first, *m.second);

  // Define only 1st order mapping for flat surfaces
  std::vector<MappingInfo<dim, spacedim> *> mappings(1);
  mappings[0] = new MappingInfo<dim, spacedim>(1);

  // Construct a map from material ids to mapping indices.
  std::map<types::material_id, unsigned int> material_id_to_mapping_index;
  for (types::material_id i = 1; i <= 6; i++)
    material_id_to_mapping_index[i] = 0;

  Table<2, Point<spacedim, double>> tria_mapping_support_points_cpu;
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
  ConfHMatrix             hmat_params{4, 4, 1, 1, 1.2, 5, 5, 0.01, false};
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
  // Define a function space
  // \f$\tilde{H}_h^{1/2}(\Gamma_{\mathrm{D}}^{\ast})\f$.
  BEMFunctionSpace<dim, spacedim, SearchableMaterialIdContainer, double>
    H_half_Gamma_D(dof_handler_H_half,
                   {5, 6},
                   static_cast<unsigned int>(hmat_params.n_min_for_ct),
                   static_cast<unsigned int>(hmat_params.cutoff_level_ct),
                   true,
                   false);
  // Define a function space \f$\tilde{H}_h^{1/2}(\Gamma_{\mathrm{N}})\f$.
  BEMFunctionSpace<dim, spacedim, SearchableMaterialIdContainer, double>
    H_half_Gamma_N(dof_handler_H_half,
                   {1, 2, 3, 4},
                   static_cast<unsigned int>(hmat_params.n_min_for_ct),
                   static_cast<unsigned int>(hmat_params.cutoff_level_ct),
                   false,
                   false);

  // Create a discontinuous Lagrangian finite element and a DoF handler for the
  // Sobolev space \f$H^{-1/2}(\Gamma)\f$ space.
  FE_DGQ<dim, spacedim>     fe_H_minus_half(0);
  DoFHandler<dim, spacedim> dof_handler_H_minus_half(tria);
  dof_handler_H_minus_half.distribute_dofs(fe_H_minus_half);
  // Define a function space \f$\tilde{H}_h^{-1/2}(\Gamma_{\mathrm{D}})\f$.
  BEMFunctionSpace<dim, spacedim, SearchableMaterialIdContainer, double>
    H_minus_half_Gamma_D(dof_handler_H_minus_half,
                         {5, 6},
                         static_cast<unsigned int>(hmat_params.n_min_for_ct),
                         static_cast<unsigned int>(hmat_params.cutoff_level_ct),
                         true,
                         false);
  // Define a function space \f$\tilde{H}_h^{-1/2}(\Gamma_{\mathrm{N}})\f$.
  BEMFunctionSpace<dim, spacedim, SearchableMaterialIdContainer, double>
    H_minus_half_Gamma_N(dof_handler_H_minus_half,
                         {1, 2, 3, 4},
                         static_cast<unsigned int>(hmat_params.n_min_for_ct),
                         static_cast<unsigned int>(hmat_params.cutoff_level_ct),
                         true,
                         false);

  // Create a bilinear form \f$b_V: b_{V_1}:
  // \tilde{H}^{-1/2}(\Gamma_{\mathrm{D}}) \times
  // \tilde{H}^{-1/2}(\Gamma_{\mathrm{D}}) \rightarrow \mathbb{R}\f$
  BEMBilinearForm<dim,
                  spacedim,
                  SearchableMaterialIdContainer,
                  SingleLayerKernel>
    bV1(H_minus_half_Gamma_D, H_minus_half_Gamma_D);
  bV1.build_block_cluster_tree(
    hmat_params.eta,
    static_cast<unsigned int>(hmat_params.n_min_for_bct),
    static_cast<unsigned int>(hmat_params.cutoff_level_bct));
  // Create a bilinear form \f$b_{K_1}:
  // \tilde{H}^{1/2}(\Gamma_{\mathrm{N}}) \times
  // \tilde{H}^{-1/2}(\Gamma_{\mathrm{D}}) \rightarrow \mathbb{R}\f$.
  BEMBilinearForm<dim,
                  spacedim,
                  SearchableMaterialIdContainer,
                  DoubleLayerKernel>
    bK1(H_half_Gamma_N, H_minus_half_Gamma_D);
  bK1.build_block_cluster_tree(
    hmat_params.eta,
    static_cast<unsigned int>(hmat_params.n_min_for_bct),
    static_cast<unsigned int>(hmat_params.cutoff_level_bct));
  // Create a bilinear form \f$b_{V_2}: H^{-1/2}(\Gamma) \times
  // \tilde{H}^{-1/2}(\Gamma_{\mathrm{D}}) \rightarrow \mathbb{R}\f$.
  BEMBilinearForm<dim,
                  spacedim,
                  SearchableMaterialIdContainer,
                  SingleLayerKernel>
    bV2(H_minus_half_Gamma_N, H_minus_half_Gamma_D);
  bV2.build_block_cluster_tree(
    hmat_params.eta,
    static_cast<unsigned int>(hmat_params.n_min_for_bct),
    static_cast<unsigned int>(hmat_params.cutoff_level_bct));
  // Create a bilinear form \f$b_{sigma I_1+K_2}: H^{1/2}(\Gamma) \times
  // \tilde{H}^{-1/2}(\Gamma_{\mathrm{D}}) \rightarrow \mathbb{R}\f$.
  BEMBilinearForm<dim,
                  spacedim,
                  SearchableMaterialIdContainer,
                  DoubleLayerKernel>
    bI1K2(H_half_Gamma_D, H_minus_half_Gamma_D);
  bI1K2.build_block_cluster_tree(
    hmat_params.eta,
    static_cast<unsigned int>(hmat_params.n_min_for_bct),
    static_cast<unsigned int>(hmat_params.cutoff_level_bct));

  // Build an H-matrix for bV1.
  std::unique_ptr<HMatrix<spacedim, double>> V1 =
    bV1.build_hmatrix(hmat_params,
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
  // Build an H-matrix for bK1.
  std::unique_ptr<HMatrix<spacedim, double>> K1 =
    bK1.build_hmatrix(hmat_params,
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
  // Build an H-matrix for bV2.
  std::unique_ptr<HMatrix<spacedim, double>> V2 =
    bV2.build_hmatrix(hmat_params,
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
  // Build an H-matrix for bI1K2.
  std::unique_ptr<HMatrix<spacedim, double>> I1K2 =
    bI1K2.build_hmatrix_with_mass_matrix(hmat_params,
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

  // Generate visualizations of all function spaces.
  visualize_dofs_in_function_space("H_half_Gamma_D", H_half_Gamma_D);
  visualize_dofs_in_function_space("H_half_Gamma_N", H_half_Gamma_N);
  visualize_dofs_in_function_space("H_minus_half_Gamma_D",
                                   H_minus_half_Gamma_D);
  visualize_dofs_in_function_space("H_minus_half_Gamma_N",
                                   H_minus_half_Gamma_N);

  // Print out the leaf set information of H-matrices. For each leaf node,
  // the DoF index ranges in the block cluster, near field/far field flag and
  // matrix rank are printed.
  std::ofstream leaf_set("V1-leaf-set.dat");
  V1->write_leaf_set_by_iteration(leaf_set);
  leaf_set.close();

  leaf_set.open("K1-leaf-set.dat");
  K1->write_leaf_set_by_iteration(leaf_set);
  leaf_set.close();

  leaf_set.open("V2-leaf-set.dat");
  V2->write_leaf_set_by_iteration(leaf_set);
  leaf_set.close();

  leaf_set.open("I1K2-leaf-set.dat");
  I1K2->write_leaf_set_by_iteration(leaf_set);
  leaf_set.close();

  tria_mapping_support_points_gpu.release();
  tria_mapping_indices_gpu.release();

  // Delete manifolds and mappings.
  for (auto &m : manifolds)
    if (m.second != nullptr)
      delete m.second;

  for (auto &m : mappings)
    if (m != nullptr)
      delete m;

  return 0;
}
