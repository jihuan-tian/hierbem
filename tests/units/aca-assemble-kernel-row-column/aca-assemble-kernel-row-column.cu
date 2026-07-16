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
 * @file aca-assemble-kernel-row-column.cu
 * @brief Verify the assembly of a row/column vector in a far field matrix
 * block.
 *
 * @ingroup test_cases hierarchical_matrices
 * @author Jihuan Tian
 * @date 2026-07-13
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

#include <catch2/catch_all.hpp>
#include <cuda_runtime.h>

#include <array>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "bem/bem_bilinear_form.h"
#include "bem/bem_function_space.h"
#include "bem/bem_tools.h"
#include "bem/bem_values.h"
#include "bem/cu_bem_values.hcu"
#include "cad_mesh/gmsh_manipulation.h"
#include "cad_mesh/subdomain_topology.h"
#include "cluster_tree/cluster_tree.h"
#include "config_file/config_structs.h"
#include "config_file/cu_related.h"
#include "dofs/dof_to_cell_topology.h"
#include "grid/grid_in_ext.h"
#include "hbem_octave_wrapper.h"
#include "hbem_test_config.h"
#include "hmatrix/aca_plus/aca_plus.hcu"
#include "hmatrix/hmatrix.h"
#include "hmatrix/hmatrix_support.h"
#include "linear_algebra/cu_table.hcu"
#include "mapping/mapping_info.h"
#include "platform_shared/laplace_kernels.h"
#include "quadrature/sauter_quadrature_task_buffer_for_vector.hcu"
#include "quadrature/sauter_quadrature_tools.h"
#include "utilities/debug_tools.h"
#include "utilities/unary_template_arg_containers.h"

using namespace dealii;
using namespace HierBEM;
using namespace Catch::Matchers;
using namespace HierBEM::PlatformShared::LaplaceKernel;

using size_type = std::make_unsigned<types::blas_int>::type;

TEST_CASE("Compute a row/column vector with ACA", "[hmatrix]")
{
  HBEMOctaveWrapper &inst = HBEMOctaveWrapper::get_instance();
  inst.add_path(SOURCE_DIR);

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
  ConfHMatrix               hmat_params{32, 32, 0.8, 5, 0.01, false};
  ConfSauterQuadFarField    sauter_quad_far_field_params;
  SauterQuadratureRule<dim> sauter_quad_rule(5, 4, 4, 3);
  ConfParallelization       parallel_params;

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
    dof_handler_H_half, static_cast<unsigned int>(hmat_params.n_min_for_ct));

  // Create a bilinear form \f$b_D: H^{1/2}(\Gamma)\times H^{1/2}(\Gamma)
  // \rightarrow \mathbb{R}\f$ for the hypersingular operator \f$D\f$ with
  // regularization but without stabilization.
  BEMBilinearForm<dim,
                  spacedim,
                  SearchableMaterialIdContainer,
                  HyperSingularKernelRegular>
    bD(H_half, H_half);
  bD.build_block_cluster_tree(
    hmat_params.eta, static_cast<unsigned int>(hmat_params.n_min_for_bct));

  // Create an empty H-matrix for bD.
  std::unique_ptr<HMatrix<spacedim, double>> D =
    std::make_unique<HMatrix<spacedim, double>>(
      bD.get_block_cluster_tree(),
      static_cast<unsigned int>(hmat_params.max_rank),
      HMatrixSupport::Property::symmetric,
      HMatrixSupport::BlockType::diagonal_block);

  // Create BEM values and scratch data.
  BEMValues<dim, spacedim, double> bem_values(
    bD.get_test_space().get_dof_handler().get_fe(),
    bD.get_trial_space().get_dof_handler().get_fe(),
    mappings,
    tria_mapping_support_points_cpu,
    sauter_quad_rule.quad_rule_for_same_panel,
    sauter_quad_rule.quad_rule_for_common_edge,
    sauter_quad_rule.quad_rule_for_common_vertex,
    sauter_quad_rule.quad_rule_for_regular,
    bD.get_kernel().needs_regularization());
  bem_values.fill_shape_function_value_tables();
  HierBEM::CUDAWrappers::CUDABEMValues<dim, spacedim, double> bem_values_gpu;
  bem_values_gpu.allocate_and_assign_from_host(bem_values,
                                               tria_mapping_support_points_gpu,
                                               tria_mapping_indices_gpu);
  bem_values_gpu.compute_bilinear_form_cell_values_for_regular(
    bD.get_cell_iterator_ptrs(),
    bD.get_local_to_global_cell_index_map(),
    SurfaceNormalDetector<dim, spacedim>(subdomain_topology));

  PairCellWiseScratchDataForHMatrixFarField<dim, spacedim, double> scratch_data(
    bD.get_test_space().get_dof_handler().get_fe(),
    bD.get_trial_space().get_dof_handler().get_fe(),
    mappings);
  PairCellWisePerTaskData<dim, spacedim> copy_data(
    bD.get_test_space().get_dof_handler().get_fe(),
    bD.get_trial_space().get_dof_handler().get_fe());

  // Get the first far field matrix block.
  HMatrix<spacedim, double> *far_field_mat = D->get_far_field_leaf_set()[0];

  const std::array<types::global_dof_index, 2> &row_index_range =
    *(far_field_mat->get_row_index_range());
  const std::array<types::global_dof_index, 2> &col_index_range =
    *(far_field_mat->get_col_index_range());

  // Get the size of each dimension of the matrix block to be built.
  const size_type m = row_index_range[1] - row_index_range[0];
  const size_type n = col_index_range[1] - col_index_range[0];

  HierBEM::CUDAWrappers::
    SauterQuadratureTaskBufferForVector<dim, spacedim, double, double>
      sauter_task_buffer_for_row_vector(
        n,
        bD.get_test_space().get_dof_to_cell_topo().max_cells_per_dof,
        bD.get_trial_space().get_dof_to_cell_topo().max_cells_per_dof,
        bem_values,
        scratch_data.cuda_stream_handle);
  HierBEM::CUDAWrappers::
    SauterQuadratureTaskBufferForVector<dim, spacedim, double, double>
      sauter_task_buffer_for_col_vector(
        m,
        bD.get_test_space().get_dof_to_cell_topo().max_cells_per_dof,
        bD.get_trial_space().get_dof_to_cell_topo().max_cells_per_dof,
        bem_values,
        scratch_data.cuda_stream_handle);

  // Row and column vectors.
  Vector<double> v(n);
  Vector<double> u(m);

  // Pin the host memory for v and u, which is required by asynchronous memory
  // copy from the device to the host.
  AssertCuda(cudaHostRegister((void *)(v.data()),
                              sizeof(double) * n,
                              cudaHostRegisterDefault));
  AssertCuda(cudaHostRegister((void *)(u.data()),
                              sizeof(double) * m,
                              cudaHostRegisterDefault));

  /**
   * Generate lists of internal DoF indices (internal DoF numbering) from
   * corresponding index ranges.
   */
  std::vector<types::global_dof_index> row_dof_indices(m);
  std::vector<types::global_dof_index> col_dof_indices(n);
  gen_linear_indices<vector_uta, types::global_dof_index>(row_dof_indices,
                                                          row_index_range[0]);
  gen_linear_indices<vector_uta, types::global_dof_index>(col_dof_indices,
                                                          col_index_range[0]);

  assemble_kernel_row(sauter_task_buffer_for_row_vector,
                      v,
                      sauter_quad_far_field_params,
                      bD.get_kernel(),
                      1.0,
                      {},
                      0.,
                      row_dof_indices[0],
                      col_dof_indices,
                      bD.get_test_space().get_dof_to_cell_topo().topology,
                      bD.get_trial_space().get_dof_to_cell_topo().topology,
                      bem_values_gpu,
                      nullptr,
                      nullptr,
                      bD.get_test_space()
                        .get_cluster_tree()
                        .get_internal_to_external_dof_numbering(),
                      bD.get_trial_space()
                        .get_cluster_tree()
                        .get_internal_to_external_dof_numbering(),
                      mappings,
                      material_id_to_mapping_index,
                      bD.get_global_to_local_cell_index_map(),
                      bem_values.mapping_support_point_table,
                      scratch_data,
                      copy_data);

  assemble_kernel_column(sauter_task_buffer_for_col_vector,
                         u,
                         sauter_quad_far_field_params,
                         bD.get_kernel(),
                         1.0,
                         {},
                         0.,
                         row_dof_indices,
                         col_dof_indices[0],
                         bD.get_test_space().get_dof_to_cell_topo().topology,
                         bD.get_trial_space().get_dof_to_cell_topo().topology,
                         bem_values_gpu,
                         nullptr,
                         nullptr,
                         bD.get_test_space()
                           .get_cluster_tree()
                           .get_internal_to_external_dof_numbering(),
                         bD.get_trial_space()
                           .get_cluster_tree()
                           .get_internal_to_external_dof_numbering(),
                         mappings,
                         material_id_to_mapping_index,
                         bD.get_global_to_local_cell_index_map(),
                         bem_values.mapping_support_point_table,
                         scratch_data,
                         copy_data);

  AssertCuda(cudaHostUnregister((void *)(v.data())));
  AssertCuda(cudaHostUnregister((void *)(u.data())));

  std::string   logfile("aca-assemble-kernel-row-column.output");
  std::ofstream ofs(logfile);
  print_vector_to_mat(ofs, "v", v);
  print_vector_to_mat(ofs, "u", u);
  ofs.close();

  // Calculate relative error
  try
    {
      inst.source_file(SOURCE_DIR "/process.m");
    }
  catch (...)
    {
      // Ignore errors
    }

  // Check relative error
  HBEMOctaveValue out;
  out = inst.eval_string("v_rel_err");
  REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-12));

  out = inst.eval_string("u_rel_err");
  REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-12));

  // Delete manifolds and mappings.
  for (auto &m : manifolds)
    if (m.second != nullptr)
      delete m.second;

  for (auto &m : mappings)
    if (m != nullptr)
      delete m;

  tria_mapping_support_points_gpu.release();
  tria_mapping_indices_gpu.release();
}
