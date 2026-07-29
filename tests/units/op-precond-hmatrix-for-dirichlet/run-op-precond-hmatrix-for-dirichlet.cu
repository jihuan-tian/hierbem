// Copyright (C) 2024-2026 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

#include <deal.II/base/logstream.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/table.h>
#include <deal.II/base/types.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <catch2/catch_all.hpp>

#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "bem/bem_tools.h"
#include "cad_mesh/outward_surface_normal_detector.h"
#include "cad_mesh/subdomain_topology.h"
#include "config_file/config_structs.h"
#include "config_file/cu_related.h"
#include "grid/grid_in_ext.h"
#include "grid/grid_out_ext.h"
#include "linear_algebra/cu_table.hcu"
#include "preconditioners/preconditioner_for_laplace_single_layer_bio.h"
#include "quadrature/sauter_quadrature_tools.h"

using namespace Catch::Matchers;
using namespace HierBEM;

void
run_op_precond_hmatrix_for_dirichlet()
{
  std::ofstream ofs("op-precond-hmatrix-for-dirichlet.log");
  deallog.pop();
  deallog.depth_console(0);
  deallog.depth_file(5);
  deallog.attach(ofs);

  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  const Point<spacedim> center(0, 0, 0);
  const double          radius(1);

  Triangulation<dim, spacedim> tria;
  GridGenerator::hyper_sphere(tria, center, radius);
  tria.refine_global(1);
  std::string   mesh_file("surface-mesh.msh");
  std::ofstream mesh_out(mesh_file);
  write_msh_correct(tria, mesh_out);
  mesh_out.close();

  // Reread the mesh as a single level triangulation.
  tria.clear();
  tria.set_mesh_smoothing(
    Triangulation<dim,
                  spacedim>::MeshSmoothing::limit_level_difference_at_vertices);
  std::ifstream mesh_in(mesh_file);
  read_msh(mesh_in, tria, false, true, false);
  mesh_in.close();

  // Create the map from material id to manifold id.
  std::map<EntityTag, types::manifold_id> manifold_description;
  manifold_description[0] = 0;

  // Create and assign manifold.
  std::map<types::manifold_id, Manifold<dim, spacedim> *> manifolds;
  SphericalManifold<dim, spacedim>                       *spherical_manifold =
    new SphericalManifold<dim, spacedim>(center);
  manifolds[0] = spherical_manifold;
  tria.set_manifold(0, *spherical_manifold);

  // Create different orders of mapping.
  std::vector<MappingInfo<dim, spacedim> *> mappings;
  const unsigned                            max_mapping_order = 2;
  mappings.reserve(max_mapping_order);
  for (unsigned int i = 1; i <= max_mapping_order; i++)
    {
      mappings.push_back(new MappingInfo<dim, spacedim>(i));
    }

  // Construct the map from material ids to mapping indices.
  std::map<types::material_id, unsigned int> material_id_to_mapping_index;
  material_id_to_mapping_index[0] = 1;

  SubdomainTopology<dim, spacedim> subdomain_topology;
  subdomain_topology.generate_single_domain_topology_for_dealii_model({0});

  // Define the primal space and dual space with respect to the single layer
  // potential operator.
  FE_DGQ<dim, spacedim> fe_primal_space(0);
  FE_Q<dim, spacedim>   fe_dual_space(1);

  // Refine the triangulation which is needed by the preconditioner.
  tria.refine_global();
  mesh_out.open("refined-mesh.msh");
  write_msh_correct(tria, mesh_out);
  mesh_out.close();

  // Create the preconditioner. Since we do not apply the preconditioner to the
  // system matrix in this case, the conversion between internal and external
  // DoF numberings is not needed. Therefore, we pass a dummy numbering to the
  // preconditioner's constructor. Its size is initialized to the number of
  // cells in the primal mesh.
  std::vector<types::global_dof_index> dummy_numbering(tria.n_cells(0));
  LaplaceSingleLayerPreconditioner<dim, spacedim, double, double> precond(
    fe_primal_space,
    fe_dual_space,
    tria,
    dummy_numbering,
    dummy_numbering,
    ConfOperatorPreconditioner());

  ConfHMatrix         hmat_params{64, 64, 8, 4, 1.0, 2, 0.1, false};
  ConfSauterQuad      sauter_quad_params;
  ConfParallelization parallel_params;

  // Initialize CUDA stack size and device properties.
  initCudaRuntime(parallel_params);

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

  precond.setup_preconditioner(hmat_params,
                               sauter_quad_params.near_field,
                               sauter_quad_params.far_field,
                               parallel_params,
                               subdomain_topology,
                               mappings,
                               material_id_to_mapping_index,
                               tria_mapping_support_points_cpu,
                               tria_mapping_support_points_gpu,
                               tria_mapping_indices_gpu,
                               OutwardSurfaceNormalDetector(),
                               SauterQuadratureRule<dim>(
                                 sauter_quad_params.hyper_singular_order),
                               QGauss<dim>(2));

  // Print out the preconditioner matrix on the refined mesh as full matrix.
  const HMatrix<spacedim, double> &Br = precond.get_preconditioner_hmatrix();
  Br.print_leaf_set_info(ofs);
  std::ofstream out("op-precond-hmatrix-for-dirichlet.output");
  Br.print_as_formatted_full_matrix(out, "Br", 15, true, 25);
  out.close();

  // Get the averaging matrix for matrix size compatibility checking.
  const SparseMatrix<double> &Cd = precond.get_averaging_matrix();
  REQUIRE(Cd.n() == Br.get_m());

  tria_mapping_support_points_gpu.release();
  tria_mapping_indices_gpu.release();

  // Release manifold objects.
  for (auto &m : manifolds)
    {
      delete m.second;
    }

  // Release mapping objects.
  for (auto m : mappings)
    {
      delete m;
    }

  ofs.close();
}
