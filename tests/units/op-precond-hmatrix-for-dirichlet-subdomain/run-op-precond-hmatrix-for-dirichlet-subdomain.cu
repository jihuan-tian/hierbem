#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <catch2/catch_all.hpp>
#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <vector>

#include "debug_tools.h"
#include "grid_in_ext.h"
#include "grid_out_ext.h"
#include "hmatrix/hmatrix_parameters.h"
#include "preconditioners/preconditioner_for_laplace_dirichlet.h"
#include "subdomain_topology.h"

using namespace Catch::Matchers;
using namespace HierBEM;

// Assign Dirichlet boundary condition to the left half sphere and Neumann
// boundary condition to the right half. Left and right is defined with respect
// to the X coordinate of the sphere center.
void
assign_material_ids(Triangulation<2, 3> &tria, const Point<3> &sphere_center)
{
  for (auto &cell : tria.active_cell_iterators())
    {
      if (cell->center()(0) <= sphere_center(0))
        cell->set_material_id(1);
      else
        cell->set_material_id(2);
    }
}

unsigned int
count_number_of_cells_with_material_id(Triangulation<2, 3>     &tria,
                                       const types::material_id id)
{
  unsigned int n = 0;
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->material_id() == id)
      n++;

  return n;
}

void
setup_preconditioner(PreconditionerForLaplaceDirichlet<2, 3, double> &precond,
                     const Triangulation<2, 3>                       &tria)
{
  precond.get_triangulation().copy_triangulation(tria);
  precond.get_triangulation().refine_global();
  precond.initialize_dof_handlers();
  precond.generate_dof_selectors();
  precond.generate_maps_between_full_and_local_dof_ids();
  precond.build_dof_to_cell_topology();
  precond.build_mass_matrix_on_refined_mesh(QGauss<2>(2));
}

class OutwardSurfaceNormalDetector
{
public:
  bool
  is_normal_vector_inward([[maybe_unused]] const types::material_id m) const
  {
    return false;
  }
};

namespace HierBEM
{
  namespace CUDAWrappers
  {
    extern cudaDeviceProp device_properties;
  }
} // namespace HierBEM

void
run_op_precond_hmatrix_for_dirichlet()
{
  std::ofstream ofs("op-precond-hmatrix-for-dirichlet-subdomain.log");
  deallog.pop();
  deallog.depth_console(0);
  deallog.depth_file(5);
  deallog.attach(ofs);

  // Initialize CUDA device parameters.
  const size_t stack_size = 1024 * 10;
  AssertCuda(cudaDeviceSetLimit(cudaLimitStackSize, stack_size));
  deallog << "CUDA stack size has been set to " << stack_size << std::endl;

  /**
   * @internal Get GPU device properties.
   */
  AssertCuda(
    cudaGetDeviceProperties(&HierBEM::CUDAWrappers::device_properties, 0));

  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  const Point<spacedim> center(0, 0, 0);
  const double          radius(1);

  Triangulation<dim, spacedim> tria;
  GridGenerator::hyper_sphere(tria, center, radius);
  tria.refine_global(2);
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

  assign_material_ids(tria, center);
  mesh_out.open("surface-mesh-with-materials.msh");
  write_msh_correct(tria, mesh_out);
  mesh_out.close();

  // Create the map from material id to manifold id.
  std::map<EntityTag, types::manifold_id> manifold_description;
  manifold_description[1] = 0;
  manifold_description[2] = 0;

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
  material_id_to_mapping_index[1] = 1;
  material_id_to_mapping_index[2] = 1;

  SubdomainTopology<dim, spacedim> subdomain_topology;
  subdomain_topology.generate_single_domain_topology_for_dealii_model({1, 2});

  // Define the primal space and dual space with respect to the single layer
  // potential operator.
  FE_DGQ<dim, spacedim> fe_primal_space(0);
  FE_Q<dim, spacedim>   fe_dual_space(1);

  // Create the preconditioner. Since we do not apply the preconditioner to the
  // system matrix in this case, the conversion between internal and external
  // DoF numberings is not needed. Therefore, we pass a dummy numbering to the
  // preconditioner's constructor. Its size is initialized to the number of
  // cells having material id 1.
  std::vector<types::global_dof_index> dummy_numbering(
    count_number_of_cells_with_material_id(tria, 1));
  std::set<types::material_id> subdomain_material_ids = {1};
  PreconditionerForLaplaceDirichlet<dim, spacedim, double> precond(
    fe_primal_space,
    fe_dual_space,
    tria,
    dummy_numbering,
    dummy_numbering,
    subdomain_material_ids);

  setup_preconditioner(precond, tria);

  // Build the preconditioner matrix on the refined mesh.
  HMatrixParameters hmat_params(64,  // Minimum cluster node size
                                64,  // Minimum block cluster node size
                                1.0, // Admissibility constant eta
                                2,   // Maximum H-matrix rank
                                0.1  // Relative error for ACA iteration
  );
  precond.build_cluster_and_block_cluster_trees(hmat_params, mappings);
  precond.build_preconditioner_hmat_on_refined_mesh(
    MultithreadInfo::n_threads(),
    hmat_params,
    subdomain_topology,
    mappings,
    material_id_to_mapping_index,
    OutwardSurfaceNormalDetector(),
    SauterQuadratureRule<dim>(5, 4, 4, 3));

  // Print out the preconditioner matrix on the refined mesh as full matrix.
  const HMatrix<spacedim, double> &Br = precond.get_preconditioner_hmatrix();
  Br.print_leaf_set_info(ofs);
  std::ofstream out("op-precond-hmatrix-for-dirichlet-subdomain.output");
  Br.print_as_formatted_full_matrix(out, "Br", 15, true, 25);
  out.close();

  // We also build the averaging matrix for matrix size compatibility checking.
  precond.build_averaging_matrix();
  const SparseMatrix<double> &Cd = precond.get_averaging_matrix();
  REQUIRE(Cd.n() == Br.get_m());

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
