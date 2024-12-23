/**
 * @file op-precond-hmatrix-for-dirichlet.cu
 * @brief Verify building the preconditioner matrix on refined mesh for
 * operator preconditioning used in Laplace Dirichlet problem.
 *
 * @ingroup preconditioner
 * @author Jihuan Tian
 * @date 2024-12-07
 */

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/manifold_lib.h>

#include <catch2/catch_all.hpp>

#include <fstream>
#include <iostream>
#include <map>

#include "debug_tools.hcu"
#include "grid_in_ext.h"
#include "hbem_test_config.h"
#include "preconditioners/preconditioner_for_laplace_dirichlet.h"

using namespace Catch::Matchers;
using namespace HierBEM;
using namespace dealii;
using namespace std;

template <int dim, int spacedim>
void
initialize_manifolds_from_manifold_description(
  Triangulation<dim, spacedim>                            &tria,
  std::map<EntityTag, types::manifold_id>                 &manifold_description,
  std::map<types::manifold_id, Manifold<dim, spacedim> *> &manifolds)
{
  // Assign manifold ids to all cells in the triangulation.
  for (auto &cell : tria.active_cell_iterators())
    {
      cell->set_all_manifold_ids(manifold_description[cell->material_id()]);
    }

  // Associate manifold objects with manifold ids in the triangulation.
  for (const auto &m : manifolds)
    {
      tria.set_manifold(m.first, *m.second);
    }
}

namespace HierBEM
{
  namespace CUDAWrappers
  {
    extern cudaDeviceProp device_properties;
  }
} // namespace HierBEM

TEST_CASE("Build preconditioner matrix for Laplace Dirichlet", "[op-precond]")
{
  ofstream ofs("op-precond-hmatrix-for-dirichlet.log");

  // Initialize CUDA device parameters.
  const size_t stack_size = 1024 * 10;
  AssertCuda(cudaDeviceSetLimit(cudaLimitStackSize, stack_size));
  deallog << "CUDA stack size has been set to " << stack_size << endl;

  /**
   * @internal Get GPU device properties.
   */
  AssertCuda(
    cudaGetDeviceProperties(&HierBEM::CUDAWrappers::device_properties, 0));

  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  // Read mesh into triangulation.
  Triangulation<dim, spacedim> tria(
    Triangulation<dim,
                  spacedim>::MeshSmoothing::limit_level_difference_at_vertices);
  ifstream in(HBEM_TEST_MODEL_DIR "sphere2d-quasi-structured-quad.msh");
  read_msh(in, tria, false, true, false);

  // Initialize Gmsh.
  gmsh::initialize();
  gmsh::option::setNumber("General.Verbosity", 0);
  gmsh::open(HBEM_TEST_MODEL_DIR "sphere.brep");
  gmsh::model::occ::synchronize();

  // Get all surface entities tags.
  gmsh::vectorpair surface_dimtags;
  gmsh::model::occ::getEntities(surface_dimtags, dim);

  // Finalize Gmsh.
  gmsh::clear();
  gmsh::finalize();

  // Manually associate manifold ids with material ids.
  std::map<EntityTag, types::manifold_id> manifold_description;
  for (unsigned int i = 0; i < surface_dimtags.size(); i++)
    {
      manifold_description[surface_dimtags[i].second] = i;
    }

  // Create and assign manifolds.
  const Point<3>                                          center(0, 0, 0);
  std::map<types::manifold_id, Manifold<dim, spacedim> *> manifolds;
  SphericalManifold<dim, spacedim> *ball_surface_manifold =
    new SphericalManifold<dim, spacedim>(center);
  manifolds[0] = ball_surface_manifold;

  initialize_manifolds_from_manifold_description(tria,
                                                 manifold_description,
                                                 manifolds);

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

  // Build surface-to-volume and volume-to-surface relationship.
  SubdomainTopology<dim, spacedim> subdomain_topology;
  subdomain_topology.generate_topology(HBEM_TEST_MODEL_DIR "sphere.brep",
                                       HBEM_TEST_MODEL_DIR
                                       "sphere2d-quasi-structured-quad.msh");

  // Define the primal space and dual space with respect to the single layer
  // potential operator.
  FE_DGQ<dim, spacedim> fe_primal_space(0);
  FE_Q<dim, spacedim>   fe_dual_space(1);

  // Create the preconditioner.
  PreconditionerForLaplaceDirichlet<dim, spacedim, double> precond(
    fe_primal_space, fe_dual_space, tria);

  precond.get_triangulation().copy_triangulation(tria);
  precond.get_triangulation().refine_global();

  // Build and print out the mass matrix on the refined mesh first, because it
  // is needed by the preconditioner matrix.
  precond.initialize_dof_handlers();
  precond.build_dof_to_cell_topology();
  precond.build_mass_matrix_on_refined_mesh(QGauss<dim>(2));
  const SparseMatrix<double> &Mr = precond.get_mass_matrix();
  print_sparse_matrix_to_mat(ofs, "Mr", Mr, 15, true, 25);

  // Build the preconditioner matrix on the refined mesh.
  precond.build_preconditioner_hmat_on_refined_mesh(
    MultithreadInfo::n_threads(),
    HMatrixParameters(64, 64, 1.0, 2, 0.1),
    mappings,
    material_id_to_mapping_index,
    SurfaceNormalDetector<dim, spacedim>(subdomain_topology),
    SauterQuadratureRule<dim>(5, 4, 4, 3));
  const HMatrixSymm<spacedim, double> &Br =
    precond.get_preconditioner_hmatrix();

  // Print out the preconditioner matrix on the refined mesh as full matrix.
  Br.print_as_formatted_full_matrix(ofs, "Br", 15, true, 25);

  // We also build and print out the two linking matrices, with which we can
  // check if their sizes are consistent with the preconditioning matrix.
  precond.build_coupling_matrix();
  const SparseMatrix<double> &Cp = precond.get_coupling_matrix();
  print_sparse_matrix_to_mat(ofs, "Cp", Cp, 15, true, 25);

  precond.build_averaging_matrix();
  const SparseMatrix<double> &Cd = precond.get_averaging_matrix();
  print_sparse_matrix_to_mat(ofs, "Cd", Cd, 15, true, 25);

  // Check the matrix sizes.
  REQUIRE(Cd.n() == Br.get_m());
  REQUIRE(Cd.n() == Mr.m());
  REQUIRE(Mr.n() == Cp.n());

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
