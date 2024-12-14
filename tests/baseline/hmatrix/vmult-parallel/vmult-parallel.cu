/**
 * @file vmult-parallel.cu
 * @brief Verify the performance of parallel \hmatrix/vector multiplication.
 *
 * @ingroup hmatrix
 * @author Jihuan Tian
 * @date 2024-03-19
 */

#include <boost/program_options.hpp>

#include <cuda_runtime.h>
#include <openblas-pthread/cblas.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>

#include "grid_in_ext.h"
#include "hbem_test_config.h"
#include "laplace_bem.hcu"
#include "mapping/mapping_info.hcu"
#include "subdomain_topology.h"

using namespace dealii;
using namespace HierBEM;
namespace po = boost::program_options;

struct CmdOpts
{
  unsigned int mapping_order;
  unsigned int refinement;
  unsigned int repeats;
};

CmdOpts
parse_cmdline(int argc, char *argv[])
{
  CmdOpts                 opts;
  po::options_description desc("Allowed options");

  // clang-format off
  desc.add_options()
    ("help,h", "show help message")
    ("mapping-order,o", po::value<unsigned int>()->default_value(2), "Mapping order for the sphere")
    ("refinement,r", po::value<unsigned int>()->default_value(1), "Number of refinements")
    ("repeats,p", po::value<unsigned int>()->default_value(10), "Repeat times for vmult");
  // clang-format on

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help"))
    {
      std::cout << desc << std::endl;
      std::exit(EXIT_SUCCESS);
    }

  opts.mapping_order = vm["mapping-order"].as<unsigned int>();
  opts.refinement    = vm["refinement"].as<unsigned int>();
  opts.repeats       = vm["repeats"].as<unsigned int>();

  return opts;
}

int
main(int argc, char *argv[])
{
  CmdOpts opts = parse_cmdline(argc, argv);

  /**
   * @internal Set number of threads used for OpenBLAS.
   */
  openblas_set_num_threads(1);

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
  manifold_id_to_mapping_order[0] = opts.mapping_order;
  manifold_id_to_mapping_order[1] = opts.mapping_order;

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

  HierBEM::CUDAWrappers::LaplaceKernel::SingleLayerKernel<spacedim>
    single_layer_kernel;

  const unsigned int n_min    = 32;
  const double       eta      = 0.8;
  const double       max_rank = 5;
  const double       epsilon  = 0.01;

  for (unsigned int i = 0; i <= opts.refinement; i++)
    {
      std::cout << "=== Mesh refinement #" << i << std::endl;

      dof_handler.distribute_dofs(fe);

      // Generate a list of cell iterators which will be used for constructing
      // the dof-to-cell topology.
      std::vector<typename DoFHandler<dim, spacedim>::cell_iterator>
        cell_iterators;
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

      // Create a symmetric H-matrix with respect to the block cluster tree.
      HMatrix<3, double>::set_leaf_set_traversal_method(
        HMatrix<3, double>::SpaceFillingCurveType::Hilbert);
      HMatrix<spacedim> V(bct,
                          max_rank,
                          HMatrixSupport::Property::symmetric,
                          HMatrixSupport::BlockType::diagonal_block);

      // Assemble the H-matrix using ACA.
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
        true);

      // Generate a random vector as @p x.
      Vector<double> x(V.get_n());
      std::mt19937   rand_engine;
      for (unsigned int j = 0; j < V.get_n(); j++)
        {
          std::uniform_real_distribution<double> uniform_distribution(1, 10);
          x(j) = uniform_distribution(rand_engine);
        }

      /**
       * Limit the number of OpenBLAS threads.
       */
      openblas_set_num_threads(1);

      Timer timer;
      V.prepare_for_vmult_or_tvmult(true, true);
      timer.stop();
      print_wall_time(std::cout,
                      timer,
                      std::string("prepare vmult with thread num=") +
                        std::to_string(
                          V.compute_vmult_or_tvmult_thread_num(true, true)));

      // Perform \hmatrix/vector multiplication.
      timer.start();
      for (unsigned int j = 0; j < opts.repeats; j++)
        {
          Vector<double> y(V.get_m());
          V.vmult_task_parallel(1.0, y, 0.3, x);
        }
      timer.stop();
      const double elapsed_time = timer.last_wall_time();
      std::cout << "Elapsed wall time for " << std::string("vmult") << " is "
                << elapsed_time / opts.repeats << "s" << std::endl;

      if (i < opts.refinement)
        // Refine the mesh.
        tria.refine_global(1);
    }

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
