/**
 * @file hmatrix-related-memory-consumption.cu
 * @brief Estimate the memory consumption of H-matrix, cluster tree and block
 * cluster tree, etc.
 *
 * @ingroup test_cases
 * @author Jihuan Tian
 * @date 2024-02-20
 */

#include <deal.II/base/memory_consumption.h>
#include <deal.II/base/table_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_dgq.h>

#include <deal.II/grid/manifold_lib.h>

#include <boost/program_options.hpp>

#include <cuda_runtime.h>

#include <fstream>
#include <iostream>

#include "cad_mesh/subdomain_topology.h"
#include "grid/grid_in_ext.h"
#include "hbem_test_config.h"
#include "hmatrix/aca_plus/aca_plus.hcu"
#include "laplace/laplace_bem.h"
#include "mapping/mapping_info.h"
#include "platform_shared/laplace_kernels.h"
#include "quadrature/sauter_quadrature.hcu"
#include "utilities/unary_template_arg_containers.h"

using namespace dealii;
using namespace HierBEM;
namespace po = boost::program_options;

struct CmdOpts
{
  unsigned int mapping_order;
  unsigned int refinement;
  unsigned int n_min;
  double       eta;
  unsigned int max_rank;
  double       epsilon;
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
    ("refinement,r", po::value<unsigned int>()->default_value(5), "Number of refinements")
    ("n-min,n", po::value<unsigned int>()->default_value(32), "n_min criteria for small cluster")
    ("eta,e", po::value<double>()->default_value(0.8), "Admissibility constant eta")
    ("max-rank,m", po::value<unsigned int>()->default_value(5), "Maximum rank allowed for a rank-k matrix")
    ("epsilon,E", po::value<double>()->default_value(0.01), "Relative error of ACA for building a rank-k matrix");
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
  opts.n_min         = vm["n-min"].as<unsigned int>();
  opts.eta           = vm["eta"].as<double>();
  opts.max_rank      = vm["max-rank"].as<unsigned int>();
  opts.epsilon       = vm["epsilon"].as<double>();

  return opts;
}

int
main(int argc, char *argv[])
{
  CmdOpts opts = parse_cmdline(argc, argv);

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

  HierBEM::PlatformShared::LaplaceKernel::SingleLayerKernel<spacedim, double>
    single_layer_kernel;

  TableHandler table;

  for (unsigned int i = 0; i <= opts.refinement; i++)
    {
      table.add_value("Refinement", i);
      table.add_value("Object", "Surface triangulation");
      table.add_value("Memory", tria.memory_consumption());

      dof_handler.distribute_dofs(fe);

      // Generate a list of cell iterators which will be used for constructing
      // the dof-to-cell topology.
      std::vector<typename DoFHandler<dim, spacedim>::cell_iterator>
        cell_iterators;
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          cell_iterators.push_back(cell);
        }

      DoFToCellTopology<dim, spacedim> dof_to_cell_topo;
      DoFToolsExt::build_dof_to_cell_topology(dof_to_cell_topo,
                                              cell_iterators,
                                              dof_handler);

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
                               opts.n_min);
      ct.partition(support_points, cell_size_at_support_points);

      table.start_new_row();
      table.add_value("Refinement", i);
      table.add_value("Object", "Cluster tree");
      table.add_value("Memory", ct.memory_consumption());

      // Create and partition the block cluster tree.
      BlockClusterTree<spacedim> bct(ct, ct, opts.eta, opts.n_min);
      bct.partition(ct.get_internal_to_external_dof_numbering(),
                    support_points,
                    cell_size_at_support_points);

      table.start_new_row();
      table.add_value("Refinement", i);
      table.add_value("Object", "Block cluster tree");
      table.add_value("Memory", bct.memory_consumption());

      // Create a symmetric H-matrix with respect to the block cluster tree.
      HMatrix<spacedim, double> V(bct,
                                  opts.max_rank,
                                  HMatrixSupport::Property::symmetric,
                                  HMatrixSupport::BlockType::diagonal_block);

      // Assemble the H-matrix using ACA.
      fill_hmatrix_with_aca_plus_smp<
        dim,
        spacedim,
        HierBEM::PlatformShared::LaplaceKernel::SingleLayerKernel,
        double,
        double,
        SurfaceNormalDetector<dim, spacedim>>(
        MultithreadInfo::n_threads(),
        V,
        ACAConfig(opts.max_rank, opts.epsilon, opts.eta),
        single_layer_kernel,
        static_cast<double>(1.0),
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
        SurfaceNormalDetector<dim, spacedim>(subdomain_topology),
        true);

      table.start_new_row();
      table.add_value("Refinement", i);
      table.add_value("Object", "SLP matrix V");
      table.add_value("Memory", V.memory_consumption());

      if (i < opts.refinement)
        // Refine the mesh.
        tria.refine_global(1);
    }

  table.write_text(std::cout, TableHandler::TextOutputFormat::org_mode_table);

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
