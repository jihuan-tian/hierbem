/**
 * @file hmatrix-related-memory-consumption.cu
 * @brief Estimate the memory consumption of H-matrix, cluster tree and block cluster tree, etc.
 *
 * The parameters are selected to be the same as Bebendorf "Hierarchical
 * matrices" P155.
 *
 * @ingroup testers
 * @author Jihuan Tian
 * @date 2024-02-20
 */

#include <deal.II/base/memory_consumption.h>
#include <deal.II/base/table_handler.h>

#include <boost/program_options.hpp>

#include <cuda_runtime.h>

#include <fstream>
#include <iostream>

#include "laplace_bem.h"

using namespace dealii;
using namespace HierBEM;
namespace po = boost::program_options;

enum RefineType
{
  GLOBAL,
  BOUNDARY
};

struct CmdOpts
{
  RefineType   refine_type;
  unsigned int min_refines;
  unsigned int max_refines;
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
    ("refine-type,T", po::value<std::string>()->default_value("boundary"), "refinement type (global or boundary)")
    ("min-refines,f", po::value<unsigned int>()->default_value(1), "minimum number of refinements")
    ("max-refines,t", po::value<unsigned int>()->default_value(5), "maximum number of refinements")
    ("n-min,n", po::value<unsigned int>()->default_value(32), "n_min criteria for small cluster")
    ("eta,e", po::value<double>()->default_value(0.8), "Admissibility constant eta")
    ("max-rank,r", po::value<unsigned int>()->default_value(5), "Maximum rank allowed for a rank-k matrix")
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

  const std::string refine_type = vm["refine-type"].as<std::string>();
  if (refine_type == "global")
    {
      opts.refine_type = RefineType::GLOBAL;
    }
  else if (refine_type == "boundary")
    {
      opts.refine_type = RefineType::BOUNDARY;
    }
  else
    {
      std::cerr << "Invalid refinement type: " << refine_type << std::endl;
      std::exit(EXIT_FAILURE);
    }

  opts.min_refines = vm["min-refines"].as<unsigned int>();
  opts.max_refines = vm["max-refines"].as<unsigned int>();
  opts.n_min       = vm["n-min"].as<unsigned int>();
  opts.eta         = vm["eta"].as<double>();
  opts.max_rank    = vm["max-rank"].as<unsigned int>();
  opts.epsilon     = vm["epsilon"].as<double>();

  return opts;
}

template <int spacedim>
void
refine_boundary_mesh_for_two_spheres(Triangulation<spacedim> &tria,
                                     const double             inter_distance,
                                     const double             radius,
                                     const std::string       &vtk_filename = "",
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

  // Write out the refined mesh
  if (vtk_filename.size() > 0)
    {
      auto flags = GridOutFlags::Vtk();

      GridOut grid_out;
      grid_out.set_flags(flags);

      std::ofstream out(vtk_filename);
      grid_out.write_vtk(tria, out);
      std::cout << "Volume grid written to " << vtk_filename << " ("
                << tria.n_active_cells() << " cells)" << std::endl;
    }
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

  TableHandler table;

  for (unsigned int i = 0; i < opts.max_refines; i++)
    {
      // Refine the volume mesh.
      if (opts.refine_type == RefineType::GLOBAL)
        {
          tria.refine_global(1);
        }
      else
        {
          refine_boundary_mesh_for_two_spheres(tria, inter_distance, radius);
        }

      if (i + 1 < opts.min_refines)
        {
          continue;
        }

      table.add_value("Refinement", i + 1);

      // Generate surface mesh from the volume mesh. N.B. Before the extraction,
      // the surface mesh generated from previous refinement should be cleared.
      surface_tria.clear();
      std::map<typename Triangulation<dim, spacedim>::cell_iterator,
               typename Triangulation<spacedim>::face_iterator>
        map_from_surface_mesh_to_volume_mesh;
      extract_surface_mesh_for_two_spheres(
        tria,
        surface_tria,
        inter_distance,
        map_from_surface_mesh_to_volume_mesh);

      table.add_value("Object", "Volume triangulation");
      table.add_value("Memory", tria.memory_consumption());

      table.start_new_row();
      table.add_value("Refinement", i + 1);
      table.add_value("Object", "Surface triangulation");
      table.add_value("Memory", surface_tria.memory_consumption());

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
      // Get the spatial coordinates of the support points.
      std::vector<Point<spacedim>> support_points(dof_handler.n_dofs());
      DoFTools::map_dofs_to_support_points(kx_mapping,
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
      table.add_value("Refinement", i + 1);
      table.add_value("Object", "Cluster tree");
      table.add_value("Memory", ct.memory_consumption());

      // Create and partition the block cluster tree.
      BlockClusterTree<spacedim> bct(ct, ct, opts.eta, opts.n_min);
      bct.partition(ct.get_internal_to_external_dof_numbering(),
                    support_points,
                    cell_size_at_support_points);

      table.start_new_row();
      table.add_value("Refinement", i + 1);
      table.add_value("Object", "Block cluster tree");
      table.add_value("Memory", bct.memory_consumption());

      // Create a symmetric H-matrix with respect to the block cluster tree.
      HMatrix<spacedim> V(bct,
                          opts.max_rank,
                          HMatrixSupport::Property::symmetric,
                          HMatrixSupport::BlockType::diagonal_block);

      // Assemble the H-matrix using ACA.
      fill_hmatrix_with_aca_plus_smp(
        MultithreadInfo::n_threads(),
        V,
        ACAConfig(opts.max_rank, opts.epsilon, opts.eta),
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

      table.start_new_row();
      table.add_value("Refinement", i + 1);
      table.add_value("Object", "SLP matrix V");
      table.add_value("Memory", V.memory_consumption());
    }

  table.write_text(std::cout, TableHandler::TextOutputFormat::org_mode_table);

  return 0;
}
