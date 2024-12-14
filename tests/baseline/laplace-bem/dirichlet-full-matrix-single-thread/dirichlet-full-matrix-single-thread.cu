/**
 * @file test-dirichlet-full-matrix-single-thread.cu
 * @brief Baseline test for solving Laplace problem with Dirichlet boundary
 * condition based on full matrix BEM, which runs in a single thread.
 *
 * @ingroup testers
 * @author Jihuan Tian
 * @date 2023-10-24
 */
#include <deal.II/fe/mapping_manifold.h>

#include <deal.II/grid/manifold.h>

#include <boost/program_options.hpp>

#include <fstream>
#include <iostream>

#include "grid_in_ext.h"
#include "grid_out_ext.h"
#include "hbem_test_config.h"
#include "laplace_bem.hcu"

using namespace dealii;
using namespace HierBEM;
namespace po = boost::program_options;

struct CmdOpts
{
  unsigned int dirichlet_space_fe_order;
  unsigned int neumann_space_fe_order;
  unsigned int mapping_order;
  bool         run_in_parallel;
  /**
   * Whether the test case directly reads a 2D mesh or reads a 3D mesh first,
   * then extract surface mesh as before.
   */
  bool use_2d_mesh;
  /**
   * Whether use the sphere manifold instead of the flat manifold.
   */
  bool use_sphere_manifold;
  /**
   * Use deal.ii to generate the mesh instead of Gmsh.
   */
  bool         use_dealii_mesh;
  bool         use_occ;
  unsigned int dealii_refinement;
  std::string  cad_file;
  std::string  mesh_file;
};

CmdOpts
parse_cmdline(int argc, char *argv[])
{
  CmdOpts                 opts;
  po::options_description desc("Allowed options");

  // clang-format off
  desc.add_options()
    ("help,h", "show help message")
    ("dirichlet-order,d", po::value<unsigned int>()->default_value(1), "Finite element space order for the Dirichlet data")
    ("neumann-order,n", po::value<unsigned int>()->default_value(0), "Finite element space order for the Neumann data")
    ("mapping-order,m", po::value<unsigned int>()->default_value(1), "Mapping order for the sphere")
    ("enable-parallel,p", po::bool_switch(&opts.run_in_parallel), "Enable parallel execution")
    ("enable-2d-mesh,2", po::bool_switch(&opts.use_2d_mesh), "Enable directly reading 2D mesh")
    ("enable-sphere-manifold,s", po::bool_switch(&opts.use_sphere_manifold), "Enable using sphere manifold")
    ("enable-occ-model", po::bool_switch(&opts.use_occ), "Use OCC model")
    ("enable-dealii-mesh", po::bool_switch(&opts.use_dealii_mesh), "Use deal.ii to generate the mesh instead of Gmsh")
    ("dealii-refinement", po::value<unsigned int>()->default_value(3), "Number of global refinement when deal.ii is used to generate the mesh")
    ("cad-file", po::value<std::string>(), "CAD file")
    ("mesh-file", po::value<std::string>(), "Mesh file");
  // clang-format on

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help"))
    {
      std::cout << desc << std::endl;
      std::exit(EXIT_SUCCESS);
    }

  opts.dirichlet_space_fe_order = vm["dirichlet-order"].as<unsigned int>();
  opts.neumann_space_fe_order   = vm["neumann-order"].as<unsigned int>();
  opts.mapping_order            = vm["mapping-order"].as<unsigned int>();
  opts.dealii_refinement        = vm["dealii-refinement"].as<unsigned int>();

  if (vm.count("mesh-file") == 0)
    {
      if (!opts.use_dealii_mesh)
        {
          if (opts.use_2d_mesh)
            {
              opts.mesh_file = HBEM_TEST_MODEL_DIR "sphere2d-refine-1.msh";
            }
          else
            {
              opts.mesh_file = HBEM_TEST_MODEL_DIR "sphere-refine-1.msh";
            }
        }
    }
  else
    {
      opts.mesh_file = vm["mesh-file"].as<std::string>();
    }

  if (opts.use_occ)
    {
      if (vm.count("cad-file") == 0)
        {
          std::cout << "When occ is enabled, the CAD file should be specified!"
                    << std::endl;
          std::exit(EXIT_FAILURE);
        }
      else
        {
          opts.cad_file = vm["cad-file"].as<std::string>();
        }
    }

  return opts;
}

/**
 * Function object for the Dirichlet boundary condition data, which is
 * also the solution of the Neumann problem. The analytical expression is:
 * \f[
 * u=\frac{1}{4\pi\norm{x-x_0}}
 * \f]
 */
class DirichletBC : public Function<3>
{
public:
  // N.B. This function should be defined outside class NeumannBC or class
  // Example2, if no inline.
  DirichletBC()
    : Function<3>()
    , x0(0.25, 0.25, 0.25)
  {}

  DirichletBC(const Point<3> &x0)
    : Function<3>()
    , x0(x0)
  {}

  double
  value(const Point<3> &p, const unsigned int component = 0) const
  {
    (void)component;
    return 1.0 / 4.0 / numbers::PI / (p - x0).norm();
  }

private:
  /**
   * Location of the Dirac point source \f$\delta(x-x_0)\f$.
   */
  Point<3> x0;
};


/**
 * Function object for the Neumann boundary condition data, which is also
 * the solution of the Dirichlet problem. The analytical expression is
 * \f[
 * \frac{\pdiff u}{\pdiff n}\Big\vert_{\Gamma} = \frac{\langle x-x_c,x_0-x
 * \rangle}{4\pi\norm{x_0-x}^3\rho}
 * \f]
 */
class NeumannBC : public Function<3>
{
public:
  NeumannBC()
    : Function<3>()
    , x0(0.25, 0.25, 0.25)
    , model_sphere_center(0.0, 0.0, 0.0)
    , model_sphere_radius(1.0)
  {}

  NeumannBC(const Point<3> &x0, const Point<3> &center, double radius)
    : Function<3>()
    , x0(x0)
    , model_sphere_center(center)
    , model_sphere_radius(radius)
  {}

  /**
   * @brief Compute the function value at the target point @p.
   *
   * \mynote{In this test case, we use this object to define the analytical
   * solution on the sphere. Therefore, we project the point @p onto the sphere.}
   *
   * @param p
   * @param component
   * @return
   */
  double
  value(const Point<3> &p, const unsigned int component = 0) const
  {
    (void)component;

    Tensor<1, 3> diff_vector = x0 - p;

    return ((p - model_sphere_center) * diff_vector) / 4.0 / numbers::PI /
           std::pow(diff_vector.norm(), 3) / model_sphere_radius;
  }

private:
  /**
   * Location of the Dirac point source \f$\delta(x-x_0)\f$.
   */
  Point<3> x0;
  Point<3> model_sphere_center;
  double   model_sphere_radius;
};


int
main(int argc, char *argv[])
{
  CmdOpts opts = parse_cmdline(argc, argv);

  // Write run-time logs to file
  std::ofstream ofs("dirichlet-full-matrix-single-thread.log");
  deallog.pop();
  deallog.depth_console(0);
  deallog.depth_file(5);
  deallog.attach(ofs);

  /**
   * @internal Pop out the default "DEAL" prefix string.
   */
  LogStream::Prefix prefix_string("HierBEM");

  Timer timer;

  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  LaplaceBEM<dim, spacedim> bem(
    opts.dirichlet_space_fe_order,
    opts.neumann_space_fe_order,
    LaplaceBEM<dim, spacedim>::ProblemType::DirichletBCProblem,
    true, // Interior problem
    MultithreadInfo::n_cores());
  bem.set_cpu_serial(!opts.run_in_parallel);

  timer.stop();
  print_wall_time(deallog, timer, "program preparation");

  timer.start();

  // Create the map from manifold ids to manifold objects. Because in the
  // destructor of LaplaceBEM the manifold objects will be released, the
  // manifold object here is created on the heap.
  const Point<3>           center(0, 0, 0);
  Manifold<dim, spacedim> *surface_manifold;
  if (opts.use_sphere_manifold)
    surface_manifold = new SphericalManifold<dim, spacedim>(center);
  else
    surface_manifold = new FlatManifold<dim, spacedim>();
  bem.get_manifolds()[0] = surface_manifold;

  // Create the map from manifold id to mapping order.
  bem.get_manifold_id_to_mapping_order()[0] = opts.mapping_order;

  if (opts.use_2d_mesh)
    {
      if (opts.use_occ)
        {
          // When OCC model is used, Gmsh API can be used to detect the
          // surface-to-subdomain topology.
          read_skeleton_mesh(opts.mesh_file, bem.get_triangulation());
          bem.get_subdomain_topology().generate_topology(opts.cad_file,
                                                         opts.mesh_file);

          // Create the map from material ids to manifold ids. When the sphere
          // is generated via OCC, there is only one surface entity with the
          // tag 1.
          bem.get_manifold_description()[1] = 0;
        }
      else
        {
          // Manually generate the subdomain topology, since
          // gmsh::model::isInside fails for 3D geo model. Entity tags from 21
          // to 28 constitute the sphere surface in the model. The volume entity
          // tag is 30.
          for (EntityTag e = 21; e <= 28; e++)
            {
              bem.get_subdomain_topology()
                .get_subdomain_to_surface()[30]
                .push_back(e);
              bem.get_subdomain_topology().get_surface_to_subdomain()[e] = {
                {0, 30}};
            }

          read_skeleton_mesh(opts.mesh_file, bem.get_triangulation());

          // Create the map from material ids to manifold ids.
          for (EntityTag e = 21; e <= 28; e++)
            {
              bem.get_manifold_description()[e] = 0;
            }
        }
    }
  else
    {
      // When 3D mesh is used, we need to extract its surface mesh using deal.ii
      // function, which will ensure normal vectors of all cells point outward.
      Triangulation<spacedim> volume_tria;

      if (opts.use_dealii_mesh)
        {
          GridGenerator::hyper_ball(volume_tria, center, 1.0);
          volume_tria.refine_global(opts.dealii_refinement);
        }
      else
        {
          GridIn<spacedim> grid_in(volume_tria);
          // There are two overloaded versions of GridIn::read_msh, only the one
          // accepting an file stream instead of a file name works here.
          std::ifstream grid_file(opts.mesh_file);
          grid_in.read_msh(grid_file);
          grid_file.close();
        }

      Triangulation<dim, spacedim> surface_tria;
      surface_tria.set_manifold(0, *surface_manifold);
      bem.extract_surface_triangulation(volume_tria,
                                        std::move(surface_tria),
                                        true);

      // Create the map from the only material id to manifold id.
      bem.get_manifold_description()[0] = 0;

      // Build surface-to-volume and volume-to-surface relationship.
      bem.get_subdomain_topology()
        .generate_single_domain_topology_for_dealii_model({0});
    }

  timer.stop();
  print_wall_time(deallog, timer, "read or generate mesh");

  timer.start();

  const Point<3> source_loc(1, 1, 1);
  DirichletBC    dirichlet_bc(source_loc);
  bem.assign_dirichlet_bc(dirichlet_bc);

  timer.stop();
  print_wall_time(deallog, timer, "assign boundary conditions");

  timer.start();

  bem.run();

  Vector<double> &numerical_solution = bem.get_neumann_data();
  // Interpolate the analytical solution, where we use the sphere manifold and
  // MappingManifold to obtain a stable and accurate vector.
  NeumannBC      analytical_solution_func(source_loc, center, 1.0);
  Vector<double> analytical_solution(bem.get_dof_handler_neumann().n_dofs());

  if (!opts.use_sphere_manifold)
    {
      bem.get_triangulation().set_manifold(
        0, SphericalManifold<dim, spacedim>(center));
    }

  VectorTools::interpolate(MappingManifold<dim, spacedim>(),
                           bem.get_dof_handler_neumann(),
                           analytical_solution_func,
                           analytical_solution);

  // Compute the L2 norm between the numerical and analytical solution.
  numerical_solution -= analytical_solution;
  deallog << "L2 error = "
          << numerical_solution.l2_norm() / analytical_solution.l2_norm()
          << std::endl;

  timer.stop();
  print_wall_time(deallog, timer, "run the solver");

  // Export the support points of shape functions in the Dirichlet DoF handler.
  std::vector<Point<spacedim>> fe_dirichlet_support_points(
    bem.get_dof_handler_dirichlet().n_dofs());
  DoFTools::map_dofs_to_support_points(
    bem.get_mappings()[opts.mapping_order - 1]->get_mapping(),
    bem.get_dof_handler_dirichlet(),
    fe_dirichlet_support_points);

  deallog << "=== fe dirichlet support points ===" << std::endl;
  print_vector_values(deallog.get_file_stream(),
                      fe_dirichlet_support_points,
                      "\n",
                      false);
  deallog << std::endl;

  // Export the support points of shape functions in the Neumann DoF handler.
  std::vector<Point<spacedim>> fe_neumann_support_points(
    bem.get_dof_handler_neumann().n_dofs());
  DoFTools::map_dofs_to_support_points(
    bem.get_mappings()[opts.mapping_order - 1]->get_mapping(),
    bem.get_dof_handler_neumann(),
    fe_neumann_support_points);

  deallog << "=== fe neumann support points ===" << std::endl;
  print_vector_values(deallog.get_file_stream(),
                      fe_neumann_support_points,
                      "\n",
                      false);
  deallog << std::endl;

  deallog << "Program exits with a total wall time " << timer.wall_time() << "s"
          << std::endl;

  bem.print_memory_consumption_table(deallog.get_file_stream());

  return 0;
}
