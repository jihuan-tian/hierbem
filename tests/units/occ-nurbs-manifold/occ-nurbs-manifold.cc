/**
 * @file occ-nurbs-manifold.cc
 * @brief
 *
 * @ingroup testers
 * @author
 * @date 2024-09-14
 */

#include <deal.II/base/point.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_manifold.h>

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <deal.II/opencascade/manifold_lib.h>

#include <boost/program_options.hpp>

#include <gmsh.h>

#include <filesystem>
#include <iostream>
#include <map>
#include <vector>

#include "gmsh_manipulation.h"
#include "grid_in_ext.h"
#include "hbem_test_config.h"
#include "mapping/mapping_q_generic_ext.h"

using namespace dealii;
using namespace HierBEM;
namespace po = boost::program_options;

enum ManifoldType
{
  FLAT,
  NORMAL_PROJECTION,
  NURBS_PATCH
};

enum MappingType
{
  // MappingQGenericExt
  LAGRANGE,
  // MappingManifold, i.e. manifold conforming
  MANIFOLD
};

struct CmdOpts
{
  unsigned int fe_order;
  ManifoldType manifold_type;
  MappingType  mapping_type;
  // When the @p MappingType is @p LAGRANGE, this specifies its order. If the
  // manifold type is FLAT and the mapping type is LAGRANGE, the mapping order
  // should be 1.
  unsigned int mapping_order;

  std::string cad_file;
  std::string mesh_file;

  // Whether convert the original surfaces to NURBS surfaces.
  bool convert_to_nurbs;

  // Tolerance for CAD based manifold.
  double tolerance;
};

CmdOpts
parse_cmdline(int argc, char *argv[])
{
  CmdOpts                 opts;
  po::options_description desc("Allowed options");

  // clang-format off
  desc.add_options()
    ("help,h", "show help message")
    ("fe-order,f", po::value<unsigned int>()->default_value(5), "Finite element space order")
    ("manifold-type,o", po::value<std::string>()->default_value("normal_projection"), "Manifold type: flat | nurbs_patch | normal_projection")
    ("mapping-type,p", po::value<std::string>()->default_value("lagrange"), "Mapping type: lagrange | manifold")
    ("mapping-order,m", po::value<unsigned int>()->default_value(2), "Mapping order for LAGRANGE mapping type")
    ("cad-file", po::value<std::string>(), "CAD file")
    ("mesh-file", po::value<std::string>(), "Mesh file")
    ("convert-to-nurbs", po::bool_switch(&opts.convert_to_nurbs), "Whether convert the original surfaces to NURBS surfaces")
    ("tolerance,t", po::value<double>()->default_value(1e-3), "Tolerance for creating CAD based manifold");
  // clang-format on

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help"))
    {
      std::cout << desc << std::endl;
      std::exit(EXIT_SUCCESS);
    }

  opts.fe_order = vm["fe-order"].as<unsigned int>();
  if (opts.fe_order == 0)
    {
      std::cerr << "Finite element order for FE_Q should be larger than 0!"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }

  const std::string manifold_type = vm["manifold-type"].as<std::string>();
  if (manifold_type == "flat")
    {
      opts.manifold_type = ManifoldType::FLAT;
    }
  else if (manifold_type == "nurbs_patch")
    {
      opts.manifold_type = ManifoldType::NURBS_PATCH;
    }
  else if (manifold_type == "normal_projection")
    {
      opts.manifold_type = ManifoldType::NORMAL_PROJECTION;
    }
  else
    {
      std::cerr << "Invalid manifold type: " << manifold_type << std::endl;
      std::exit(EXIT_FAILURE);
    }

  const std::string mapping_type = vm["mapping-type"].as<std::string>();
  if (mapping_type == "lagrange")
    {
      opts.mapping_type = MappingType::LAGRANGE;
    }
  else if (mapping_type == "manifold")
    {
      opts.mapping_type = MappingType::MANIFOLD;
    }
  else
    {
      std::cerr << "Invalid mapping type: " << mapping_type << std::endl;
      std::exit(EXIT_FAILURE);
    }

  opts.mapping_order = vm["mapping-order"].as<unsigned int>();
  if (opts.mapping_order == 0)
    {
      std::cerr
        << "Mapping order for MappingQGenericExt should be larger than 0!"
        << std::endl;
      std::exit(EXIT_FAILURE);
    }

  if (vm.count("cad-file"))
    {
      opts.cad_file = vm["cad-file"].as<std::string>();
    }
  else
    {
      opts.cad_file = HBEM_TEST_MODEL_DIR "two-spheres.brep";
    }

  if (!std::filesystem::exists(opts.cad_file))
    {
      std::cerr << "CAD file " << opts.cad_file << " does not exist!"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }

  if (vm.count("mesh-file"))
    {
      opts.mesh_file = vm["mesh-file"].as<std::string>();
    }
  else
    {
      opts.mesh_file = HBEM_TEST_MODEL_DIR "two-spheres.msh";
    }

  if (!std::filesystem::exists(opts.mesh_file))
    {
      std::cerr << "Mesh file " << opts.mesh_file << " does not exist!"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }

  opts.tolerance = vm["tolerance"].as<double>();

  return opts;
}

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

int
main(int argc, char *argv[])
{
  CmdOpts opts = parse_cmdline(argc, argv);

  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  // Read the mesh.
  Triangulation<dim, spacedim> tria;
  read_skeleton_mesh(opts.mesh_file, tria);

  // Initialize Gmsh.
  gmsh::initialize();
  gmsh::option::setNumber("General.Verbosity", 0);
  gmsh::open(opts.cad_file);
  gmsh::model::occ::synchronize();

  // Get all surface entities tags.
  gmsh::vectorpair surface_dimtags;
  gmsh::model::occ::getEntities(surface_dimtags, 2);

  // Manually associate manifold ids with material ids.
  std::map<EntityTag, types::manifold_id> manifold_description;
  for (unsigned int i = 0; i < surface_dimtags.size(); i++)
    {
      manifold_description[surface_dimtags[i].second] = i;
    }

  // Create and assign manifolds.
  std::map<types::manifold_id, Manifold<dim, spacedim> *> manifolds;
  if (opts.manifold_type == ManifoldType::FLAT)
    {
      for (auto &m : manifold_description)
        manifolds[m.second] = new FlatManifold<dim, spacedim>();
    }
  else
    {
      if (opts.convert_to_nurbs)
        {
          // Convert all surfaces to NURBS surfaces. By default, a surface
          // created in OpenCASCADE has its own parametric representation and
          // a local uv chart is assigned.
          gmsh::model::occ::convertToNURBS(surface_dimtags);
        }

      for (auto &m : manifold_description)
        {
          switch (opts.manifold_type)
            {
                case ManifoldType::NORMAL_PROJECTION: {
                  // Get the OpenCASCADE surface as a shape associated with
                  // the surface entity in Gmsh.
                  TopoDS_Shape shape =
                    gmsh::model::occ::getTopoDSShape(dim, m.first);
                  // With the quasi quad mesh algorithm, the vertices in the
                  // mesh are not exactly on the sphere, therefore the
                  // tolerance here is set to a larger value than the default
                  // value.
                  manifolds[m.second] =
                    new OpenCASCADE::NormalProjectionManifold<dim, spacedim>(
                      shape, opts.tolerance);
                  break;
                }
                case ManifoldType::NURBS_PATCH: {
                  // Get the OpenCASCADE surface as a face associated with
                  // the surface entity in Gmsh.
                  TopoDS_Face face = gmsh::model::occ::getTopoDSFace(m.first);
                  manifolds[m.second] =
                    new OpenCASCADE::NURBSPatchManifold<dim, spacedim>(
                      face, opts.tolerance);
                  break;
                }
                default: {
                  std::cerr << "Invalid manifold type: " << opts.manifold_type
                            << std::endl;
                  std::exit(EXIT_FAILURE);
                }
            }
        }
    }

  initialize_manifolds_from_manifold_description(tria,
                                                 manifold_description,
                                                 manifolds);

  // Create the mapping.
  Mapping<dim, spacedim> *mapping;
  switch (opts.mapping_type)
    {
        case MappingType::LAGRANGE: {
          mapping = new MappingQGenericExt<dim, spacedim>(opts.mapping_order);
          break;
        }
        case MappingType::MANIFOLD: {
          mapping = new MappingManifold<dim, spacedim>();
          break;
        }
        default: {
          std::cerr << "Invalid mapping type: " << opts.mapping_type
                    << std::endl;
          std::exit(EXIT_FAILURE);
        }
    }

  // Create high order finite element.
  DoFHandler<dim, spacedim> dof_handler(tria);
  FE_Q<dim, spacedim>       fe(opts.fe_order);
  dof_handler.distribute_dofs(fe);

  // Generate support points of the finite element in real cells.
  std::vector<Point<spacedim>> fe_support_points(dof_handler.n_dofs());
  DoFTools::map_dofs_to_support_points(*mapping,
                                       dof_handler,
                                       fe_support_points);

  for (auto &p : fe_support_points)
    {
      std::cout << p << "\n";
    }
  std::cout << std::endl;

  dof_handler.clear();

  // Destroy all manifolds and mapping.
  for (auto &m : manifolds)
    {
      delete m.second;
    }

  delete mapping;

  // Finalize Gmsh.
  gmsh::clear();
  gmsh::finalize();
}
