/**
 * @file ddm_efield.h
 * @brief Introduction of ddm_efield.h
 *
 * @date 2024-07-26
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_ELECTRIC_FIELD_DDM_EFIELD_H_
#define HIERBEM_INCLUDE_ELECTRIC_FIELD_DDM_EFIELD_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/patterns.h>
#include <deal.II/base/point.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_data.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_description.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_interpolate.templates.h>

#include <gmsh.h>

#include <array>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "config.h"
#include "ddm_efield_global_preconditioner.h"
#include "ddm_efield_matrix.h"
#include "dof_to_cell_topology.h"
#include "dof_tools_ext.h"
#include "generic_functors.h"
#include "grid_out_ext.h"
#include "hmatrix/aca_plus/aca_config.h"
#include "mapping/mapping_info.h"
#include "subdomain_steklov_poincare_hmatrix.h"
#include "subdomain_topology.h"
#include "unary_template_arg_containers.h"
#include "platform_shared/laplace_kernels.h"

HBEM_NS_OPEN

using namespace dealii;

enum SubdomainType
{
  SurroundingSpace,
  Dielectric,
  VoltageConductor,
  FloatingConductor
};

template <int spacedim>
class EfieldSubdomain;

template <int spacedim>
class EfieldSurface
{
public:
  EfieldSurface(const EntityTag             entity_tag,
                EfieldSurface<spacedim>    *neighbor_surface,
                EfieldSubdomain<spacedim>  *parent_subdomain,
                const bool                  is_normal_outward,
                const bool                  is_dirichlet_boundary,
                Function<spacedim, double> *dirichlet_voltage);

  EfieldSurface(const EfieldSurface &surface) = default;

  EfieldSurface &
  operator=(const EfieldSurface &surface) = default;

  Function<spacedim, double> *
  get_dirichlet_voltage()
  {
    return dirichlet_voltage;
  }

  void
  set_dirichlet_voltage(Function<spacedim, double> *dirichletVoltage)
  {
    dirichlet_voltage = dirichletVoltage;
  }

  EntityTag
  get_entity_tag() const
  {
    return entity_tag;
  }

  void
  set_entity_tag(const EntityTag entityTag)
  {
    entity_tag = entityTag;
  }

  bool
  get_is_dirichlet_boundary() const
  {
    return is_dirichlet_boundary;
  }

  void
  set_is_dirichlet_boundary(const bool isDirichletBoundary)
  {
    is_dirichlet_boundary = isDirichletBoundary;
  }

  bool
  get_is_normal_outward() const
  {
    return is_normal_outward;
  }

  void
  set_is_normal_outward(const bool isNormalOutward)
  {
    is_normal_outward = isNormalOutward;
  }

  EfieldSurface<spacedim> *
  get_neighbor_surface()
  {
    return neighbor_surface;
  }

  void
  set_neighbor_surface(EfieldSurface<spacedim> *neighborSurface)
  {
    neighbor_surface = neighborSurface;
  }

  EfieldSubdomain<spacedim> *
  get_parent_subdomain()
  {
    return parent_subdomain;
  }

  void
  set_parent_subdomain(EfieldSubdomain<spacedim> *parentSubdomain)
  {
    parent_subdomain = parentSubdomain;
  }

private:
  EntityTag                   entity_tag;
  EfieldSurface<spacedim>    *neighbor_surface;
  EfieldSubdomain<spacedim>  *parent_subdomain;
  bool                        is_normal_outward;
  bool                        is_dirichlet_boundary;
  Function<spacedim, double> *dirichlet_voltage;
};


template <int spacedim>
EfieldSurface<spacedim>::EfieldSurface(
  const EntityTag             entity_tag,
  EfieldSurface<spacedim>    *neighbor_surface,
  EfieldSubdomain<spacedim>  *parent_subdomain,
  const bool                  is_normal_outward,
  const bool                  is_dirichlet_boundary,
  Function<spacedim, double> *dirichlet_voltage)
  : entity_tag(entity_tag)
  , neighbor_surface(neighbor_surface)
  , parent_subdomain(parent_subdomain)
  , is_normal_outward(is_normal_outward)
  , is_dirichlet_boundary(is_dirichlet_boundary)
  , dirichlet_voltage(dirichlet_voltage)
{}


/**
 * A subdomain in electric field problem
 */
template <int spacedim>
class EfieldSubdomain
{
public:
  EfieldSubdomain() = default;

  EfieldSubdomain(const EntityTag     entity_tag,
                  const SubdomainType type,
                  const double        permittivity,
                  const double        voltage);

  EfieldSubdomain(const EfieldSubdomain &subdomain) = default;

  EfieldSubdomain &
  operator=(const EfieldSubdomain &subdomain) = default;

  const std::vector<EfieldSurface<spacedim>> &
  get_surfaces_touching_dielectric() const
  {
    return surfaces_touching_dielectric;
  }

  const std::vector<EfieldSurface<spacedim>> &
  get_surfaces_touching_floating_conductor() const
  {
    return surfaces_touching_floating_conductor;
  }

  const std::vector<EfieldSurface<spacedim>> &
  get_surfaces_touching_voltage_conductor() const
  {
    return surfaces_touching_voltage_conductor;
  }

  std::vector<EfieldSurface<spacedim>> &
  get_surfaces_touching_dielectric()
  {
    return surfaces_touching_dielectric;
  }

  std::vector<EfieldSurface<spacedim>> &
  get_surfaces_touching_floating_conductor()
  {
    return surfaces_touching_floating_conductor;
  }

  std::vector<EfieldSurface<spacedim>> &
  get_surfaces_touching_voltage_conductor()
  {
    return surfaces_touching_voltage_conductor;
  }

  EntityTag
  get_entity_tag() const
  {
    return entity_tag;
  }

  double
  get_permittivity() const
  {
    return permittivity;
  }

  SubdomainType
  get_type() const
  {
    return type;
  }

  double
  get_voltage() const
  {
    return voltage;
  }

private:
  EntityTag                            entity_tag;
  SubdomainType                        type;
  double                               permittivity;
  double                               voltage;
  std::vector<EfieldSurface<spacedim>> surfaces_touching_dielectric;
  std::vector<EfieldSurface<spacedim>> surfaces_touching_voltage_conductor;
  std::vector<EfieldSurface<spacedim>> surfaces_touching_floating_conductor;
};


template <int spacedim>
EfieldSubdomain<spacedim>::EfieldSubdomain(const EntityTag     entity_tag,
                                           const SubdomainType type,
                                           const double        permittivity,
                                           const double        voltage)
  : entity_tag(entity_tag)
  , type(type)
  , permittivity(permittivity)
  , voltage(voltage)
{}


template <int spacedim>
class EfieldSubdomainDescription
{
public:
  EfieldSubdomainDescription() = default;

  const std::map<EntityTag, EfieldSubdomain<spacedim>> &
  get_subdomains() const
  {
    return subdomains;
  }

  const std::vector<EfieldSubdomain<spacedim> *> &
  get_dielectric_subdomains() const
  {
    return dielectric_subdomains;
  }

  const std::vector<EfieldSubdomain<spacedim> *> &
  get_floating_conductor_subdomains() const
  {
    return floating_conductor_subdomains;
  }

  const std::vector<EfieldSubdomain<spacedim> *> &
  get_voltage_conductor_subdomains() const
  {
    return voltage_conductor_subdomains;
  }

  std::vector<EfieldSubdomain<spacedim> *> &
  get_dielectric_subdomains()
  {
    return dielectric_subdomains;
  }

  std::vector<EfieldSubdomain<spacedim> *> &
  get_floating_conductor_subdomains()
  {
    return floating_conductor_subdomains;
  }

  std::vector<EfieldSubdomain<spacedim> *> &
  get_voltage_conductor_subdomains()
  {
    return voltage_conductor_subdomains;
  }

  std::map<EntityTag, EfieldSubdomain<spacedim>> &
  get_subdomains()
  {
    return subdomains;
  }

private:
  std::map<EntityTag, EfieldSubdomain<spacedim>> subdomains;
  std::vector<EfieldSubdomain<spacedim> *>       dielectric_subdomains;
  std::vector<EfieldSubdomain<spacedim> *>       voltage_conductor_subdomains;
  std::vector<EfieldSubdomain<spacedim> *>       floating_conductor_subdomains;
};

template <int dim, int spacedim>
class DDMEfield
{
public:
  /**
   * Maximum mapping order.
   */
  inline static const unsigned int max_mapping_order = 3;

  DDMEfield();

  DDMEfield(const unsigned int fe_order_for_dirichlet_space,
            const unsigned int fe_order_for_neumann_space,
            const unsigned int n_min_for_ct,
            const unsigned int n_min_for_bct,
            const double       eta,
            const unsigned int max_hmat_rank,
            const double       aca_relative_error,
            const double       eta_for_preconditioner,
            const unsigned int max_hmat_rank_for_preconditioner,
            const double       aca_relative_error_for_preconditioner,
            const unsigned int thread_num);

  ~DDMEfield();

  /**
   * Read the CAD file and build the association relationship between volumes
   * and surfaces.
   *
   * @pre
   * @post
   * @param cad_file
   */
  void
  read_subdomain_topology(const std::string &cad_file,
                          const std::string &mesh_file);

  /**
   * Manually initialize problem parameters for testing purpose.
   *
   * @pre
   * @post
   */
  void
  initialize_parameters();

  void
  initialize_manifolds_and_mappings();

  /**
   * Interpolate Dirichlet boundary conditions on dielectric surfaces or
   * interfaces.
   *
   * @pre
   * @post
   */
  void
  interpolate_surface_dirichlet_bc();

  void
  create_efield_subdomains_and_surfaces();

  /**
   * Initialize \hmats for each dielectric subdomain.
   * @pre
   * @post
   */
  void
  initialize_subdomain_hmatrices();

  void
  generate_cell_iterators();

  void
  setup_system();

  void
  assemble_system();

  void
  assemble_preconditioner();

  void
  solve();

  void
  output_results();

  const SubdomainTopology<dim, spacedim> &
  get_subdomain_topology() const
  {
    return subdomain_topology;
  }

  SubdomainTopology<dim, spacedim> &
  get_subdomain_topology()
  {
    return subdomain_topology;
  }

  const Triangulation<dim, spacedim> &
  get_triangulation() const
  {
    return tria;
  }

  Triangulation<dim, spacedim> &
  get_triangulation()
  {
    return tria;
  }

private:
  SubdomainTopology<dim, spacedim>     subdomain_topology;
  EfieldSubdomainDescription<spacedim> domain;
  Triangulation<dim, spacedim>         tria;
  /**
   * A list of mapping objects from 1st to 3rd order.
   */
  std::vector<MappingInfo<dim, spacedim> *> mappings;

  /**
   * Kernel function for the single layer potential.
   */
  HierBEM::PlatformShared::LaplaceKernel::SingleLayerKernel<3>
    single_layer_kernel;
  /**
   * Kernel function for the double layer potential.
   */
  HierBEM::PlatformShared::LaplaceKernel::DoubleLayerKernel<3>
    double_layer_kernel;
  /**
   * Kernel function for the hyper-singular potential.
   */
  HierBEM::PlatformShared::LaplaceKernel::HyperSingularKernelRegular<3>
    hyper_singular_kernel;

  /**
   * Dirichlet space on the whole skeleton.
   */
  DoFHandler<dim, spacedim> dof_handler_for_dirichlet_space;
  /**
   * Neumann space on the whole skeleton.
   */
  DoFHandler<dim, spacedim> dof_handler_for_neumann_space;

  std::string project_name;

  /**
   * Finite element order for the Dirichlet space.
   */
  unsigned int fe_order_for_dirichlet_space;

  /**
   * Finite element order for the Neumann space.
   */
  unsigned int fe_order_for_neumann_space;

  /**
   * Finite element \f$H^{\frac{1}{2}+s}\f$ for the Dirichlet space. At
   * present, it is implemented as a continuous Lagrange space.
   */
  FE_Q<dim, spacedim> fe_for_dirichlet_space;
  /**
   * Finite element \f$H^{-\frac{1}{2}+s}\f$ for the Neumann space. At
   * present, it is implemented as a discontinuous Lagrange space.
   */
  FE_DGQ<dim, spacedim> fe_for_neumann_space;

  std::vector<typename DoFHandler<dim, spacedim>::cell_iterator>
    cell_iterators_for_dirichlet_space;
  std::vector<typename DoFHandler<dim, spacedim>::cell_iterator>
    cell_iterators_for_neumann_space;

  /**
   * DoF-to-cell topologies for various DoF handlers, which are used for
   * matrix assembly on a pair of DoFs.
   */
  DofToCellTopology<dim, spacedim> dof_to_cell_topo_for_dirichlet_space;
  DofToCellTopology<dim, spacedim> dof_to_cell_topo_for_neumann_space;

  /**
   * Minimum cluster size. At present, assume all \bcts share this same
   * parameter.
   */
  unsigned int n_min_for_ct;
  /**
   * Minimum block cluster size. At present, assume all \bcts share this
   * same parameter.
   */
  unsigned int n_min_for_bct;
  /**
   * Admissibility constant. At present, assume all \bcts share this same
   * parameter.
   */
  double eta;
  /**
   * Maximum rank of the \hmatrices to be built. At present, assume all
   * \hmatrices share this same parameter.
   */
  unsigned int max_hmat_rank;
  /**
   * Relative approximation error used in ACA+. At present, assume all
   * \hmatrices share this same parameter.
   */
  double aca_relative_error;
  /**
   * Admissibility constant for the preconditioner.
   */
  double eta_for_preconditioner;
  /**
   * Maximum rank of the \hmatrices to be built for the preconditioner.
   */
  unsigned int max_hmat_rank_for_preconditioner;
  /**
   * Relative approximation error used in ACA+ for the preconditioner.
   */
  double aca_relative_error_for_preconditioner;

  unsigned int thread_num;

  DDMEfieldMatrix<spacedim, double>               system_matrix;
  DDMEfieldGlobalPreconditioner<spacedim, double> system_preconditioner;
  Vector<double>                                  rhs_for_transmission_eqn;
  Vector<double>                                  rhs_for_charge_neutrality_eqn;

  /**
   * Dirichlet boundary condition data on all DoFs in the associated DoF
   * handler.
   */
  Vector<double> dirichlet_bc;

  /**
   * Map volume entity tag to subdomain type.
   */
  std::map<EntityTag, SubdomainType> subdomain_types;
  /**
   * Map volume entity tag to permittivity.
   */
  std::map<EntityTag, double> permittivities;
  /**
   * Map volume entity tag to voltages of conductors.
   */
  std::map<EntityTag, double> conductor_voltages;
  /**
   * Map surface entity tag to Dirichlet boundary condition.
   */
  std::map<EntityTag, Function<spacedim, double> *>
    dirichlet_boundary_conditions;
  /**
   * Map surface entity tag to manifold id. At the moment, the material for
   * each surface is the same as the entity tag in Gmsh.
   */
  std::map<EntityTag, types::manifold_id> manifold_description;
  /**
   * Map @p manifold_id to the pointer of a Manifold object.
   */
  std::map<types::manifold_id, Manifold<dim, spacedim> *> manifolds;
  /**
   * Map @p manifold_id to mapping order.
   */
  std::map<types::manifold_id, unsigned int> manifold_id_to_mapping_order;
};


template <int dim, int spacedim>
DDMEfield<dim, spacedim>::DDMEfield()
  : project_name("default")
  , fe_order_for_dirichlet_space(1)
  , fe_order_for_neumann_space(0)
  , fe_for_dirichlet_space(fe_order_for_dirichlet_space)
  , fe_for_neumann_space(fe_order_for_neumann_space)
  , n_min_for_ct(0)
  , n_min_for_bct(0)
  , eta(0)
  , max_hmat_rank(0)
  , aca_relative_error(0)
  , eta_for_preconditioner(0)
  , max_hmat_rank_for_preconditioner(0)
  , aca_relative_error_for_preconditioner(0)
  , thread_num(0)
{}


template <int dim, int spacedim>
DDMEfield<dim, spacedim>::DDMEfield(
  const unsigned int fe_order_for_dirichlet_space,
  const unsigned int fe_order_for_neumann_space,
  const unsigned int n_min_for_ct,
  const unsigned int n_min_for_bct,
  const double       eta,
  const unsigned int max_hmat_rank,
  const double       aca_relative_error,
  const double       eta_for_preconditioner,
  const unsigned int max_hmat_rank_for_preconditioner,
  const double       aca_relative_error_for_preconditioner,
  const unsigned int thread_num)
  : project_name("default")
  , fe_order_for_dirichlet_space(fe_order_for_dirichlet_space)
  , fe_order_for_neumann_space(fe_order_for_neumann_space)
  , fe_for_dirichlet_space(fe_order_for_dirichlet_space)
  , fe_for_neumann_space(fe_order_for_neumann_space)
  , n_min_for_ct(n_min_for_ct)
  , n_min_for_bct(n_min_for_bct)
  , eta(eta)
  , max_hmat_rank(max_hmat_rank)
  , aca_relative_error(aca_relative_error)
  , eta_for_preconditioner(eta_for_preconditioner)
  , max_hmat_rank_for_preconditioner(max_hmat_rank_for_preconditioner)
  , aca_relative_error_for_preconditioner(aca_relative_error_for_preconditioner)
  , thread_num(thread_num)
{}


template <int dim, int spacedim>
DDMEfield<dim, spacedim>::~DDMEfield()
{
  // Release function objects defining Dirichlet boundary conditions on
  // dielectric surfaces or interfaces.
  for (auto &d : dirichlet_boundary_conditions)
    {
      delete d.second;
    }

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
}


template <int dim, int spacedim>
void
DDMEfield<dim, spacedim>::read_subdomain_topology(const std::string &cad_file,
                                                  const std::string &mesh_file)
{
  subdomain_topology.generate_topology(cad_file, mesh_file);
}


template <int dim, int spacedim>
void
DDMEfield<dim, spacedim>::initialize_parameters()
{
  subdomain_types[0] = SubdomainType::SurroundingSpace;
  subdomain_types[1] = SubdomainType::VoltageConductor;
  subdomain_types[2] = SubdomainType::Dielectric;
  subdomain_types[3] = SubdomainType::Dielectric;

  permittivities[0] = 1;
  permittivities[2] = 2;
  permittivities[3] = 4;

  // Assign Dirichlet boundary condition on dielectric boundary or interface.
  conductor_voltages[1] = 10;

  std::string                   symbolic_variables = "x,y,z";
  std::map<std::string, double> symbolic_constants;
  symbolic_constants["pi"] = numbers::PI;
  symbolic_constants["e"]  = numbers::E;

  dirichlet_boundary_conditions[11] = new FunctionParser<spacedim>(1);
  std::string expr                  = "sin(x) * cos(z)";
  static_cast<FunctionParser<spacedim> *>(dirichlet_boundary_conditions[11])
    ->initialize(symbolic_variables, expr, symbolic_constants);

  dirichlet_boundary_conditions[6] = new FunctionParser<spacedim>(1);
  expr                             = "cos(x) * sin(z)";
  static_cast<FunctionParser<spacedim> *>(dirichlet_boundary_conditions[6])
    ->initialize(symbolic_variables, expr, symbolic_constants);

  // Spherical manifold
  manifold_description[1] = 1;
  manifold_description[2] = 1;

  // Flat manifold
  for (unsigned int i = 3; i <= 13; i++)
    {
      manifold_description[i] = 0;
    }

  // We use the first order mapping for the flat manifold and the second order
  // mapping for the spherical manifold.
  manifold_id_to_mapping_order[0] = 1;
  manifold_id_to_mapping_order[1] = 2;
}


template <int dim, int spacedim>
void
DDMEfield<dim, spacedim>::initialize_manifolds_and_mappings()
{
  // Assign manifold ids to all cells in the triangulation. Because there are
  // no physical groups defined, the manifold ids cannot be read from the MSH
  // file and have to be initialized here.
  for (auto &cell : tria.active_cell_iterators())
    {
      cell->set_all_manifold_ids(manifold_description[cell->material_id()]);
    }

  // Create and associate manifold objects.
  for (const auto &desc : manifold_description)
    {
      switch (desc.second)
        {
            case 1: {
              typename decltype(manifolds)::iterator pos =
                manifolds.find(desc.second);

              if (pos == manifolds.end())
                {
                  // Define a 2D spherical manifold centered at the origin, if
                  // it has not been defined before.
                  manifolds[desc.second] = new SphericalManifold<dim, spacedim>(
                    Point<spacedim>(0., 0., 0.));
                  tria.set_manifold(desc.second, *manifolds[desc.second]);
                }

              break;
            }
        }
    }

  // Initialize MappingInfo objects from 1st to 3rd order.
  mappings.reserve(max_mapping_order);
  for (unsigned int i = 1; i <= max_mapping_order; i++)
    {
      mappings.push_back(new MappingInfo<dim, spacedim>(i));
    }
}


template <int dim, int spacedim>
void
DDMEfield<dim, spacedim>::interpolate_surface_dirichlet_bc()
{
  dirichlet_bc.reinit(dof_handler_for_dirichlet_space.n_dofs());

  // Because each surface may be assigned a different mapping object, here we
  // interpolate the Dirichlet boundary condition vector surface by surface.
  for (const auto &bc : dirichlet_boundary_conditions)
    {
      std::map<types::material_id, const Function<spacedim, double> *>
        single_pair_map;
      single_pair_map[static_cast<types::material_id>(bc.first)] = bc.second;

      VectorTools::interpolate_based_on_material_id(
        mappings[manifold_id_to_mapping_order[manifold_description[bc.first]] -
                 1]
          ->get_mapping(),
        dof_handler_for_dirichlet_space,
        single_pair_map,
        dirichlet_bc);
    }
}


template <int dim, int spacedim>
void
DDMEfield<dim, spacedim>::create_efield_subdomains_and_surfaces()
{
  // Create the default surrounding space subdomain (e.g. air domain) and add
  // it as the first dielectric subdomain in the domain description object.
  domain.get_subdomains()[0] = EfieldSubdomain<spacedim>(
    0, SubdomainType::SurroundingSpace, permittivities[0], 0);
  domain.get_dielectric_subdomains().push_back(&domain.get_subdomains()[0]);

  // Create each subdomain.
  for (const auto &record : subdomain_topology.get_subdomain_to_surface())
    {
      const EntityTag     entity_tag = record.first;
      const SubdomainType type       = subdomain_types[record.first];
      const double        permittivity =
        (type == SubdomainType::Dielectric) ? permittivities[entity_tag] : 0;
      const double voltage = (type == SubdomainType::VoltageConductor) ?
                               conductor_voltages[entity_tag] :
                               0;

      domain.get_subdomains()[entity_tag] =
        EfieldSubdomain<spacedim>(entity_tag, type, permittivity, voltage);

      // Add the subdomain to corresponding list in the domain description
      // object.
      switch (type)
        {
          case (SubdomainType::Dielectric):
            domain.get_dielectric_subdomains().push_back(
              &domain.get_subdomains()[entity_tag]);
            break;
          case (SubdomainType::VoltageConductor):
            domain.get_voltage_conductor_subdomains().push_back(
              &domain.get_subdomains()[entity_tag]);
            break;
          case (SubdomainType::FloatingConductor):
            domain.get_floating_conductor_subdomains().push_back(
              &domain.get_subdomains()[entity_tag]);
            break;
          default:
            Assert(false, ExcInternalError());
            break;
        }
    }

  // Create each surface. For a same geometric surface, there are actually two
  // @p EfieldSurface objects are created and associated.
  for (const auto &record : subdomain_topology.get_surface_to_subdomain())
    {
      // Determine if the current surface is assigned a Dirichlet boundary
      // condition.
      typename std::map<EntityTag, Function<spacedim, double> *>::iterator pos =
        dirichlet_boundary_conditions.find(record.first);
      bool                        is_dirichlet_surface;
      Function<spacedim, double> *dirichlet_voltage;

      if (pos != dirichlet_boundary_conditions.end())
        {
          is_dirichlet_surface = true;
          dirichlet_voltage    = pos->second;
        }
      else
        {
          is_dirichlet_surface = false;
          dirichlet_voltage    = nullptr;
        }

      // Get the two subdomains sharing the current surface.
      EfieldSubdomain<spacedim> &subdomain_surface_normal_point_from =
        domain.get_subdomains()[record.second[0]];
      EfieldSubdomain<spacedim> &subdomain_surface_normal_point_to =
        domain.get_subdomains()[record.second[1]];

      EfieldSurface<spacedim> *surface_of_from_subdomain;
      EfieldSurface<spacedim> *surface_of_to_subdomain;

      // Create a surface object for the "from" subdomain.
      // Here we check its neighboring subdomain type and add the surface to
      // corresponding list in the current subdomain.
      switch (subdomain_surface_normal_point_to.get_type())
        {
          case SubdomainType::SurroundingSpace:
            case SubdomainType::Dielectric: {
              subdomain_surface_normal_point_from
                .get_surfaces_touching_dielectric()
                .emplace_back(record.first,
                              nullptr, // Temporarily, the neighbor surface is
                                       // not connected.
                              &subdomain_surface_normal_point_from,
                              true,
                              is_dirichlet_surface,
                              dirichlet_voltage);
              surface_of_from_subdomain = &subdomain_surface_normal_point_from
                                             .get_surfaces_touching_dielectric()
                                             .back();
              break;
            }
            case SubdomainType::VoltageConductor: {
              subdomain_surface_normal_point_from
                .get_surfaces_touching_voltage_conductor()
                .emplace_back(record.first,
                              nullptr, // Temporarily, the neighbor surface is
                                       // not connected.
                              &subdomain_surface_normal_point_from,
                              true,
                              is_dirichlet_surface,
                              dirichlet_voltage);
              surface_of_from_subdomain =
                &subdomain_surface_normal_point_from
                   .get_surfaces_touching_voltage_conductor()
                   .back();
              break;
            }
            case SubdomainType::FloatingConductor: {
              subdomain_surface_normal_point_from
                .get_surfaces_touching_floating_conductor()
                .emplace_back(record.first,
                              nullptr, // Temporarily, the neighbor surface is
                                       // not connected.
                              &subdomain_surface_normal_point_from,
                              true,
                              is_dirichlet_surface,
                              dirichlet_voltage);
              surface_of_from_subdomain =
                &subdomain_surface_normal_point_from
                   .get_surfaces_touching_floating_conductor()
                   .back();
              break;
            }
            default: {
              surface_of_from_subdomain = nullptr;
              Assert(false, ExcInternalError());
            }
        }

      // Create a surface object for the "to" subdomain.
      // Here we check its neighboring subdomain type and add the surface to
      // corresponding list in the current subdomain.
      switch (subdomain_surface_normal_point_from.get_type())
        {
          case SubdomainType::SurroundingSpace:
            case SubdomainType::Dielectric: {
              subdomain_surface_normal_point_to
                .get_surfaces_touching_dielectric()
                .emplace_back(record.first,
                              nullptr, // Temporarily, the neighbor surface is
                                       // not connected.
                              &subdomain_surface_normal_point_to,
                              false,
                              is_dirichlet_surface,
                              dirichlet_voltage);
              surface_of_to_subdomain = &subdomain_surface_normal_point_to
                                           .get_surfaces_touching_dielectric()
                                           .back();
              break;
            }
            case SubdomainType::VoltageConductor: {
              subdomain_surface_normal_point_to
                .get_surfaces_touching_voltage_conductor()
                .emplace_back(record.first,
                              nullptr, // Temporarily, the neighbor surface is
                                       // not connected.
                              &subdomain_surface_normal_point_to,
                              false,
                              is_dirichlet_surface,
                              dirichlet_voltage);
              surface_of_to_subdomain =
                &subdomain_surface_normal_point_to
                   .get_surfaces_touching_voltage_conductor()
                   .back();
              break;
            }
            case SubdomainType::FloatingConductor: {
              subdomain_surface_normal_point_to
                .get_surfaces_touching_floating_conductor()
                .emplace_back(record.first,
                              nullptr, // Temporarily, the neighbor surface is
                                       // not connected.
                              &subdomain_surface_normal_point_to,
                              false,
                              is_dirichlet_surface,
                              dirichlet_voltage);
              surface_of_to_subdomain =
                &subdomain_surface_normal_point_to
                   .get_surfaces_touching_floating_conductor()
                   .back();
              break;
            }
            default: {
              surface_of_to_subdomain = nullptr;
              Assert(false, ExcInternalError());
            }
        }

      // Connect neighboring surfaces.
      surface_of_from_subdomain->set_neighbor_surface(surface_of_to_subdomain);
      surface_of_to_subdomain->set_neighbor_surface(surface_of_from_subdomain);
    }
}


template <int dim, int spacedim>
void
DDMEfield<dim, spacedim>::initialize_subdomain_hmatrices()
{
  auto &subdomain_hmatrices = system_matrix.get_subdomain_hmatrices();
  // TODO 2024-08-10 At the moment, the memory should be reserved for this
  // vector to prevent std::vector copying or moving data, because this class
  // has no copy or move-copy constructor defined yet.
  subdomain_hmatrices.reserve(domain.get_dielectric_subdomains().size());

  const unsigned int n_dofs_for_dirichlet_space =
    dof_handler_for_dirichlet_space.n_dofs();
  const unsigned int n_dofs_for_neumann_space =
    dof_handler_for_neumann_space.n_dofs();

  // Generate selectors for Dirichlet space on non-Dirichlet boundary.
  std::vector<bool>
    negated_dof_selectors_for_dirichlet_space_on_nondirichlet_boundary;
  negated_dof_selectors_for_dirichlet_space_on_nondirichlet_boundary.resize(
    dof_handler_for_dirichlet_space.n_dofs());

  std::set<types::material_id> material_ids;
  for (const auto subdomain : domain.get_voltage_conductor_subdomains())
    {
      for (const auto &surface : subdomain->get_surfaces_touching_dielectric())
        {
          material_ids.insert(
            static_cast<types::material_id>(surface.get_entity_tag()));
        }
    }

  for (const auto subdomain : domain.get_floating_conductor_subdomains())
    {
      for (const auto &surface : subdomain->get_surfaces_touching_dielectric())
        {
          material_ids.insert(
            static_cast<types::material_id>(surface.get_entity_tag()));
        }
    }

  for (const auto subdomain : domain.get_dielectric_subdomains())
    {
      for (const auto &surface : subdomain->get_surfaces_touching_dielectric())
        {
          if (surface.get_is_dirichlet_boundary())
            {
              material_ids.insert(
                static_cast<types::material_id>(surface.get_entity_tag()));
            }
        }
    }

  DoFToolsExt::extract_material_domain_dofs(
    dof_handler_for_dirichlet_space,
    material_ids,
    negated_dof_selectors_for_dirichlet_space_on_nondirichlet_boundary);

  system_matrix
    .build_local_to_global_dirichlet_dof_map_and_inverse_on_nondirichlet_boundary(
      dof_handler_for_dirichlet_space,
      negated_dof_selectors_for_dirichlet_space_on_nondirichlet_boundary);

  std::cout
    << "Number of DoFs in local Dirichlet space restricted to non-Dirichlet boundary: "
    << system_matrix
         .get_nondirichlet_boundary_to_skeleton_dirichlet_dof_index_map()
         .size()
    << std::endl;

  /**
   * Build the DoF-to-cell topology.
   *
   * \mynote{Access of this topology for the Dirichlet space
   * requires the map from local to full DoF indices.}
   */
  generate_cell_iterators();
  build_dof_to_cell_topology(dof_to_cell_topo_for_dirichlet_space,
                             cell_iterators_for_dirichlet_space,
                             dof_handler_for_dirichlet_space);
  build_dof_to_cell_topology(dof_to_cell_topo_for_neumann_space,
                             cell_iterators_for_neumann_space,
                             dof_handler_for_neumann_space);

  // Iterate over each dielectric subdomain.
  for (const auto subdomain : domain.get_dielectric_subdomains())
    {
      // Create an empty Steklov_poincare \hmat for the current subdomain.
      subdomain_hmatrices.emplace_back();

      // Generate selectors for Dirichlet space.
      std::vector<bool> dof_selectors_for_dirichlet_space;
      dof_selectors_for_dirichlet_space.resize(
        dof_handler_for_dirichlet_space.n_dofs());

      material_ids.clear();
      for (const auto &surface : subdomain->get_surfaces_touching_dielectric())
        {
          material_ids.insert(
            static_cast<types::material_id>(surface.get_entity_tag()));
        }

      for (const auto &surface :
           subdomain->get_surfaces_touching_floating_conductor())
        {
          material_ids.insert(
            static_cast<types::material_id>(surface.get_entity_tag()));
        }

      for (const auto &surface :
           subdomain->get_surfaces_touching_voltage_conductor())
        {
          material_ids.insert(
            static_cast<types::material_id>(surface.get_entity_tag()));
        }

      DoFToolsExt::extract_material_domain_dofs(
        dof_handler_for_dirichlet_space,
        material_ids,
        dof_selectors_for_dirichlet_space);

      // Generate selectors for Neumann space.
      std::vector<bool> dof_selectors_for_neumann_space;
      dof_selectors_for_neumann_space.resize(
        dof_handler_for_neumann_space.n_dofs());

      material_ids.clear();
      for (const auto &surface : subdomain->get_surfaces_touching_dielectric())
        {
          material_ids.insert(
            static_cast<types::material_id>(surface.get_entity_tag()));
        }

      for (const auto &surface :
           subdomain->get_surfaces_touching_floating_conductor())
        {
          material_ids.insert(
            static_cast<types::material_id>(surface.get_entity_tag()));
        }

      for (const auto &surface :
           subdomain->get_surfaces_touching_voltage_conductor())
        {
          material_ids.insert(
            static_cast<types::material_id>(surface.get_entity_tag()));
        }

      DoFToolsExt::extract_material_domain_dofs(
        dof_handler_for_neumann_space,
        material_ids,
        dof_selectors_for_neumann_space);

      SubdomainSteklovPoincareHMatrix<spacedim, double> &steklov_poincare_hmat =
        subdomain_hmatrices.back();

      steklov_poincare_hmat.build_local_to_global_dof_maps_and_inverses(
        dof_handler_for_dirichlet_space,
        dof_handler_for_neumann_space,
        dof_selectors_for_dirichlet_space,
        dof_selectors_for_neumann_space);

      // Get the number of effective DoF number for each DoF handler.
      const unsigned int n_dofs_for_local_dirichlet_space =
        subdomain_hmatrices.back()
          .get_subdomain_to_skeleton_dirichlet_dof_index_map()
          .size();
      const unsigned int n_dofs_for_local_neumann_space =
        subdomain_hmatrices.back()
          .get_subdomain_to_skeleton_neumann_dof_index_map()
          .size();

      std::cout << "=== DoF information on subdomain "
                << subdomain->get_entity_tag() << "===" << std::endl;
      std::cout << "Number of DoFs in local Dirichlet space: "
                << n_dofs_for_local_dirichlet_space << "\n";
      std::cout << "Number of DoFs in local Neumann space: "
                << n_dofs_for_local_neumann_space << std::endl;

      std::vector<types::global_dof_index>
        dof_indices_for_local_dirichlet_space(n_dofs_for_local_dirichlet_space);
      std::vector<types::global_dof_index> dof_indices_for_local_neumann_space(
        n_dofs_for_local_neumann_space);

      gen_linear_indices<vector_uta, types::global_dof_index>(
        dof_indices_for_local_dirichlet_space);
      gen_linear_indices<vector_uta, types::global_dof_index>(
        dof_indices_for_local_neumann_space);

      std::vector<Point<spacedim>> support_points_for_local_dirichlet_space(
        n_dofs_for_local_dirichlet_space);
      std::vector<Point<spacedim>> support_points_for_local_neumann_space(
        n_dofs_for_local_neumann_space);

      // N.B. Because the support points are only used for cluster tree
      // partition, there is no need to use the actually mapping object
      // associated with each surface. Only the first order mapping is enough.
      DoFToolsExt::map_dofs_to_support_points(
        mappings[0]->get_mapping(),
        dof_handler_for_dirichlet_space,
        steklov_poincare_hmat
          .get_subdomain_to_skeleton_dirichlet_dof_index_map(),
        support_points_for_local_dirichlet_space);

      DoFToolsExt::map_dofs_to_support_points(
        mappings[0]->get_mapping(),
        dof_handler_for_neumann_space,
        steklov_poincare_hmat.get_subdomain_to_skeleton_neumann_dof_index_map(),
        support_points_for_local_neumann_space);

      // Compute average mesh cell size at each support point.
      std::vector<double> cell_sizes_for_local_dirichlet_space(
        n_dofs_for_local_dirichlet_space);
      std::vector<double> cell_sizes_for_local_neumann_space(
        n_dofs_for_local_neumann_space);

      cell_sizes_for_local_dirichlet_space.assign(
        n_dofs_for_local_dirichlet_space, 0);
      cell_sizes_for_local_neumann_space.assign(n_dofs_for_local_neumann_space,
                                                0);

      DoFToolsExt::map_dofs_to_average_cell_size(
        dof_handler_for_dirichlet_space,
        steklov_poincare_hmat
          .get_subdomain_to_skeleton_dirichlet_dof_index_map(),
        cell_sizes_for_local_dirichlet_space);

      DoFToolsExt::map_dofs_to_average_cell_size(
        dof_handler_for_neumann_space,
        steklov_poincare_hmat.get_subdomain_to_skeleton_neumann_dof_index_map(),
        cell_sizes_for_local_neumann_space);

      // Initialize cluster trees.
      steklov_poincare_hmat.get_ct_for_subdomain_dirichlet_space() =
        ClusterTree<spacedim>(dof_indices_for_local_dirichlet_space,
                              support_points_for_local_dirichlet_space,
                              cell_sizes_for_local_dirichlet_space,
                              n_min_for_ct);
      steklov_poincare_hmat.get_ct_for_subdomain_neumann_space() =
        ClusterTree<spacedim>(dof_indices_for_local_neumann_space,
                              support_points_for_local_neumann_space,
                              cell_sizes_for_local_neumann_space,
                              n_min_for_ct);

      steklov_poincare_hmat.get_ct_for_subdomain_dirichlet_space().partition(
        support_points_for_local_dirichlet_space,
        cell_sizes_for_local_dirichlet_space);
      steklov_poincare_hmat.get_ct_for_subdomain_neumann_space().partition(
        support_points_for_local_neumann_space,
        cell_sizes_for_local_neumann_space);

      steklov_poincare_hmat.set_dof_e2i_numbering_for_subdomain_dirichlet_space(
        &(steklov_poincare_hmat.get_ct_for_subdomain_dirichlet_space()
            .get_external_to_internal_dof_numbering()));
      steklov_poincare_hmat.set_dof_i2e_numbering_for_subdomain_dirichlet_space(
        &(steklov_poincare_hmat.get_ct_for_subdomain_dirichlet_space()
            .get_internal_to_external_dof_numbering()));
      steklov_poincare_hmat.set_dof_e2i_numbering_for_subdomain_neumann_space(
        &(steklov_poincare_hmat.get_ct_for_subdomain_neumann_space()
            .get_external_to_internal_dof_numbering()));
      steklov_poincare_hmat.set_dof_i2e_numbering_for_subdomain_neumann_space(
        &(steklov_poincare_hmat.get_ct_for_subdomain_neumann_space()
            .get_internal_to_external_dof_numbering()));

      // Initialize block cluster trees.
      steklov_poincare_hmat.get_bct_for_bilinear_form_D() =
        BlockClusterTree<spacedim>(
          steklov_poincare_hmat.get_ct_for_subdomain_dirichlet_space(),
          steklov_poincare_hmat.get_ct_for_subdomain_dirichlet_space(),
          eta,
          n_min_for_bct);

      steklov_poincare_hmat.get_bct_for_bilinear_form_K() =
        BlockClusterTree<spacedim>(
          steklov_poincare_hmat.get_ct_for_subdomain_neumann_space(),
          steklov_poincare_hmat.get_ct_for_subdomain_dirichlet_space(),
          eta,
          n_min_for_bct);

      steklov_poincare_hmat.get_bct_for_bilinear_form_V() =
        BlockClusterTree<spacedim>(
          steklov_poincare_hmat.get_ct_for_subdomain_neumann_space(),
          steklov_poincare_hmat.get_ct_for_subdomain_neumann_space(),
          eta,
          n_min_for_bct);

      steklov_poincare_hmat.get_bct_for_bilinear_form_D().partition(
        *steklov_poincare_hmat
           .get_dof_i2e_numbering_for_subdomain_dirichlet_space(),
        support_points_for_local_dirichlet_space,
        cell_sizes_for_local_dirichlet_space);

      steklov_poincare_hmat.get_bct_for_bilinear_form_K().partition(
        *steklov_poincare_hmat
           .get_dof_i2e_numbering_for_subdomain_neumann_space(),
        *steklov_poincare_hmat
           .get_dof_i2e_numbering_for_subdomain_dirichlet_space(),
        support_points_for_local_neumann_space,
        support_points_for_local_dirichlet_space,
        cell_sizes_for_local_neumann_space,
        cell_sizes_for_local_dirichlet_space);

      steklov_poincare_hmat.get_bct_for_bilinear_form_V().partition(
        *steklov_poincare_hmat
           .get_dof_i2e_numbering_for_subdomain_neumann_space(),
        support_points_for_local_neumann_space,
        cell_sizes_for_local_neumann_space);

      // Initialize subdomain local \hmatrices.
      steklov_poincare_hmat.get_D() = HMatrixSymm<spacedim>(
        steklov_poincare_hmat.get_bct_for_bilinear_form_D(), max_hmat_rank);
      steklov_poincare_hmat.get_K_with_mass_matrix() =
        HMatrix<spacedim>(steklov_poincare_hmat.get_bct_for_bilinear_form_K(),
                          max_hmat_rank,
                          HMatrixSupport::Property::general,
                          HMatrixSupport::BlockType::diagonal_block);
      steklov_poincare_hmat.get_V() = HMatrixSymm<spacedim>(
        steklov_poincare_hmat.get_bct_for_bilinear_form_V(), max_hmat_rank);

      // Create the two wrapper classes for transmission equation and charge
      // neutrality equation.
      system_matrix.get_hmatrices_for_transmission_eqn().emplace_back(
        &steklov_poincare_hmat,
        &system_matrix
           .get_nondirichlet_boundary_to_skeleton_dirichlet_dof_index_map(),
        &system_matrix
           .get_skeleton_to_nondirichlet_boundary_dirichlet_dof_index_map());
      system_matrix.get_hmatrices_for_charge_neutrality_eqn().emplace_back(
        &steklov_poincare_hmat,
        &system_matrix
           .get_nondirichlet_boundary_to_skeleton_dirichlet_dof_index_map(),
        &system_matrix
           .get_skeleton_to_nondirichlet_boundary_dirichlet_dof_index_map());
    }
}


template <int dim, int spacedim>
void
DDMEfield<dim, spacedim>::generate_cell_iterators()
{
  cell_iterators_for_dirichlet_space.reserve(
    dof_handler_for_dirichlet_space.get_triangulation().n_active_cells());
  for (const auto &cell :
       dof_handler_for_dirichlet_space.active_cell_iterators())
    {
      cell_iterators_for_dirichlet_space.push_back(cell);
    }

  cell_iterators_for_neumann_space.reserve(
    dof_handler_for_neumann_space.get_triangulation().n_active_cells());
  for (const auto &cell : dof_handler_for_neumann_space.active_cell_iterators())
    {
      cell_iterators_for_neumann_space.push_back(cell);
    }
}


template <int dim, int spacedim>
void
DDMEfield<dim, spacedim>::setup_system()
{
  // Create subdomain and surface objects.
  create_efield_subdomains_and_surfaces();

  initialize_manifolds_and_mappings();

  // #if ENABLE_DEBUG == 1
  //     // Refine the mesh to see if the assigned spherical manifold works.
  //     tria.refine_global(1);
  //     // Write out the mesh to check if elementary tags have been assigned
  //     // to material ids.
  //     std::ofstream out("output.msh");
  //     write_msh_correct<dim, spacedim>(tria, out);
  //     out.close();
  // #endif

  // Initialize DoF handlers.
  dof_handler_for_dirichlet_space.reinit(tria);
  dof_handler_for_dirichlet_space.distribute_dofs(fe_for_dirichlet_space);
  dof_handler_for_neumann_space.reinit(tria);
  dof_handler_for_neumann_space.distribute_dofs(fe_for_neumann_space);

  // Apply Dirichlet boundary conditions on dielectric surfaces.
  interpolate_surface_dirichlet_bc();

  initialize_subdomain_hmatrices();
}


template <int dim, int spacedim>
void
DDMEfield<dim, spacedim>::assemble_system()
{
  LogStream::Prefix prefix_string("assemble_system");
  Timer             timer;
  MultithreadInfo::set_thread_limit(thread_num);

  /**
   * Define the @p ACAConfig object.
   */
  ACAConfig aca_config(max_hmat_rank, aca_relative_error, eta);

  for (auto &steklov_poincare_hmat : system_matrix.get_subdomain_hmatrices())
    {
    }
}


template <int dim, int spacedim>
void
DDMEfield<dim, spacedim>::output_results()
{
  std::cout << "=== Output results ===" << std::endl;

  // Write the Dirichlet boundary condition.
  std::ofstream          vtk_output(project_name + std::string(".vtk"));
  DataOut<dim, spacedim> data_out;
  data_out.add_data_vector(dof_handler_for_dirichlet_space,
                           dirichlet_bc,
                           "dirichlet_bc");

  Vector<double> dofs_on_nondirichlet_boundary;
  dofs_on_nondirichlet_boundary.reinit(
    dof_handler_for_dirichlet_space.n_dofs());
  for (auto index :
       system_matrix
         .get_nondirichlet_boundary_to_skeleton_dirichlet_dof_index_map())
    {
      dofs_on_nondirichlet_boundary[index] = 1.0;
    }
  data_out.add_data_vector(dof_handler_for_dirichlet_space,
                           dofs_on_nondirichlet_boundary,
                           "nondirichlet");

  data_out.build_patches();
  data_out.write_vtk(vtk_output);
  vtk_output.close();
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_ELECTRIC_FIELD_DDM_EFIELD_H_
