/** \file laplace_bem.h
 * \brief Implementation of BEM involving kernel functions and singular
 * numerical quadratures.
 *
 * \ingroup sauter_quadrature
 * \date 2020-11-02
 * \author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_LAPLACE_BEM_H_
#define HIERBEM_INCLUDE_LAPLACE_BEM_H_

#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria.h>

#include <memory>
#include <string>
#include <vector>

#include "config.h"
#include "dof_tools_ext.h"
#include "gmsh_manipulation.h"
#include "mapping/mapping_info.h"
#include "sauter_quadrature_tools.h"
#include "subdomain_topology.h"
#include <experimental/propagate_const>

HBEM_NS_OPEN

using namespace dealii;

template <int dim, int spacedim = dim>
class LaplaceBEM
{
public:
#pragma region ==== Typedefs ====

  /**
   * A class for detecting if a surface normal vector points into a volume.
   *
   * If so, the surface normal vector computed for a cell should be negated,
   * because we assume an outward normal vector adopted for the problem
   * domain, whether we're solving an interior BEM problem or exterior
   * problem.
   */
  class SurfaceNormalDetector
  {
  public:
    SurfaceNormalDetector() = delete;
    SurfaceNormalDetector(SubdomainTopology<dim, spacedim> &subdomain_topology)
      : subdom_topo_(subdomain_topology)
    {}

    /**
     * Given a material id of a cell, this function checks if its normal
     * vector points into a corresponding domain by checking the
     * surface-to-subdomain relationship.
     *
     * \mynote{In the Laplace solver, a domain (with a non-zero subdomain tag)
     * must be fully in contact with the surrounding space (whose subdomain
     * tag is zero). This still holds if there are several subdomains in the
     * model, because they are all well separated from each other. This leads
     * to the fact the in a record in the surface-to-subdomain relationship,
     * there should be only one non-zero value. We use this fact to check the
     * direction of the surface normal vector.}
     *
     * @pre
     * @post
     * @param m
     * @return
     */
    bool
    is_normal_vector_inward(const types::material_id m) const
    {
      if (subdom_topo_.get_surface_to_subdomain()[m][0] > 0)
        {
          Assert(subdom_topo_.get_surface_to_subdomain()[m][1] == 0,
                 ExcInternalError());
          return false;
        }
      else
        {
          Assert(subdom_topo_.get_surface_to_subdomain()[m][1] > 0,
                 ExcInternalError());
          return true;
        }
    }

  private:
    SubdomainTopology<dim, spacedim> &subdom_topo_;
  };

  /**
   * Enum for various types of Laplace problem
   */
  enum ProblemType
  {
    NeumannBCProblem,   //!< NeumannBCProblem
    DirichletBCProblem, //!< DirichletBCProblem
    MixedBCProblem,     //!< MixedBCProblem
    UndefinedProblem
  };

#pragma endregion
#pragma region ==== Constants ====
  /**
   * Maximum mapping order.
   */
  inline static const unsigned int max_mapping_order = 3;

  /**
   * The large integer for shifting the material id in the interfacial domain
   * \f$\Omega_I\f$.
   */
  const static types::material_id material_id_shift_for_interfacial_domain;
#pragma endregion
#pragma region ==== Ctor and Dtor ====

  /**
   * Default constructor
   */
  LaplaceBEM();

  /**
   * @brief Constructor for solving Laplace problem using full matrix, which
   * has only verification purpose.
   *
   * @param fe_order_for_dirichlet_space
   * @param fe_order_for_neumann_space
   * @param problem_type
   * @param is_interior_problem
   * @param thread_num
   */
  LaplaceBEM(unsigned int fe_order_for_dirichlet_space,
             unsigned int fe_order_for_neumann_space,
             ProblemType  problem_type,
             bool         is_interior_problem,
             unsigned int thread_num);

  /**
   * @brief Constructor for solving Laplace problem using \hmatrix.
   *
   * @param fe_order_for_dirichlet_space
   * @param fe_order_for_neumann_space
   * @param problem_type
   * @param is_interior_problem
   * @param n_min_for_ct
   * @param n_min_for_bct
   * @param eta
   * @param max_hmat_rank
   * @param aca_relative_error
   * @param eta_for_preconditioner
   * @param max_hmat_rank_for_preconditioner
   * @param aca_relative_error_for_preconditioner
   * @param thread_num
   */
  LaplaceBEM(unsigned int fe_order_for_dirichlet_space,
             unsigned int fe_order_for_neumann_space,
             ProblemType  problem_type,
             bool         is_interior_problem,
             unsigned int n_min_for_ct,
             unsigned int n_min_for_bct,
             double       eta,
             unsigned int max_hmat_rank,
             double       aca_relative_error,
             double       eta_for_preconditioner,
             unsigned int max_hmat_rank_for_preconditioner,
             double       aca_relative_error_for_preconditioner,
             unsigned int thread_num);

  /**
   * Destructor, where DoF handlers are cleared.
   */
  ~LaplaceBEM();

#pragma endregion
#pragma region ==== Public member functions ====

  /**
   * Extract the surface mesh from the given volume mesh.
   *
   * Before calling this function, the association between surface manifold
   * objects and manifold ids should be configured, if there is any.
   */
  void
  extract_surface_triangulation(
    const Triangulation<dim + 1, spacedim> &volume_triangulation,
    Triangulation<dim, spacedim>          &&surf_tria,
    const bool                              debug = false);

  /**
   * Prepare for matrix assembly, which includes:
   * . initialization of DoF handlers
   * . memory allocation for matrices
   */
  void
  setup_system();

  /**
   * Assign Dirichlet boundary condition function object to all or a specific
   * surface.
   *
   * @param f
   * @param surface_tag Surface entity tag. When it is -1, assign this
   * function to all surfaces in the model.
   */
  void
  assign_dirichlet_bc(Function<spacedim, double> &f,
                      const EntityTag             surface_tag = -1);

  /**
   * Assign Dirichlet boundary condition function object to a set of surfaces.
   *
   * @pre
   * @post
   * @param f
   * @param surface_tags
   */
  void
  assign_dirichlet_bc(Function<spacedim, double>   &f,
                      const std::vector<EntityTag> &surface_tags);

  /**
   * Assign Neumann boundary condition function object to all or a specific
   * surface.
   *
   * @param f
   * @param surface_tag Surface entity tag. When it is -1, assign this
   * function to all surfaces in the model.
   */
  void
  assign_neumann_bc(Function<spacedim, double> &f,
                    const EntityTag             surface_tag = -1);

  /**
   * Assign Neumann boundary condition function object to a set of surfaces.
   *
   * @param f
   * @param surface_tags
   */
  void
  assign_neumann_bc(Function<spacedim, double>   &f,
                    const std::vector<EntityTag> &surface_tags);

  void
  initialize_manifolds_from_manifold_description();

  void
  initialize_mappings();

  /**
   * Interpolate Dirichlet boundary conditions.
   */
  void
  interpolate_dirichlet_bc();

  /**
   * Interpolate Neumann boundary conditions.
   */
  void
  interpolate_neumann_bc();

  /**
   * Assemble full matrix system, which is only for verification purpose.
   */
  void
  assemble_full_matrix_system();

  /**
   * Assemble \hmatrix system.
   */
  void
  assemble_hmatrix_system();

  /**
   * Assemble \hmatrix preconditioner.
   */
  void
  assemble_hmatrix_preconditioner();

  void
  solve();

  void
  output_results() const;

  void
  output_potential_at_target_points() const;

  void
  run();

  /**
   * Verify if the calculated Neumann solution is in the space
   * \f$H^{1/2}_*(\Gamma)\f$.
   */
  void
  verify_neumann_solution_in_space();

  /**
   * Print out the memory consumption table.
   */
  void
  print_memory_consumption_table(std::ostream &out) const;

#pragma endregion
#pragma region ==== Accessors =====

  double
  get_alpha_for_neumann() const;

  void
  set_alpha_for_neumann(double alphaForNeumann);

  bool
  is_cpu_serial() const;

  void
  set_cpu_serial(bool cpuSerial);

  bool
  is_use_hmat() const;

  void
  set_use_hmat(bool useHmat);

  const std::string &
  get_project_name() const;

  void
  set_project_name(const std::string &projectName);

  const SubdomainTopology<dim, spacedim> &
  get_subdomain_topology() const;

  SubdomainTopology<dim, spacedim> &
  get_subdomain_topology();

  const std::map<EntityTag, types::manifold_id> &
  get_manifold_description() const;

  std::map<EntityTag, types::manifold_id> &
  get_manifold_description();

  const std::map<types::manifold_id, unsigned int> &
  get_manifold_id_to_mapping_order() const;

  std::map<types::manifold_id, unsigned int> &
  get_manifold_id_to_mapping_order();

  const std::map<types::manifold_id, Manifold<dim, spacedim> *> &
  get_manifolds() const;

  std::map<types::manifold_id, Manifold<dim, spacedim> *> &
  get_manifolds();

  const Triangulation<dim, spacedim> &
  get_triangulation() const;

  Triangulation<dim, spacedim> &
  get_triangulation();

  const std::vector<MappingInfo<dim, spacedim> *> &
  get_mappings() const;

  std::vector<MappingInfo<dim, spacedim> *> &
  get_mappings();

  const DoFHandler<dim, spacedim> &
  get_dof_handler_dirichlet() const;

  DoFHandler<dim, spacedim> &
  get_dof_handler_dirichlet();

  const DoFHandler<dim, spacedim> &
  get_dof_handler_neumann() const;

  DoFHandler<dim, spacedim> &
  get_dof_handler_neumann();

  const Vector<double> &
  get_dirichlet_data() const;

  Vector<double> &
  get_dirichlet_data();

  const Vector<double> &
  get_neumann_data() const;

  Vector<double> &
  get_neumann_data();
#pragma endregion

private:
  class Priv;

  std::experimental::propagate_const<std::unique_ptr<Priv>> priv_;
};

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_LAPLACE_BEM_H_
