// Copyright (C) 2020-2025 Jihuan Tian <jihuan_tian@hotmail.com>
// Copyright (C) 2023-2024 Xiaozhe Wang <chaoslawful@gmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file laplace_bem.h
 * @brief Implementation of BEM involving kernel functions and singular
 * numerical quadratures.
 *
 * @ingroup sauter_quadrature
 * @date 2020-11-02
 * @author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_LAPLACE_LAPLACE_BEM_H_
#define HIERBEM_INCLUDE_LAPLACE_LAPLACE_BEM_H_

#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <memory>
#include <string>
#include <vector>

#include "cad_mesh/gmsh_manipulation.h"
#include "cad_mesh/subdomain_topology.h"
#include "config.h"
#include "dofs/dof_tools_ext.h"
#include "hmatrix/hmatrix_vmult_strategy.h"
#include "mapping/mapping_info.h"
#include "preconditioners/preconditioner_type.h"
#include "quadrature/sauter_quadrature_tools.h"
#include <experimental/propagate_const>

HBEM_NS_OPEN

using namespace dealii;

template <int dim,
          int spacedim,
          typename RangeNumberType  = double,
          typename KernelNumberType = RangeNumberType>
class LaplaceBEM
{
public:
#pragma region == == Typedefs == ==
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

  using real_type = typename numbers::NumberTraits<RangeNumberType>::real_type;

#pragma endregion
#pragma region == == Constants == ==
  /**
   * Maximum mapping order used for representing curved manifolds.
   */
  inline static const unsigned int max_mapping_order = 3;

  /**
   * The large integer for shifting the material id in the interfacial domain
   * \f$\Omega_I\f$.
   */
  const static types::material_id material_id_shift_for_interfacial_domain;
#pragma endregion
#pragma region == == Ctor and Dtor == ==

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
             real_type    eta,
             unsigned int max_hmat_rank,
             real_type    aca_relative_error,
             real_type    eta_for_preconditioner,
             unsigned int max_hmat_rank_for_preconditioner,
             real_type    aca_relative_error_for_preconditioner,
             unsigned int thread_num);

  /**
   * Destructor, where DoF handlers are cleared.
   */
  ~LaplaceBEM();

#pragma endregion
#pragma region == == Public member functions == ==

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
  assign_dirichlet_bc(Function<spacedim, RangeNumberType> &f,
                      const EntityTag                      surface_tag = -1);

  /**
   * Assign Dirichlet boundary condition function object to a set of surfaces.
   *
   * @pre
   * @post
   * @param f
   * @param surface_tags
   */
  void
  assign_dirichlet_bc(Function<spacedim, RangeNumberType> &f,
                      const std::vector<EntityTag>        &surface_tags);

  /**
   * Assign Neumann boundary condition function object to all or a specific
   * surface.
   *
   * @param f
   * @param surface_tag Surface entity tag. When it is -1, assign this
   * function to all surfaces in the model.
   */
  void
  assign_neumann_bc(Function<spacedim, RangeNumberType> &f,
                    const EntityTag                      surface_tag = -1);

  /**
   * Assign Neumann boundary condition function object to a set of surfaces.
   *
   * @param f
   * @param surface_tags
   */
  void
  assign_neumann_bc(Function<spacedim, RangeNumberType> &f,
                    const std::vector<EntityTag>        &surface_tags);

  /**
   * Validate the subdomain topology.
   *
   * This function should be called before @p setup_system.
   */
  bool
  validate_subdomain_topology() const;

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

  template <int TargetDim>
  void
  output_results_on_target_tria(const std::string                   vtk_file,
                                Triangulation<TargetDim, spacedim> &tria) const;

  void
  run();

  /**
   * Print out the memory consumption table.
   */
  void
  print_memory_consumption_table(std::ostream &out) const;

#pragma endregion
#pragma region == == Accessors == == =

  KernelNumberType
  get_alpha_for_neumann() const;

  void
  set_alpha_for_neumann(KernelNumberType alphaForNeumann);

  bool
  is_cpu_serial() const;

  void
  set_cpu_serial(bool cpuSerial);

  bool
  is_use_hmat() const;

  void
  set_use_hmat(bool useHmat);

  void
  set_iterative_solver_vmult_type(const IterativeSolverVmultType type);

  void
  set_preconditioner_type(const PreconditionerType type);

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

  const Vector<RangeNumberType> &
  get_dirichlet_data() const;

  Vector<RangeNumberType> &
  get_dirichlet_data();

  const Vector<RangeNumberType> &
  get_neumann_data() const;

  Vector<RangeNumberType> &
  get_neumann_data();
#pragma endregion

private:
  class Priv;

  std::experimental::propagate_const<std::unique_ptr<Priv>> priv_;
};

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_LAPLACE_LAPLACE_BEM_H_
