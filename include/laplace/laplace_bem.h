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
 * @brief Definition of the class for solving the Laplace equation using BEM.
 *
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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cad_mesh/gmsh_manipulation.h"
#include "cad_mesh/subdomain_topology.h"
#include "config.h"
#include "config_file/config_structs.h"
#include "hmatrix/hmatrix_vmult_strategy.h"
#include "mapping/mapping_info.h"
#include "preconditioners/preconditioner_type.h"
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
  using real_type = typename numbers::NumberTraits<RangeNumberType>::real_type;
#pragma endregion

#pragma region == == Constants == ==
  /**
   * Maximum mapping order used for representing curved manifolds.
   */
  inline static const unsigned int max_mapping_order = 3;
#pragma endregion
#pragma region == == Ctor and Dtor == ==
  /**
   * @brief Constructor for solving Laplace problem using full matrix, which
   * has only verification purpose.
   *
   * @param bem_params
   */
  LaplaceBEM(const ConfLaplaceBEM      &bem_params,
             const ConfSauterQuad      &sauter_quad_params,
             const ConfLinearSolver    &linear_solver_params,
             const ConfParallelization &parallel_params);

  /**
   * @brief Constructor for solving Laplace problem using \hmatrix.
   *
   * @param bem_params
   * @param hmat_params
   * @param hmat_preconditioner_params
   * @param sauter_quad_params
   * @param sauter_quad_precond_params
   * @param linear_solver_params
   * @param op_precond_params
   * @param parallel_params
   */
  LaplaceBEM(const ConfLaplaceBEM             &bem_params,
             const ConfHMatrix                &hmat_params,
             const ConfHMatrix                &hmat_preconditioner_params,
             const ConfSauterQuad             &sauter_quad_params,
             const ConfSauterQuad             &sauter_quad_precond_params,
             const ConfLinearSolver           &linear_solver_params,
             const ConfOperatorPreconditioner &op_precond_params,
             const ConfParallelization        &parallel_params);

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

  /**
   * Evaluate potentials and conormal traces on the target 2D triangulation.
   */
  void
  output_results_on_target_tria(const std::string                  &vtk_file,
                                const Triangulation<dim, spacedim> &tria,
                                const unsigned int mapping_order) const;

  /**
   * Evaluate potentials on the target 3D triangulation.
   */
  void
  output_results_on_target_tria(const std::string                &vtk_file,
                                const Triangulation<3, spacedim> &tria,
                                const unsigned int mapping_order) const;

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

  const std::map<EntityTag, Function<spacedim, RangeNumberType> *> &
  get_dirichlet_bc_definition() const;

  std::map<EntityTag, Function<spacedim, RangeNumberType> *> &
  get_dirichlet_bc_definition();

  const std::map<EntityTag, Function<spacedim, RangeNumberType> *> &
  get_neumann_bc_definition() const;

  std::map<EntityTag, Function<spacedim, RangeNumberType> *> &
  get_neumann_bc_definition();

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
