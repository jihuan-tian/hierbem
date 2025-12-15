// Copyright (C) 2020-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file helmholtz_acoustic_bem.h
 * @brief Definition of class for solving the Helmholtz acoustic equation using
 * BEM.
 *
 * @date 2025-12-14
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_HELMHOLTZ_HELMHOLTZ_ACOUSTIC_BEM_H_
#define HIERBEM_INCLUDE_HELMHOLTZ_HELMHOLTZ_ACOUSTIC_BEM_H_

#include <deal.II/base/numbers.h>

#include "config.h"
#include "hmatrix/hmatrix_parameters.h"

HBEM_NS_OPEN

using namespace dealii;

template <int dim,
          int spacedim,
          typename RangeNumberType,
          typename KernelNumberType>
class HelmholtzAcousticBEM
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
  HelmholtzAcousticBEM(const ProblemType       problem_type,
                       const bool              is_interior_problem,
                       const HMatrixParameters hmat_params,
                       const unsigned int      thread_num);

  ~HelmholtzAcousticBEM();
#pragma endregion
#pragma region == == Public member functions == ==
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

  bool
  validate_subdomain_topology() const;

  void
  initialize_manifolds_from_manifold_description();

  void
  initialize_mappings();

  void
  interpolate_dirichlet_bc();

  void
  interpolate_neumann_bc();

  void
  assemble_hmatrix_system();

  void
  assemble_hmatrix_preconditioner();
#pragma endregion
};

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_HELMHOLTZ_HELMHOLTZ_ACOUSTIC_BEM_H_