// Copyright (C) 2024-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file ddm_efield_matrix.h
 * @brief Introduction of ddm_efield_matrix.h
 *
 * @date 2024-08-08
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_ELECTRIC_FIELD_DDM_EFIELD_MATRIX_H_
#define HIERBEM_INCLUDE_ELECTRIC_FIELD_DDM_EFIELD_MATRIX_H_

#include <deal.II/lac/vector.h>

#include <vector>

#include "config.h"
#include "subdomain_hmatrix_for_charge_neutrality_eqn.h"
#include "subdomain_hmatrix_for_transmission_eqn.h"
#include "subdomain_steklov_poincare_hmatrix.h"

HBEM_NS_OPEN

template <int spacedim, typename Number = double>
class DDMEfieldMatrix
{
public:
  /**
   * Calculate \hmatrix/vector multiplication as \f$y = y + M \cdot x\f$.
   *
   * @param y Result vector
   * @param x Input vector
   */
  void
  vmult(Vector<Number> &y, const Vector<Number> &x) const;

  /**
   * Calculate \hmatrix/vector multiplication as \f$y = y + \alpha \cdot M
   * \cdot x\f$.
   *
   * @param y Result vector
   * @param alpha Scalar factor before \f$x\f$
   * @param x Input vector
   */
  void
  vmult(Vector<Number> &y, const Number alpha, const Vector<Number> &x) const;

  template <int dim>
  void
  build_local_to_global_dirichlet_dof_map_and_inverse_on_nondirichlet_boundary(
    const DoFHandler<dim, spacedim> &dof_handler_for_dirichlet_space,
    const std::vector<bool>
      &negated_dof_selectors_for_dirichlet_space_on_nondirichlet_boundary);

  const std::vector<SubdomainHMatrixForChargeNeutralityEqn<spacedim, Number>> &
  get_hmatrices_for_charge_neutrality_eqn() const
  {
    return hmatrices_for_charge_neutrality_eqn;
  }

  const std::vector<SubdomainHMatrixForTransmissionEqn<spacedim, Number>> &
  get_hmatrices_for_transmission_eqn() const
  {
    return hmatrices_for_transmission_eqn;
  }

  const std::vector<SubdomainSteklovPoincareHMatrix<spacedim, Number>> &
  get_subdomain_hmatrices() const
  {
    return subdomain_hmatrices;
  }

  const std::vector<types::global_dof_index> &
  get_nondirichlet_boundary_to_skeleton_dirichlet_dof_index_map() const
  {
    return nondirichlet_boundary_to_skeleton_dirichlet_dof_index_map;
  }

  std::vector<SubdomainHMatrixForChargeNeutralityEqn<spacedim, Number>> &
  get_hmatrices_for_charge_neutrality_eqn()
  {
    return hmatrices_for_charge_neutrality_eqn;
  }

  std::vector<SubdomainHMatrixForTransmissionEqn<spacedim, Number>> &
  get_hmatrices_for_transmission_eqn()
  {
    return hmatrices_for_transmission_eqn;
  }

  std::vector<SubdomainSteklovPoincareHMatrix<spacedim, Number>> &
  get_subdomain_hmatrices()
  {
    return subdomain_hmatrices;
  }

  std::vector<types::global_dof_index> &
  get_nondirichlet_boundary_to_skeleton_dirichlet_dof_index_map()
  {
    return nondirichlet_boundary_to_skeleton_dirichlet_dof_index_map;
  }

  const std::vector<int> &
  get_skeleton_to_nondirichlet_boundary_dirichlet_dof_index_map() const
  {
    return skeleton_to_nondirichlet_boundary_dirichlet_dof_index_map;
  }

  std::vector<int> &
  get_skeleton_to_nondirichlet_boundary_dirichlet_dof_index_map()
  {
    return skeleton_to_nondirichlet_boundary_dirichlet_dof_index_map;
  }

private:
  std::vector<SubdomainSteklovPoincareHMatrix<spacedim, Number>>
    subdomain_hmatrices;
  std::vector<types::global_dof_index>
                   nondirichlet_boundary_to_skeleton_dirichlet_dof_index_map;
  std::vector<int> skeleton_to_nondirichlet_boundary_dirichlet_dof_index_map;
  std::vector<SubdomainHMatrixForTransmissionEqn<spacedim, Number>>
    hmatrices_for_transmission_eqn;
  std::vector<SubdomainHMatrixForChargeNeutralityEqn<spacedim, Number>>
    hmatrices_for_charge_neutrality_eqn;
};


template <int spacedim, typename Number>
template <int dim>
void
DDMEfieldMatrix<spacedim, Number>::
  build_local_to_global_dirichlet_dof_map_and_inverse_on_nondirichlet_boundary(
    const DoFHandler<dim, spacedim> &dof_handler_for_dirichlet_space,
    const std::vector<bool>
      &negated_dof_selectors_for_dirichlet_space_on_nondirichlet_boundary)
{
  nondirichlet_boundary_to_skeleton_dirichlet_dof_index_map.reserve(
    dof_handler_for_dirichlet_space.n_dofs());
  skeleton_to_nondirichlet_boundary_dirichlet_dof_index_map.resize(
    dof_handler_for_dirichlet_space.n_dofs());
  std::fill_n(skeleton_to_nondirichlet_boundary_dirichlet_dof_index_map.begin(),
              skeleton_to_nondirichlet_boundary_dirichlet_dof_index_map.size(),
              -1);

  for (types::global_dof_index i = 0;
       i < dof_handler_for_dirichlet_space.n_dofs();
       i++)
    {
      if (!negated_dof_selectors_for_dirichlet_space_on_nondirichlet_boundary
            [i])
        {
          nondirichlet_boundary_to_skeleton_dirichlet_dof_index_map.push_back(
            i);
          skeleton_to_nondirichlet_boundary_dirichlet_dof_index_map[i] =
            nondirichlet_boundary_to_skeleton_dirichlet_dof_index_map.size() -
            1;
        }
    }
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_ELECTRIC_FIELD_DDM_EFIELD_MATRIX_H_
