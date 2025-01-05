/**
 * @file subdomain_hmatrix_for_charge_neutrality_eqn.h
 * @brief Introduction of subdomain_hmatrix_for_charge_neutrality_eqn.h
 *
 * @date 2024-08-08
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_ELECTRIC_FIELD_SUBDOMAIN_HMATRIX_FOR_CHARGE_NEUTRALITY_EQN_H_
#define HIERBEM_INCLUDE_ELECTRIC_FIELD_SUBDOMAIN_HMATRIX_FOR_CHARGE_NEUTRALITY_EQN_H_

#include "config.h"
#include "subdomain_steklov_poincare_hmatrix.h"

HBEM_NS_OPEN

template <int spacedim, typename Number = double>
class SubdomainHMatrixForChargeNeutralityEqn
{
public:
  SubdomainHMatrixForChargeNeutralityEqn() = default;

  SubdomainHMatrixForChargeNeutralityEqn(
    SubdomainSteklovPoincareHMatrix<spacedim, Number> *S,
    std::vector<types::global_dof_index>
      *nondirichlet_boundary_to_skeleton_dirichlet_dof_index_map,
    std::vector<int>
      *skeleton_to_nondirichlet_boundary_dirichlet_dof_index_map);

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

private:
  SubdomainSteklovPoincareHMatrix<spacedim, Number> *S;
  std::vector<types::global_dof_index>
                   *nondirichlet_boundary_to_skeleton_dirichlet_dof_index_map;
  std::vector<int> *skeleton_to_nondirichlet_boundary_dirichlet_dof_index_map;
};


template <int spacedim, typename Number>
SubdomainHMatrixForChargeNeutralityEqn<spacedim, Number>::
  SubdomainHMatrixForChargeNeutralityEqn(
    SubdomainSteklovPoincareHMatrix<spacedim, Number> *S,
    std::vector<types::global_dof_index>
                     *nondirichlet_boundary_to_skeleton_dirichlet_dof_index_map,
    std::vector<int> *skeleton_to_nondirichlet_boundary_dirichlet_dof_index_map)
  : S(S)
  , nondirichlet_boundary_to_skeleton_dirichlet_dof_index_map(
      nondirichlet_boundary_to_skeleton_dirichlet_dof_index_map)
  , skeleton_to_nondirichlet_boundary_dirichlet_dof_index_map(
      skeleton_to_nondirichlet_boundary_dirichlet_dof_index_map)
{}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_ELECTRIC_FIELD_SUBDOMAIN_HMATRIX_FOR_CHARGE_NEUTRALITY_EQN_H_
