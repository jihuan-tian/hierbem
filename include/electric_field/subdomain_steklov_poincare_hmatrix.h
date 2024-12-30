/**
 * @file subdomain_steklov_poincare_hmatrix.h
 * @brief Introduction of subdomain_steklov_poincare_hmatrix.h
 *
 * @date 2024-08-08
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_ELECTRIC_FIELD_SUBDOMAIN_STEKLOV_POINCARE_HMATRIX_H_
#define HIERBEM_INCLUDE_ELECTRIC_FIELD_SUBDOMAIN_STEKLOV_POINCARE_HMATRIX_H_

#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/vector.h>

#include <vector>

#include "config.h"
#include "hmatrix/hmatrix.h"
#include "hmatrix/hmatrix_symm.h"
#include "hmatrix/preconditioner/hmatrix_symm_preconditioner.h"

HBEM_NS_OPEN

using namespace dealii;

template <int spacedim, typename Number = double>
class SubdomainSteklovPoincareHMatrix
{
public:
  template <int dim>
  void
  build_local_to_global_dof_maps_and_inverses(
    const DoFHandler<dim, spacedim> &dof_handler_for_dirichlet_space,
    const DoFHandler<dim, spacedim> &dof_handler_for_neumann_space,
    const std::vector<bool>         &dof_selectors_for_dirichlet_space,
    const std::vector<bool>         &dof_selectors_for_neumann_space);

  template <int dim>
  void
  build_cluster_trees(
    const DoFHandler<dim, spacedim> &dof_handler_for_dirichlet_space,
    const DoFHandler<dim, spacedim> &dof_handler_for_neumann_space);

  void
  build_hmatrices();

  void
  build_and_prepare_local_preconditioners();

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

  const BlockClusterTree<spacedim, Number> &
  get_bct_for_bilinear_form_D() const
  {
    return bct_for_bilinear_form_D;
  }

  const BlockClusterTree<spacedim, Number> &
  get_bct_for_bilinear_form_K() const
  {
    return bct_for_bilinear_form_K;
  }

  const BlockClusterTree<spacedim, Number> &
  get_bct_for_bilinear_form_V() const
  {
    return bct_for_bilinear_form_V;
  }

  const ClusterTree<spacedim, Number> &
  get_ct_for_subdomain_dirichlet_space() const
  {
    return ct_for_subdomain_dirichlet_space;
  }

  const ClusterTree<spacedim, Number> &
  get_ct_for_subdomain_neumann_space() const
  {
    return ct_for_subdomain_neumann_space;
  }

  const HMatrixSymm<spacedim, Number> &
  get_D() const
  {
    return D;
  }

  const HMatrix<spacedim, Number> &
  get_K_with_mass_matrix() const
  {
    return K_with_mass_matrix;
  }

  const std::vector<types::global_dof_index> &
  get_subdomain_to_skeleton_dirichlet_dof_index_map() const
  {
    return subdomain_to_skeleton_dirichlet_dof_index_map;
  }

  const std::vector<types::global_dof_index> &
  get_subdomain_to_skeleton_neumann_dof_index_map() const
  {
    return subdomain_to_skeleton_neumann_dof_index_map;
  }

  const HMatrixSymm<spacedim, Number> &
  get_V() const
  {
    return V;
  }

  const HMatrixSymmPreconditioner<spacedim, Number> &
  get_V_preconditioner() const
  {
    return V_preconditioner;
  }

  BlockClusterTree<spacedim, Number> &
  get_bct_for_bilinear_form_D()
  {
    return bct_for_bilinear_form_D;
  }

  BlockClusterTree<spacedim, Number> &
  get_bct_for_bilinear_form_K()
  {
    return bct_for_bilinear_form_K;
  }

  BlockClusterTree<spacedim, Number> &
  get_bct_for_bilinear_form_V()
  {
    return bct_for_bilinear_form_V;
  }

  ClusterTree<spacedim, Number> &
  get_ct_for_subdomain_dirichlet_space()
  {
    return ct_for_subdomain_dirichlet_space;
  }

  ClusterTree<spacedim, Number> &
  get_ct_for_subdomain_neumann_space()
  {
    return ct_for_subdomain_neumann_space;
  }

  HMatrixSymm<spacedim, Number> &
  get_D()
  {
    return D;
  }

  HMatrix<spacedim, Number> &
  get_K_with_mass_matrix()
  {
    return K_with_mass_matrix;
  }

  std::vector<types::global_dof_index> &
  get_subdomain_to_skeleton_dirichlet_dof_index_map()
  {
    return subdomain_to_skeleton_dirichlet_dof_index_map;
  }

  std::vector<types::global_dof_index> &
  get_subdomain_to_skeleton_neumann_dof_index_map()
  {
    return subdomain_to_skeleton_neumann_dof_index_map;
  }

  HMatrixSymm<spacedim, Number> &
  get_V()
  {
    return V;
  }

  HMatrixSymmPreconditioner<spacedim, Number> &
  get_V_preconditioner()
  {
    return V_preconditioner;
  }

  const std::vector<int> &
  get_skeleton_to_subdomain_dirichlet_dof_index_map() const
  {
    return skeleton_to_subdomain_dirichlet_dof_index_map;
  }

  const std::vector<int> &
  get_skeleton_to_subdomain_neumann_dof_index_map() const
  {
    return skeleton_to_subdomain_neumann_dof_index_map;
  }

  std::vector<int> &
  get_skeleton_to_subdomain_dirichlet_dof_index_map()
  {
    return skeleton_to_subdomain_dirichlet_dof_index_map;
  }

  std::vector<int> &
  get_skeleton_to_subdomain_neumann_dof_index_map()
  {
    return skeleton_to_subdomain_neumann_dof_index_map;
  }

  const std::vector<types::global_dof_index> *
  get_dof_e2i_numbering_for_subdomain_dirichlet_space() const
  {
    return dof_e2i_numbering_for_subdomain_dirichlet_space;
  }

  void
  set_dof_e2i_numbering_for_subdomain_dirichlet_space(
    const std::vector<types::global_dof_index>
      *dofE2iNumberingForSubdomainDirichletSpace)
  {
    dof_e2i_numbering_for_subdomain_dirichlet_space =
      dofE2iNumberingForSubdomainDirichletSpace;
  }

  const std::vector<types::global_dof_index> *
  get_dof_e2i_numbering_for_subdomain_neumann_space() const
  {
    return dof_e2i_numbering_for_subdomain_neumann_space;
  }

  void
  set_dof_e2i_numbering_for_subdomain_neumann_space(
    const std::vector<types::global_dof_index>
      *dofE2iNumberingForSubdomainNeumannSpace)
  {
    dof_e2i_numbering_for_subdomain_neumann_space =
      dofE2iNumberingForSubdomainNeumannSpace;
  }

  const std::vector<types::global_dof_index> *
  get_dof_i2e_numbering_for_subdomain_dirichlet_space() const
  {
    return dof_i2e_numbering_for_subdomain_dirichlet_space;
  }

  void
  set_dof_i2e_numbering_for_subdomain_dirichlet_space(
    const std::vector<types::global_dof_index>
      *dofI2eNumberingForSubdomainDirichletSpace)
  {
    dof_i2e_numbering_for_subdomain_dirichlet_space =
      dofI2eNumberingForSubdomainDirichletSpace;
  }

  const std::vector<types::global_dof_index> *
  get_dof_i2e_numbering_for_subdomain_neumann_space() const
  {
    return dof_i2e_numbering_for_subdomain_neumann_space;
  }

  void
  set_dof_i2e_numbering_for_subdomain_neumann_space(
    const std::vector<types::global_dof_index>
      *dofI2eNumberingForSubdomainNeumannSpace)
  {
    dof_i2e_numbering_for_subdomain_neumann_space =
      dofI2eNumberingForSubdomainNeumannSpace;
  }

private:
  HMatrixSymm<spacedim, Number>               D;
  HMatrix<spacedim, Number>                   K_with_mass_matrix;
  HMatrixSymm<spacedim, Number>               V;
  HMatrixSymmPreconditioner<spacedim, Number> V_preconditioner;

  std::vector<types::global_dof_index>
    subdomain_to_skeleton_dirichlet_dof_index_map;
  std::vector<types::global_dof_index>
                   subdomain_to_skeleton_neumann_dof_index_map;
  std::vector<int> skeleton_to_subdomain_dirichlet_dof_index_map;
  std::vector<int> skeleton_to_subdomain_neumann_dof_index_map;

  ClusterTree<spacedim, Number> ct_for_subdomain_dirichlet_space;
  ClusterTree<spacedim, Number> ct_for_subdomain_neumann_space;
  const std::vector<types::global_dof_index>
    *dof_e2i_numbering_for_subdomain_dirichlet_space;
  const std::vector<types::global_dof_index>
    *dof_i2e_numbering_for_subdomain_dirichlet_space;
  const std::vector<types::global_dof_index>
    *dof_e2i_numbering_for_subdomain_neumann_space;
  const std::vector<types::global_dof_index>
    *dof_i2e_numbering_for_subdomain_neumann_space;
  BlockClusterTree<spacedim, Number> bct_for_bilinear_form_D;
  BlockClusterTree<spacedim, Number> bct_for_bilinear_form_K;
  BlockClusterTree<spacedim, Number> bct_for_bilinear_form_V;
};


template <int spacedim, typename Number>
template <int dim>
void
SubdomainSteklovPoincareHMatrix<spacedim, Number>::
  build_local_to_global_dof_maps_and_inverses(
    const DoFHandler<dim, spacedim> &dof_handler_for_dirichlet_space,
    const DoFHandler<dim, spacedim> &dof_handler_for_neumann_space,
    const std::vector<bool>         &dof_selectors_for_dirichlet_space,
    const std::vector<bool>         &dof_selectors_for_neumann_space)
{
  // Build the local-to-global DoF map for Dirichlet space.
  subdomain_to_skeleton_dirichlet_dof_index_map.reserve(
    dof_handler_for_dirichlet_space.n_dofs());
  skeleton_to_subdomain_dirichlet_dof_index_map.resize(
    dof_handler_for_dirichlet_space.n_dofs());
  std::fill_n(skeleton_to_subdomain_dirichlet_dof_index_map.begin(),
              skeleton_to_subdomain_dirichlet_dof_index_map.size(),
              -1);

  for (types::global_dof_index i = 0;
       i < dof_handler_for_dirichlet_space.n_dofs();
       i++)
    {
      if (dof_selectors_for_dirichlet_space[i])
        {
          subdomain_to_skeleton_dirichlet_dof_index_map.push_back(i);
          skeleton_to_subdomain_dirichlet_dof_index_map[i] =
            subdomain_to_skeleton_dirichlet_dof_index_map.size() - 1;
        }
    }

  subdomain_to_skeleton_neumann_dof_index_map.reserve(
    dof_handler_for_neumann_space.n_dofs());
  skeleton_to_subdomain_neumann_dof_index_map.resize(
    dof_handler_for_neumann_space.n_dofs());
  std::fill_n(skeleton_to_subdomain_neumann_dof_index_map.begin(),
              skeleton_to_subdomain_neumann_dof_index_map.size(),
              -1);

  for (types::global_dof_index i = 0;
       i < dof_handler_for_neumann_space.n_dofs();
       i++)
    {
      if (dof_selectors_for_neumann_space[i])
        {
          subdomain_to_skeleton_neumann_dof_index_map.push_back(i);
          skeleton_to_subdomain_neumann_dof_index_map[i] =
            subdomain_to_skeleton_neumann_dof_index_map.size() - 1;
        }
    }
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_ELECTRIC_FIELD_SUBDOMAIN_STEKLOV_POINCARE_HMATRIX_H_
