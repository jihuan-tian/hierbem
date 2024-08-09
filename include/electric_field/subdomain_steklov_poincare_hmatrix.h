/**
 * @file subdomain_steklov_poincare_hmatrix.h
 * @brief Introduction of subdomain_steklov_poincare_hmatrix.h
 *
 * @date 2024-08-08
 * @author Jihuan Tian
 */
#ifndef INCLUDE_ELECTRIC_FIELD_SUBDOMAIN_STEKLOV_POINCARE_HMATRIX_H_
#define INCLUDE_ELECTRIC_FIELD_SUBDOMAIN_STEKLOV_POINCARE_HMATRIX_H_

#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/vector.h>

#include <vector>

#include "hmatrix.h"
#include "hmatrix_symm.h"
#include "hmatrix_symm_preconditioner.h"

namespace HierBEM
{
  using namespace dealii;

  template <int spacedim, typename Number = double>
  class SubdomainSteklovPoincareHMatrix
  {
  public:
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

  private:
    HMatrixSymm<spacedim, Number>               D;
    HMatrix<spacedim, Number>                   K_with_mass_matrix;
    HMatrixSymm<spacedim, Number>               V;
    HMatrixSymmPreconditioner<spacedim, Number> V_preconditioner;

    std::vector<types::global_dof_index>
      subdomain_to_skeleton_dirichlet_dof_index_map;
    std::vector<types::global_dof_index>
      subdomain_to_skeleton_neumann_dof_index_map;
    std::vector<types::global_dof_index>
      nondirichlet_boundary_to_skeleton_dirichlet_dof_index_map;

    ClusterTree<spacedim, Number>      ct_for_subdomain_dirichlet_space;
    ClusterTree<spacedim, Number>      ct_for_subdomain_neumann_space;
    BlockClusterTree<spacedim, Number> bct_for_bilinear_form_D;
    BlockClusterTree<spacedim, Number> bct_for_bilinear_form_K;
    BlockClusterTree<spacedim, Number> bct_for_bilinear_form_V;
  };
} // namespace HierBEM


#endif /* INCLUDE_ELECTRIC_FIELD_SUBDOMAIN_STEKLOV_POINCARE_HMATRIX_H_ */
