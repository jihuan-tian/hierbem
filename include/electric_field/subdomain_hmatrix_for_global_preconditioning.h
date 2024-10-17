/**
 * @file subdomain_hmatrix_for_global_preconditioning.h
 * @brief Introduction of subdomain_hmatrix_for_global_preconditioning.h
 *
 * @date 2024-08-08
 * @author Jihuan Tian
 */
#ifndef INCLUDE_ELECTRIC_FIELD_SUBDOMAIN_HMATRIX_FOR_GLOBAL_PRECONDITIONING_H_
#define INCLUDE_ELECTRIC_FIELD_SUBDOMAIN_HMATRIX_FOR_GLOBAL_PRECONDITIONING_H_

#include "subdomain_steklov_poincare_hmatrix.h"

namespace HierBEM
{
  template <int spacedim, typename Number = double>
  class SubdomainHMatrixForGlobalPreconditioning
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

  private:
    SubdomainSteklovPoincareHMatrix<spacedim, Number> *S;
  };
} // namespace HierBEM

#endif /* INCLUDE_ELECTRIC_FIELD_SUBDOMAIN_HMATRIX_FOR_GLOBAL_PRECONDITIONING_H_ \
        */
