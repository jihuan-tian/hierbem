/**
 * @file ddm_efield_global_preconditioner.h
 * @brief Introduction of ddm_efield_global_preconditioner.h
 *
 * @date 2024-08-08
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_ELECTRIC_FIELD_DDM_EFIELD_GLOBAL_PRECONDITIONER_H_
#define HIERBEM_INCLUDE_ELECTRIC_FIELD_DDM_EFIELD_GLOBAL_PRECONDITIONER_H_

#include <vector>

#include "config.h"
#include "subdomain_hmatrix_for_global_preconditioning.h"

HBEM_NS_OPEN

using namespace dealii;

template <int spacedim, typename Number = double>
class DDMEfieldGlobalPreconditioner
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
  std::vector<SubdomainHMatrixForGlobalPreconditioning<spacedim, Number>>
    hmatrices_for_global_preconditioning;
};

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_ELECTRIC_FIELD_DDM_EFIELD_GLOBAL_PRECONDITIONER_H_
