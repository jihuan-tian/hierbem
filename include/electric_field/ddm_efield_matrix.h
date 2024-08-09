/**
 * @file ddm_efield_matrix.h
 * @brief Introduction of ddm_efield_matrix.h
 *
 * @date 2024-08-08
 * @author Jihuan Tian
 */
#ifndef INCLUDE_ELECTRIC_FIELD_DDM_EFIELD_MATRIX_H_
#define INCLUDE_ELECTRIC_FIELD_DDM_EFIELD_MATRIX_H_

#include <deal.II/lac/vector.h>

#include <vector>

#include "subdomain_hmatrix_for_charge_neutrality_eqn.h"
#include "subdomain_hmatrix_for_transmission_eqn.h"

namespace HierBEM
{
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

  private:
    std::vector<SubdomainHMatrixForTransmissionEqn<spacedim, Number>>
      hmatrices_for_transmission_eqn;
    std::vector<SubdomainHMatrixForChargeNeutralityEqn<spacedim, Number>>
      hmatrices_for_charge_neutrality_eqn;
  };
} // namespace HierBEM

#endif /* INCLUDE_ELECTRIC_FIELD_DDM_EFIELD_MATRIX_H_ */
