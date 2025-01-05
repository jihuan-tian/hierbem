#ifndef HIERBEM_INCLUDE_HMATRIX_ACA_PLUS_ACA_CONFIG_H_
#define HIERBEM_INCLUDE_HMATRIX_ACA_PLUS_ACA_CONFIG_H_

#include "config.h"

HBEM_NS_OPEN

/**
 * Configuration for ACA+.
 */
struct ACAConfig
{
  ACAConfig();
  ACAConfig(unsigned int v_max_iter, double v_epsilon, double v_eta);

  /**
   * Maximum number of iteration, which is also the maximum rank \f$k\f$ for
   * the far field matrix block to be built.
   */
  unsigned int max_iter;
  /**
   * Relative error between the current cross and the approximant matrix
   * \f$S\f$, i.e. \f[ \norm{u_k}_2\norm{v_k}_2 \leq
   * \frac{\varepsilon(1-\eta)}{1+\varepsilon} \norm{S}_{\rm F}. \f]
   */
  double epsilon;
  /**
   * Admissibility constant
   */
  double eta;
};

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_HMATRIX_ACA_PLUS_ACA_CONFIG_H_