#ifndef HIERBEM_INCLUDE_HMATRIX_HMATRIX_PARAMETERS_H_
#define HIERBEM_INCLUDE_HMATRIX_HMATRIX_PARAMETERS_H_

#include "config.h"

HBEM_NS_OPEN

/**
 * @brief Class holding the set of \hmatrix parameters.
 */
class HMatrixParameters
{
public:
  HMatrixParameters(const unsigned int n_min_for_ct,
                    const unsigned int n_min_for_bct,
                    const double       eta,
                    const unsigned int max_hmat_rank,
                    const double       aca_relative_error)
    : n_min_for_ct(n_min_for_ct)
    , n_min_for_bct(n_min_for_bct)
    , eta(eta)
    , max_hmat_rank(max_hmat_rank)
    , aca_relative_error(aca_relative_error)
  {}

  /**
   * Minimum cluster node size in the cluster tree.
   */
  unsigned int n_min_for_ct;
  /**
   * Minimum block cluster node size in the block cluster tree.
   */
  unsigned int n_min_for_bct;
  /**
   * Admissibility constant.
   */
  double eta;
  /**
   * Maximum \hmat rank.
   */
  unsigned int max_hmat_rank;
  /**
   * Relative error for ACA iteration.
   */
  double aca_relative_error;
};

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_HMATRIX_HMATRIX_PARAMETERS_H_