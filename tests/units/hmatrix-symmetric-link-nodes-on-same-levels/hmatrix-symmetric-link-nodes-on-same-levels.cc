/**
 * @file hmatrix-symmetric-link-nodes-on-same-levels.cu
 * @brief Link all \hmatrix nodes on a same level. The \hmatrix is symmetric.
 *
 * @ingroup testers
 * @author Jihuan Tian
 * @date 2024-01-04
 */

#include <iostream>

#include "hbem_octave_wrapper.h"
#include "hmatrix/hmatrix.h"
#include "lapack_full_matrix_ext.h"

using namespace std;
using namespace HierBEM;

int
main()
{
  HBEMOctaveWrapper &inst    = HBEMOctaveWrapper::get_instance();
  auto               oct_val = inst.eval_string(
    "reshape((1:32*32) / 100, 32, 32) * reshape((1:32*32) / 100, 32, 32)'");
  std::vector<double> values;
  unsigned int        n;
  oct_val.matrix_value(values, n, n);

  LAPACKFullMatrixExt<double> M;
  LAPACKFullMatrixExt<double>::Reshape(n, n, values, M);
  M.set_property(LAPACKSupport::Property::symmetric);

  /**
   * Generate index set.
   */
  std::vector<types::global_dof_index> index_set(n);

  for (unsigned int i = 0; i < n; i++)
    {
      index_set.at(i) = i;
    }

  const unsigned int n_min = 2;

  /**
   * Generate cluster tree.
   */
  ClusterTree<3, double> cluster_tree(index_set, n_min);
  cluster_tree.partition();

  /**
   * Generate block cluster tree with the two component cluster trees being the
   * same.
   */
  BlockClusterTree<3, double> bct(cluster_tree, cluster_tree);
  bct.partition_fine_non_tensor_product();

  /**
   * Create the \hmatrix from the source matrix.
   */
  const unsigned int fixed_rank = 2;
  HMatrix<3, double> H(bct, M, fixed_rank);

  H.print_matrix_info_as_dot(std::cout);

  return 0;
}
