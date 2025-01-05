/**
 * \file hmatrix-symm-write-leaf-set-by-iteration-hilbert-curve.cu
 * \brief Verify the method for write out leaf set by iteration over the
 * constructed leaf set instead of recursion. The traversal follows the Hilbert
 * curve.
 *
 * The \hmatrix in this test case is symmetric.
 *
 * \ingroup
 * \author Jihuan Tian
 * \date 2024-03-11
 */

#include <fstream>
#include <iostream>

#include "hbem_octave_wrapper.h"
#include "hmatrix/hmatrix.h"

using namespace std;
using namespace HierBEM;

int
main()
{
  /**
   * Create the global index set.
   */
  const unsigned int                   p = 4;
  const unsigned int                   n = std::pow(2, p);
  std::vector<types::global_dof_index> index_set(n);

  for (unsigned int i = 0; i < n; i++)
    {
      index_set.at(i) = i;
    }

  /**
   * Construct the cluster tree.
   */
  const unsigned int n_min = 1;
  ClusterTree<3>     cluster_tree(index_set, n_min);
  cluster_tree.partition();

  /**
   * Construct the block cluster tree.
   */
  BlockClusterTree<3, double> block_cluster_tree(cluster_tree, cluster_tree);
  block_cluster_tree.partition_fine_non_tensor_product();

  /**
   * Create a full matrix with data.
   */
  HBEMOctaveWrapper &inst = HBEMOctaveWrapper::get_instance();
  inst.add_path(SOURCE_DIR);
  // Execute script `gen_symmetric_matrix.m` to generate M.dat
  inst.source_file(SOURCE_DIR "/gen_symmetric_matrix.m");

  LAPACKFullMatrixExt<double> M;
  ifstream                    in("M.dat");
  M.read_from_mat(in, "M");
  in.close();

  /**
   * Set the property of the full matrix as @p symmetric.
   */
  M.set_property(LAPACKSupport::symmetric);

  /**
   * Create a rank-1 HMatrix, whose property is automatically set to @p symmetric.
   */
  const unsigned int fixed_rank_k = 1;
  HMatrix<3, double>::set_leaf_set_traversal_method(
    HMatrix<3, double>::SpaceFillingCurveType::Hilbert);
  HMatrix<3, double> hmat(block_cluster_tree, M, fixed_rank_k);

  /**
   * Write out the leaf nodes by iteration.
   */
  hmat.write_leaf_set_by_iteration(std::cout);

  return 0;
}
