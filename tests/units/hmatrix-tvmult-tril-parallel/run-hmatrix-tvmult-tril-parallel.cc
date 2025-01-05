#include <catch2/catch_all.hpp>
#include <openblas-pthread/cblas.h>

#include <fstream>

#include "hmatrix/hmatrix.h"
#include "lapack_full_matrix_ext.h"
#include "read_octave_data.h"

using namespace HierBEM;
using namespace Catch::Matchers;

void
run_hmatrix_tvmult_tril_parallel()
{
  std::ofstream ofs("hmatrix-tvmult-tril-parallel.output");

  /**
   * Load a matrix where only the lower triangular and diagonal parts are
   * stored.
   */
  LAPACKFullMatrixExt<double> M;
  std::ifstream               in("M.dat");
  M.read_from_mat(in, "M");
  in.close();
  REQUIRE(M.size()[0] > 0);
  REQUIRE(M.size()[0] == M.size()[1]);

  /**
   * Set the property of the full matrix as @p lower_triangular.
   */
  M.set_property(LAPACKSupport::Property::lower_triangular);

  /**
   * Read the vector \f$x\f$.
   */
  Vector<double> x;
  in.open("xy.dat");
  read_vector_from_octave(in, "x", x);
  in.close();
  REQUIRE(x.size() == M.size()[0]);

  /**
   * Read the initial values of the vector \f$y\f$.
   */
  Vector<double> y;
  in.open("xy.dat");
  read_vector_from_octave(in, "y0", y);
  in.close();
  REQUIRE(y.size() == M.size()[1]);

  /**
   * Generate index set.
   */
  const unsigned int                   p = 6;
  const unsigned int                   n = std::pow(2, p);
  std::vector<types::global_dof_index> index_set(n);

  for (unsigned int i = 0; i < n; i++)
    {
      index_set.at(i) = i;
    }

  /**
   * Generate cluster tree.
   */
  const unsigned int n_min = 2;
  ClusterTree<3>     cluster_tree(index_set, n_min);
  cluster_tree.partition();

  /**
   * Generate block cluster tree with the two component cluster trees being the
   * same.
   */
  BlockClusterTree<3, double> block_cluster_tree(cluster_tree, cluster_tree);
  block_cluster_tree.partition_fine_non_tensor_product();

  /**
   * Generate the \hmatrix from the lower triangular full matrix. Its property
   * will automatically be set to @p HMatrixSupport::Property::lower_triangular.
   * The leaf set traversal method should be set to Hilbert, so that the row and
   * column index sets of leaf \hmatrix nodes in a same interval obtained from
   * sequence partition are contiguous respectively. This will reduce the size
   * of the local result vector on each thread.
   */
  const unsigned int fixed_rank_k = n / 4;
  HMatrix<3, double>::set_leaf_set_traversal_method(
    HMatrix<3, double>::SpaceFillingCurveType::Hilbert);
  HMatrix<3, double> H(block_cluster_tree, M, fixed_rank_k);
  REQUIRE(H.get_m() == M.size()[0]);
  REQUIRE(H.get_n() == M.size()[1]);

  /**
   * Convert the \hmatrix back to full matrix for comparison with the original
   * full matrix.
   */
  LAPACKFullMatrixExt<double> H_full;
  H.convertToFullMatrix(H_full);
  REQUIRE(H_full.size()[0] == M.size()[0]);
  REQUIRE(H_full.size()[1] == M.size()[1]);

  H_full.print_formatted_to_mat(ofs, "H_full", 15, false, 25, "0");

  /**
   * Limit the number of OpenBLAS threads.
   */
  openblas_set_num_threads(1);

  /**
   * Perform transposed \hmatrix/vector multiplication.
   */
  H.prepare_for_vmult_or_tvmult(false, true);
  H.Tvmult_task_parallel(0.3, y, 1.5, x);
  print_vector_to_mat(ofs, "y1_cpp", y);

  H.Tvmult_task_parallel(3.7, y, 8.2, x);
  print_vector_to_mat(ofs, "y2_cpp", y);

  ofs.close();
}
