#include <catch2/catch_all.hpp>

#include <complex>
#include <fstream>

#include "hmatrix/hmatrix.h"
#include "linear_algebra/lapack_full_matrix_ext.h"
#include "utilities/read_octave_data.h"

using namespace HierBEM;
using namespace Catch::Matchers;

void
run_hmatrix_Hvmult()
{
  std::ofstream ofs("hmatrix-Hvmult.output");

  /**
   * Load a general matrix.
   */
  LAPACKFullMatrixExt<std::complex<double>> M_complex;

  std::ifstream in("M.dat");
  M_complex.read_from_mat(in, "M_complex");
  in.close();

  REQUIRE(M_complex.size()[0] > 0);
  REQUIRE(M_complex.size()[0] == M_complex.size()[1]);

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
  const unsigned int            spacedim = 3;
  const unsigned int            n_min    = 2;
  ClusterTree<spacedim, double> cluster_tree(index_set, n_min);
  cluster_tree.partition();

  /**
   * Generate a block cluster tree with the two component cluster trees being
   * the same.
   */
  BlockClusterTree<spacedim, double> block_cluster_tree(cluster_tree,
                                                        cluster_tree);
  block_cluster_tree.partition_fine_non_tensor_product();

  /**
   * Create a rank-k HMatrix.
   */
  const unsigned int                      fixed_rank_k = n / 4;
  HMatrix<spacedim, std::complex<double>> H_complex(block_cluster_tree,
                                                    M_complex,
                                                    fixed_rank_k);

  REQUIRE(H_complex.get_property() == HMatrixSupport::Property::general);
  REQUIRE(H_complex.get_m() == M_complex.size()[0]);
  REQUIRE(H_complex.get_n() == M_complex.size()[1]);

  /**
   * Convert the \hmatrix back to full matrix for comparison with the original
   * full matrix.
   */
  LAPACKFullMatrixExt<std::complex<double>> H_full_complex;

  H_complex.convertToFullMatrix(H_full_complex);

  REQUIRE(H_full_complex.size()[0] == M_complex.size()[0]);
  REQUIRE(H_full_complex.size()[1] == M_complex.size()[1]);

  H_full_complex.print_formatted_to_mat(
    ofs, "H_full_complex", 15, false, 45, "0");

  /**
   * Read the vector \f$x\f$.
   */
  Vector<std::complex<double>> x_complex;

  in.open("x.dat");
  read_vector_from_octave(in, "x_complex", x_complex);
  in.close();

  REQUIRE(x_complex.size() == M_complex.size()[1]);

  /**
   * Perform \hmatrix/vector multiplication.
   */
  Vector<std::complex<double>> y_complex(n);
  const double                 factor = 0.5;
  const std::complex<double>   factor_complex(0.5, 0.3);

  H_complex.Hvmult(y_complex, x_complex);
  print_vector_to_mat(ofs, "y1_complex", y_complex);
  H_complex.Hvmult(y_complex, factor, x_complex);
  print_vector_to_mat(ofs, "y2_complex", y_complex);
  H_complex.Hvmult(y_complex, factor_complex, x_complex);
  print_vector_to_mat(ofs, "y3_complex", y_complex);

  ofs.close();
}
