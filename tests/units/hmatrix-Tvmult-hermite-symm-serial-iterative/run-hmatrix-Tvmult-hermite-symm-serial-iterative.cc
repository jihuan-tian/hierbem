#include <catch2/catch_all.hpp>

#include <complex>
#include <fstream>

#include "hmatrix/hmatrix.h"
#include "lapack_full_matrix_ext.h"
#include "read_octave_data.h"

using namespace HierBEM;
using namespace Catch::Matchers;

void
run_hmatrix_Tvmult_hermite_symm_serial_iterative()
{
  std::ofstream ofs("hmatrix-Tvmult-hermite-symm-serial-iterative.output");

  /**
   * Load an Hermite symmetric matrix, where only diagonal and lower triangular
   * parts are stored.
   */
  LAPACKFullMatrixExt<std::complex<double>> M_complex;

  std::ifstream in("M.dat");
  M_complex.read_from_mat(in, "M_complex");
  in.close();

  REQUIRE(M_complex.size()[0] > 0);
  REQUIRE(M_complex.size()[0] == M_complex.size()[1]);

  /**
   * Set the property of the full matrix as @p hermite_symmetric.
   */
  M_complex.set_property(LAPACKSupport::Property::hermite_symmetric);

  /**
   * Read the vector \f$x\f$ and the initial values of the vector \f$y\f$.
   */
  Vector<std::complex<double>> x_complex;
  Vector<std::complex<double>> y_complex;

  in.open("xy.dat");
  read_vector_from_octave(in, "x_complex", x_complex);
  read_vector_from_octave(in, "y0_complex", y_complex);
  in.close();

  REQUIRE(x_complex.size() == M_complex.size()[1]);
  REQUIRE(y_complex.size() == M_complex.size()[0]);

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
   * Generate block cluster tree with the two component cluster trees being the
   * same.
   */
  BlockClusterTree<spacedim, double> block_cluster_tree(cluster_tree,
                                                        cluster_tree);
  block_cluster_tree.partition_fine_non_tensor_product();

  /**
   * Generate the \hmatrix from the Hermite symmetric full matrix. Its property
   * will
   * automatically be set to @p HMatrixSupport::Property::hermite_symmetric. The leaf
   * set traversal method should be set to Hilbert, so that the row and column
   * index sets of leaf \hmatrix nodes in a same interval obtained from sequence
   * partition are contiguous respectively. This will reduce the size of the
   * local result vector on each thread.
   */
  const unsigned int                      fixed_rank_k = n / 4;
  HMatrix<spacedim, std::complex<double>> H_complex(block_cluster_tree,
                                                    M_complex,
                                                    fixed_rank_k);

  REQUIRE(H_complex.get_property() ==
          HMatrixSupport::Property::hermite_symmetric);
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
   * Perform \hmatrix/vector multiplication.
   */
  H_complex.Tvmult_serial_iterative(0.3, y_complex, 1.5, x_complex);
  print_vector_to_mat(ofs, "y1_cpp_complex", y_complex);
  H_complex.Tvmult_serial_iterative(std::complex<double>(0.3, 0.2),
                                    y_complex,
                                    std::complex<double>(1.5, 2.1),
                                    x_complex);
  print_vector_to_mat(ofs, "y2_cpp_complex", y_complex);
  H_complex.Tvmult_serial_iterative(std::complex<double>(0.3, 0.2),
                                    y_complex,
                                    1.5,
                                    x_complex);
  print_vector_to_mat(ofs, "y3_cpp_complex", y_complex);
  H_complex.Tvmult_serial_iterative(0.3,
                                    y_complex,
                                    std::complex<double>(1.5, 2.1),
                                    x_complex);
  print_vector_to_mat(ofs, "y4_cpp_complex", y_complex);

  ofs.close();
}
