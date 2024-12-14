#include <catch2/catch_all.hpp>
#include <openblas-pthread/cblas.h>

#include <fstream>

#include "cu_debug_tools.hcu"
#include "hmatrix.h"
#include "lapack_full_matrix_ext.h"
#include "read_octave_data.h"

using namespace HierBEM;
using namespace Catch::Matchers;

void
run_hmatrix_solve_cholesky_task_parallel(const unsigned int trial_no)
{
  std::ofstream ofs(std::string("hmatrix-solve-cholesky-task-parallel-") +
                    std::to_string(trial_no) + std::string(".output"));

  /**
   * @internal Set number of threads used for OpenBLAS.
   */
  openblas_set_num_threads(1);

  LAPACKFullMatrixExt<double> M;
  std::ifstream               in(std::string("M") + std::to_string(trial_no) +
                   std::string(".dat"));
  M.read_from_mat(in, "M");
  M.set_property(LAPACKSupport::symmetric);
  in.close();
  REQUIRE(M.size()[0] > 0);
  REQUIRE(M.size()[0] == M.size()[1]);

  Vector<double> b;
  in.open(std::string("b") + std::to_string(trial_no) + std::string(".dat"));
  read_vector_from_octave(in, "b", b);
  in.close();
  REQUIRE(b.size() == M.size()[0]);

  /**
   * Generate index set.
   */
  const unsigned int                   n = M.m();
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
   * Create the \hmatrix.
   */
  const unsigned int fixed_rank = n / 4;
  HMatrix<3, double> H(bct, M, fixed_rank);
  REQUIRE(H.get_m() == M.size()[0]);
  REQUIRE(H.get_n() == M.size()[1]);

  HMatrix<3, double> H_serial(bct, M, fixed_rank);
  REQUIRE(H_serial.get_m() == M.size()[0]);
  REQUIRE(H_serial.get_n() == M.size()[1]);

  std::ofstream H_bct("H_bct.dat");
  H.write_leaf_set_by_iteration(H_bct);
  H_bct.close();

  LAPACKFullMatrixExt<double> H_full;
  H.convertToFullMatrix(H_full);
  REQUIRE(H_full.size()[0] == M.size()[0]);
  REQUIRE(H_full.size()[1] == M.size()[1]);

  H_full.print_formatted_to_mat(ofs, "H_full", 15, false, 25, "0");

  /**
   * Perform Cholesky factorization in serial.
   */
  H_serial.compute_cholesky_factorization(fixed_rank);

  /**
   * Perform Cholesky factorization in task parallel.
   */
  H.compute_cholesky_factorization_task_parallel(fixed_rank);
  ofs << "H's state after Cholesky factorization: "
      << HMatrixSupport::state_name(H.get_state()) << std::endl;

  /**
   * Convert the \Hcal-Cholesky matrix to full matrix.
   */
  LAPACKFullMatrixExt<double> L_full_serial;
  H_serial.convertToFullMatrix(L_full_serial);
  REQUIRE(L_full_serial.size()[0] == M.size()[0]);
  REQUIRE(L_full_serial.size()[1] == M.size()[1]);

  LAPACKFullMatrixExt<double> L_full;
  H.convertToFullMatrix(L_full);
  REQUIRE(L_full.size()[0] == M.size()[0]);
  REQUIRE(L_full.size()[1] == M.size()[1]);

  L_full.print_formatted_to_mat(ofs, "L_full", 15, false, 25, "0");
  L_full_serial.print_formatted_to_mat(
    ofs, "L_full_serial", 15, false, 25, "0");

  /**
   * Solve the matrix.
   */
  Vector<double> x;
  H.solve_cholesky(x, b);
  REQUIRE(x.size() == b.size());

  /**
   * Print the result vector.
   */
  print_vector_to_mat(ofs, "x", x);

  ofs.close();
}
