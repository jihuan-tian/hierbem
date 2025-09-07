#include <catch2/catch_all.hpp>

#include <fstream>

#include "hmatrix/hmatrix.h"
#include "linear_algebra/lapack_full_matrix_ext.h"
#include "utilities/cu_debug_tools.hcu"
#include "utilities/read_octave_data.h"

using namespace HierBEM;
using namespace Catch::Matchers;

void
run_hmatrix_solve_lu_task_parallel(const unsigned int trial_no)
{
  std::ofstream ofs(std::string("hmatrix-solve-lu-task-parallel-") +
                    std::to_string(trial_no) + std::string(".output"));

  LAPACKFullMatrixExt<double> M;
  std::ifstream               in(std::string("M") + std::to_string(trial_no) +
                   std::string(".dat"));
  M.read_from_mat(in, "M");
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
   * Perform LU factorization in serial.
   */
  H_serial.compute_lu_factorization(fixed_rank);

  /**
   * Perform LU factorization in task parallel.
   */
  H.compute_lu_factorization_task_parallel(fixed_rank);
  ofs << "H's state after LU factorization: "
      << HMatrixSupport::state_name(H.get_state()) << std::endl;

  /**
   * Convert the \Hcal-LU matrix to full matrix.
   */
  LAPACKFullMatrixExt<double> LU_full_serial;
  H_serial.convertToFullMatrix(LU_full_serial);
  REQUIRE(LU_full_serial.size()[0] == M.size()[0]);
  REQUIRE(LU_full_serial.size()[1] == M.size()[1]);

  LAPACKFullMatrixExt<double> LU_full;
  H.convertToFullMatrix(LU_full);
  REQUIRE(LU_full.size()[0] == M.size()[0]);
  REQUIRE(LU_full.size()[1] == M.size()[1]);

  LU_full.print_formatted_to_mat(ofs, "LU_full", 15, false, 25, "0");
  LU_full_serial.print_formatted_to_mat(
    ofs, "LU_full_serial", 15, false, 25, "0");

  /**
   * Solve the matrix.
   */
  Vector<double> x;
  H.solve_lu(x, b);
  REQUIRE(x.size() == b.size());

  /**
   * Print the result vector.
   */
  print_vector_to_mat(ofs, "x", x);

  ofs.close();
}
