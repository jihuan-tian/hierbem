/**
 * \file hmatrix-memory-consumption.cc
 * \brief Verify the memory consumption calculation for \hmatrix.
 *
 * \ingroup testers hierarchical_matrices
 * \author Jihuan Tian
 * \date 2022-05-06
 */

#include <cmath>
#include <iostream>

#include "hmatrix.h"

using namespace HierBEM;

int
main()
{
  std::cout << "# Matrix size,Memory (bytes)" << std::endl;

  {
    const unsigned int p = 4;
    const unsigned int n = std::pow(2, p);

    /**
     * Generate index set.
     */
    std::vector<types::global_dof_index> index_set(n);

    for (unsigned int i = 0; i < n; i++)
      {
        index_set.at(i) = i;
      }

    const unsigned int n_min        = 2;
    const unsigned int fixed_rank_k = 2;

    /**
     * Generate cluster tree.
     */
    ClusterTree<3> cluster_tree(index_set, n_min);
    cluster_tree.partition();

    /**
     * Generate block cluster tree via fine structured non-tensor product
     * partition.
     */
    BlockClusterTree<3, double> block_cluster_tree(cluster_tree, cluster_tree);
    block_cluster_tree.partition_fine_non_tensor_product();

    /**
     * Create the full matrix as the source data.
     */
    LAPACKFullMatrixExt<double> M(n, n);
    double                      counter = 1.0;
    for (auto it = M.begin(); it != M.end(); it++)
      {
        (*it) = counter * std::sin(counter);
        counter += 1.0;
      }

    /**
     * Create the \hmatrix from @p M.
     */
    HMatrix<3, double> H(block_cluster_tree, M, fixed_rank_k);
    std::cout << n << "," << H.memory_consumption() << std::endl;
  }

  {
    const unsigned int p = 5;
    const unsigned int n = std::pow(2, p);

    /**
     * Generate index set.
     */
    std::vector<types::global_dof_index> index_set(n);

    for (unsigned int i = 0; i < n; i++)
      {
        index_set.at(i) = i;
      }

    const unsigned int n_min        = 2;
    const unsigned int fixed_rank_k = 2;

    /**
     * Generate cluster tree.
     */
    ClusterTree<3> cluster_tree(index_set, n_min);
    cluster_tree.partition();

    /**
     * Generate block cluster tree via fine structured non-tensor product
     * partition.
     */
    BlockClusterTree<3, double> block_cluster_tree(cluster_tree, cluster_tree);
    block_cluster_tree.partition_fine_non_tensor_product();

    /**
     * Create the full matrix as the source data.
     */
    LAPACKFullMatrixExt<double> M(n, n);
    double                      counter = 1.0;
    for (auto it = M.begin(); it != M.end(); it++)
      {
        (*it) = counter * std::sin(counter);
        counter += 1.0;
      }

    /**
     * Create the \hmatrix from @p M.
     */
    HMatrix<3, double> H(block_cluster_tree, M, fixed_rank_k);
    std::cout << n << "," << H.memory_consumption() << std::endl;
  }

  {
    const unsigned int p = 6;
    const unsigned int n = std::pow(2, p);

    /**
     * Generate index set.
     */
    std::vector<types::global_dof_index> index_set(n);

    for (unsigned int i = 0; i < n; i++)
      {
        index_set.at(i) = i;
      }

    const unsigned int n_min        = 2;
    const unsigned int fixed_rank_k = 2;

    /**
     * Generate cluster tree.
     */
    ClusterTree<3> cluster_tree(index_set, n_min);
    cluster_tree.partition();

    /**
     * Generate block cluster tree via fine structured non-tensor product
     * partition.
     */
    BlockClusterTree<3, double> block_cluster_tree(cluster_tree, cluster_tree);
    block_cluster_tree.partition_fine_non_tensor_product();

    /**
     * Create the full matrix as the source data.
     */
    LAPACKFullMatrixExt<double> M(n, n);
    double                      counter = 1.0;
    for (auto it = M.begin(); it != M.end(); it++)
      {
        (*it) = counter * std::sin(counter);
        counter += 1.0;
      }

    /**
     * Create the \hmatrix from @p M.
     */
    HMatrix<3, double> H(block_cluster_tree, M, fixed_rank_k);
    std::cout << n << "," << H.memory_consumption() << std::endl;
  }

  {
    const unsigned int p = 7;
    const unsigned int n = std::pow(2, p);

    /**
     * Generate index set.
     */
    std::vector<types::global_dof_index> index_set(n);

    for (unsigned int i = 0; i < n; i++)
      {
        index_set.at(i) = i;
      }

    const unsigned int n_min        = 2;
    const unsigned int fixed_rank_k = 2;

    /**
     * Generate cluster tree.
     */
    ClusterTree<3> cluster_tree(index_set, n_min);
    cluster_tree.partition();

    /**
     * Generate block cluster tree via fine structured non-tensor product
     * partition.
     */
    BlockClusterTree<3, double> block_cluster_tree(cluster_tree, cluster_tree);
    block_cluster_tree.partition_fine_non_tensor_product();

    /**
     * Create the full matrix as the source data.
     */
    LAPACKFullMatrixExt<double> M(n, n);
    double                      counter = 1.0;
    for (auto it = M.begin(); it != M.end(); it++)
      {
        (*it) = counter * std::sin(counter);
        counter += 1.0;
      }

    /**
     * Create the \hmatrix from @p M.
     */
    HMatrix<3, double> H(block_cluster_tree, M, fixed_rank_k);
    std::cout << n << "," << H.memory_consumption() << std::endl;
  }

  {
    const unsigned int p = 8;
    const unsigned int n = std::pow(2, p);

    /**
     * Generate index set.
     */
    std::vector<types::global_dof_index> index_set(n);

    for (unsigned int i = 0; i < n; i++)
      {
        index_set.at(i) = i;
      }

    const unsigned int n_min        = 2;
    const unsigned int fixed_rank_k = 2;

    /**
     * Generate cluster tree.
     */
    ClusterTree<3> cluster_tree(index_set, n_min);
    cluster_tree.partition();

    /**
     * Generate block cluster tree via fine structured non-tensor product
     * partition.
     */
    BlockClusterTree<3, double> block_cluster_tree(cluster_tree, cluster_tree);
    block_cluster_tree.partition_fine_non_tensor_product();

    /**
     * Create the full matrix as the source data.
     */
    LAPACKFullMatrixExt<double> M(n, n);
    double                      counter = 1.0;
    for (auto it = M.begin(); it != M.end(); it++)
      {
        (*it) = counter * std::sin(counter);
        counter += 1.0;
      }

    /**
     * Create the \hmatrix from @p M.
     */
    HMatrix<3, double> H(block_cluster_tree, M, fixed_rank_k);
    std::cout << n << "," << H.memory_consumption() << std::endl;
  }

  {
    const unsigned int p = 9;
    const unsigned int n = std::pow(2, p);

    /**
     * Generate index set.
     */
    std::vector<types::global_dof_index> index_set(n);

    for (unsigned int i = 0; i < n; i++)
      {
        index_set.at(i) = i;
      }

    const unsigned int n_min        = 2;
    const unsigned int fixed_rank_k = 2;

    /**
     * Generate cluster tree.
     */
    ClusterTree<3> cluster_tree(index_set, n_min);
    cluster_tree.partition();

    /**
     * Generate block cluster tree via fine structured non-tensor product
     * partition.
     */
    BlockClusterTree<3, double> block_cluster_tree(cluster_tree, cluster_tree);
    block_cluster_tree.partition_fine_non_tensor_product();

    /**
     * Create the full matrix as the source data.
     */
    LAPACKFullMatrixExt<double> M(n, n);
    double                      counter = 1.0;
    for (auto it = M.begin(); it != M.end(); it++)
      {
        (*it) = counter * std::sin(counter);
        counter += 1.0;
      }

    /**
     * Create the \hmatrix from @p M.
     */
    HMatrix<3, double> H(block_cluster_tree, M, fixed_rank_k);
    std::cout << n << "," << H.memory_consumption() << std::endl;
  }

  {
    const unsigned int p = 10;
    const unsigned int n = std::pow(2, p);

    /**
     * Generate index set.
     */
    std::vector<types::global_dof_index> index_set(n);

    for (unsigned int i = 0; i < n; i++)
      {
        index_set.at(i) = i;
      }

    const unsigned int n_min        = 2;
    const unsigned int fixed_rank_k = 2;

    /**
     * Generate cluster tree.
     */
    ClusterTree<3> cluster_tree(index_set, n_min);
    cluster_tree.partition();

    /**
     * Generate block cluster tree via fine structured non-tensor product
     * partition.
     */
    BlockClusterTree<3, double> block_cluster_tree(cluster_tree, cluster_tree);
    block_cluster_tree.partition_fine_non_tensor_product();

    /**
     * Create the full matrix as the source data.
     */
    LAPACKFullMatrixExt<double> M(n, n);
    double                      counter = 1.0;
    for (auto it = M.begin(); it != M.end(); it++)
      {
        (*it) = counter * std::sin(counter);
        counter += 1.0;
      }

    /**
     * Create the \hmatrix from @p M.
     */
    HMatrix<3, double> H(block_cluster_tree, M, fixed_rank_k);
    std::cout << n << "," << H.memory_consumption() << std::endl;
  }

  {
    const unsigned int p = 11;
    const unsigned int n = std::pow(2, p);

    /**
     * Generate index set.
     */
    std::vector<types::global_dof_index> index_set(n);

    for (unsigned int i = 0; i < n; i++)
      {
        index_set.at(i) = i;
      }

    const unsigned int n_min        = 2;
    const unsigned int fixed_rank_k = 2;

    /**
     * Generate cluster tree.
     */
    ClusterTree<3> cluster_tree(index_set, n_min);
    cluster_tree.partition();

    /**
     * Generate block cluster tree via fine structured non-tensor product
     * partition.
     */
    BlockClusterTree<3, double> block_cluster_tree(cluster_tree, cluster_tree);
    block_cluster_tree.partition_fine_non_tensor_product();

    /**
     * Create the full matrix as the source data.
     */
    LAPACKFullMatrixExt<double> M(n, n);
    double                      counter = 1.0;
    for (auto it = M.begin(); it != M.end(); it++)
      {
        (*it) = counter * std::sin(counter);
        counter += 1.0;
      }

    /**
     * Create the \hmatrix from @p M.
     */
    HMatrix<3, double> H(block_cluster_tree, M, fixed_rank_k);
    std::cout << n << "," << H.memory_consumption() << std::endl;
  }

  {
    const unsigned int p = 12;
    const unsigned int n = std::pow(2, p);

    /**
     * Generate index set.
     */
    std::vector<types::global_dof_index> index_set(n);

    for (unsigned int i = 0; i < n; i++)
      {
        index_set.at(i) = i;
      }

    const unsigned int n_min        = 2;
    const unsigned int fixed_rank_k = 2;

    /**
     * Generate cluster tree.
     */
    ClusterTree<3> cluster_tree(index_set, n_min);
    cluster_tree.partition();

    /**
     * Generate block cluster tree via fine structured non-tensor product
     * partition.
     */
    BlockClusterTree<3, double> block_cluster_tree(cluster_tree, cluster_tree);
    block_cluster_tree.partition_fine_non_tensor_product();

    /**
     * Create the full matrix as the source data.
     */
    LAPACKFullMatrixExt<double> M(n, n);
    double                      counter = 1.0;
    for (auto it = M.begin(); it != M.end(); it++)
      {
        (*it) = counter * std::sin(counter);
        counter += 1.0;
      }

    /**
     * Create the \hmatrix from @p M.
     */
    HMatrix<3, double> H(block_cluster_tree, M, fixed_rank_k);
    std::cout << n << "," << H.memory_consumption() << std::endl;
  }

  {
    const unsigned int p = 13;
    const unsigned int n = std::pow(2, p);

    /**
     * Generate index set.
     */
    std::vector<types::global_dof_index> index_set(n);

    for (unsigned int i = 0; i < n; i++)
      {
        index_set.at(i) = i;
      }

    const unsigned int n_min        = 2;
    const unsigned int fixed_rank_k = 2;

    /**
     * Generate cluster tree.
     */
    ClusterTree<3> cluster_tree(index_set, n_min);
    cluster_tree.partition();

    /**
     * Generate block cluster tree via fine structured non-tensor product
     * partition.
     */
    BlockClusterTree<3, double> block_cluster_tree(cluster_tree, cluster_tree);
    block_cluster_tree.partition_fine_non_tensor_product();

    /**
     * Create the full matrix as the source data.
     */
    LAPACKFullMatrixExt<double> M(n, n);
    double                      counter = 1.0;
    for (auto it = M.begin(); it != M.end(); it++)
      {
        (*it) = counter * std::sin(counter);
        counter += 1.0;
      }

    /**
     * Create the \hmatrix from @p M.
     */
    HMatrix<3, double> H(block_cluster_tree, M, fixed_rank_k);
    std::cout << n << "," << H.memory_consumption() << std::endl;
  }

  return 0;
}
