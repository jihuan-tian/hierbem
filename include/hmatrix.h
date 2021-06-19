/**
 * \file hmatrix.h
 * \brief Definition of hierarchical matrix.
 * \ingroup hierarchical_matrices
 * \date 2021-06-06
 * \author Jihuan Tian
 */

#ifndef INCLUDE_HMATRIX_H_
#define INCLUDE_HMATRIX_H_

#include <deal.II/lac/full_matrix.h>

#include <vector>

#include "block_cluster.h"
#include "block_cluster_tree.h"
#include "rkmatrix.h"

using namespace dealii;

/**
 * Matrix type of an HMaxtrix, which can be full matrix in the near field,
 * rank-k matrix in the far field and hierarchical matrix which does not
 * belong to the leaf set of a block cluster tree.
 */
enum HMatrixType
{
  FullMatrixType,     //!< FullMatrixType
  RkMatrixType,       //!< RkMatrixType
  HierarchicalType,   //!< HierarchicalType
  UndefinedMatrixType //!< UndefinedMatrixType
};

template <int spacedim, typename Number = double>
class HMatrix
{
public:
  template <int spacedim1, typename Number1>
  friend void
  InitAndCreateHMatrixChildren(
    HMatrix<spacedim1, Number1> *hmat,
    typename BlockClusterTree<spacedim1, Number1>::node_const_pointer_type
                 bc_node,
    unsigned int fixed_rank_k);

  template <int spacedim1, typename Number1>
  friend void
  InitAndCreateHMatrixChildren(
    HMatrix<spacedim1, Number1> *hmat,
    typename BlockClusterTree<spacedim1, Number1>::node_const_pointer_type
                              bc_node,
    unsigned int              fixed_rank_k,
    const FullMatrix<Number> &M);

  /**
   * Default constructor.
   */
  HMatrix();

  /**
   * Construct the hierarchical structure without data from the root node of a
   * BlockClusterTree.
   */
  HMatrix(const BlockClusterTree<spacedim, Number> &bct);

  /**
   * Construct the hierarchical structure without data from a TreeNode in a
   * BlockClusterTree.
   */
  HMatrix(typename BlockClusterTree<spacedim, Number>::node_const_pointer_type
            bc_node);

  /**
   * Construct from the root node of a BlockClusterTree while copying the data
   * of a FullMatrix.
   */
  HMatrix(const BlockClusterTree<spacedim, Number> &bct,
          const FullMatrix<Number> &                M);

  /**
   * Construct from a TreeNode in a BlockClusterTree while copying the data of a
   * FullMatrix.
   */
  HMatrix(typename BlockClusterTree<spacedim, Number>::node_const_pointer_type
                                    bc_node,
          const FullMatrix<Number> &M);

  /**
   * TODO: Construct from a BlockClusterTree and a Sauter quadrature
   * object/functor.
   */

  /**
   * Destructor which releases the memory by recursion.
   */
  ~HMatrix();

private:
  /**
   * Matrix type.
   */
  HMatrixType type;

  /**
   * A list of submatrices of type HMatrix.
   */
  std::vector<HMatrix *> submatrices;

  /**
   * Pointer to the rank-k matrix. It is not null when the current HMatrix
   * object belongs to the far field.
   */
  RkMatrix<Number> *rkmatrix;

  /**
   * Pointer to the full matrix. It is not null when the current HMatrix object
   * belongs to the near field.
   */
  FullMatrix<Number> *fullmatrix;

  /**
   * Pointer to the corresponding block cluster node in a BlockClusterTree.
   */
  typename BlockClusterTree<spacedim, Number>::node_pointer_type bc_node;

  /**
   * Row indices.
   */
  std::vector<types::global_dof_index> *row_indices;

  /**
   * Column indices.
   */
  std::vector<types::global_dof_index> *col_indices;

  /**
   * Total number of rows in the matrix.
   */
  unsigned int m;

  /**
   * Total number of columns in the matrix.
   */
  unsigned int n;
};


/**
 * Recursively construct the children of an HMatrix with respect to a block
 * cluster tree by starting from a tree node. The matrices in the leaf set are
 * initialized with zero values. The rank of the near field matrices are
 * predefined fixed values.
 * @param hmat Pointer to the current HMatrix.
 * @param bc_node Pointer to a TreeNode in a BlockClusterTree.
 */
template <int spacedim, typename Number = double>
void
InitAndCreateHMatrixChildren(
  HMatrix<spacedim, Number> *                                          hmat,
  typename BlockClusterTree<spacedim, Number>::node_const_pointer_type bc_node,
  unsigned int fixed_rank_k)
{
  /**
   * Link \p hmat with \p bc_node.
   */
  hmat->bc_node =
    const_cast<typename BlockClusterTree<spacedim, Number>::node_pointer_type>(
      bc_node);

  /**
   * Link row and column indices.
   */
  hmat->row_indices = &(bc_node->get_data_reference()
                          .get_tau_node()
                          ->get_data_reference()
                          .get_index_set());
  hmat->col_indices = &(bc_node->get_data_reference()
                          .get_sigma_node()
                          ->get_data_reference()
                          .get_index_set());

  /**
   * Update the matrix dimension of \p hmat.
   */
  hmat->m = hmat->row_indices->size();
  hmat->n = hmat->col_indices->size();

  const unsigned int bc_node_child_num = bc_node->get_child_num();
  if (bc_node_child_num > 0)
    {
      /**
       * When the block cluster node \p bc_node has children, set the current \p
       * hmat type as \p HierarchicalType. Then we will
       * continue constructing hierarchical submatrices.
       */
      hmat->type = HierarchicalType;

      for (unsigned int i = 0; i < bc_node_child_num; i++)
        {
          /**
           * Create an empty HMatrix on the heap.
           */
          HMatrix<spacedim, Number> *child_hmat =
            new HMatrix<spacedim, Number>();

          InitAndCreateHMatrixChildren(child_hmat,
                                       bc_node->get_child_pointer(i));

          /**
           * Append the initialized child to the list of submatrices of \p hmat.
           */
          hmat->submatrices.push_back(child_hmat);
        }
    }
  else
    {
      /**
       * Update the current matrix type from the block cluster node. When the
       * block cluster belongs to the near field, \p hmat should be represented
       * as a \p FullMatrix. When the block cluster belongs to the far field, \p
       * hmat should be represented as a \p RkMatrix.
       */
      if (bc_node->get_data_reference().get_is_near_field())
        {
          hmat->type       = FullMatrixType;
          hmat->fullmatrix = new FullMatrix<Number>(hmat->m, hmat->n);
        }
      else
        {
          hmat->type     = RkMatrixType;
          hmat->rkmatrix = new RkMatrix<Number>(hmat->m, hmat->n, fixed_rank_k);
        }
    }
}


/**
 * Recursively construct the children of an HMatrix with respect to a block
 * cluster tree by starting from a tree node. The matrices in the leaf set are
 * initialized with the data in the given \p full_matrix. The rank of the near
 * field matrices are predefined fixed values.
 * @param hmat Pointer to the current HMatrix.
 * @param bc_node Pointer to a TreeNode in a BlockClusterTree.
 */
template <int spacedim, typename Number = double>
void
InitAndCreateHMatrixChildren(
  HMatrix<spacedim, Number> *                                          hmat,
  typename BlockClusterTree<spacedim, Number>::node_const_pointer_type bc_node,
  unsigned int              fixed_rank_k,
  const FullMatrix<Number> &M)
{
  /**
   * Link \p hmat with \p bc_node.
   */
  hmat->bc_node =
    const_cast<typename BlockClusterTree<spacedim, Number>::node_pointer_type>(
      bc_node);

  /**
   * Link row and column indices.
   */
  hmat->row_indices = &(bc_node->get_data_reference()
                          .get_tau_node()
                          ->get_data_reference()
                          .get_index_set());
  hmat->col_indices = &(bc_node->get_data_reference()
                          .get_sigma_node()
                          ->get_data_reference()
                          .get_index_set());

  /**
   * Update the matrix dimension of \p hmat.
   */
  hmat->m = hmat->row_indices->size();
  hmat->n = hmat->col_indices->size();

  const unsigned int bc_node_child_num = bc_node->get_child_num();
  if (bc_node_child_num > 0)
    {
      /**
       * When the block cluster node \p bc_node has children, set the current \p
       * hmat type as \p HierarchicalType. Then we will
       * continue constructing hierarchical submatrices.
       */
      hmat->type = HierarchicalType;

      for (unsigned int i = 0; i < bc_node_child_num; i++)
        {
          /**
           * Create an empty HMatrix on the heap.
           */
          HMatrix<spacedim, Number> *child_hmat =
            new HMatrix<spacedim, Number>();

          InitAndCreateHMatrixChildren(child_hmat,
                                       bc_node->get_child_pointer(i));

          /**
           * Append the initialized child to the list of submatrices of \p hmat.
           */
          hmat->submatrices.push_back(child_hmat);
        }
    }
  else
    {
      /**
       * Update the current matrix type from the block cluster node. When the
       * block cluster belongs to the near field, \p hmat should be represented
       * as a \p FullMatrix. When the block cluster belongs to the far field, \p
       * hmat should be represented as a \p RkMatrix.
       */
      if (bc_node->get_data_reference().get_is_near_field())
        {
          hmat->type       = FullMatrixType;
          hmat->fullmatrix = new FullMatrix<Number>(hmat->m, hmat->n);

          /**
           * Assign matrix values from \p full_matrix to the current HMatrix.
           */
          for (unsigned int i = 0; i < hmat->m; i++)
            {
              for (unsigned int j = 0; j < hmat->n; j++)
                {
                  hmat->fullmatrix(i, j) =
                    M(hmat->row_indices->at(i), hmat->col_indices->at(j));
                }
            }
        }
      else
        {
          hmat->type     = RkMatrixType;
          hmat->rkmatrix = new RkMatrix<Number>(*(hmat->row_indices),
                                                *(hmat->col_indices),
                                                fixed_rank_k,
                                                M);
        }
    }
}


template <int spacedim, typename Number>
HMatrix<spacedim, Number>::HMatrix()
  : type(UndefinedMatrixType)
  , submatrices(0)
  , rkmatrix(nullptr)
  , fullmatrix(nullptr)
  , bc_node(nullptr)
  , row_indices(nullptr)
  , col_indices(nullptr)
  , m(0)
  , n(0)
{}


template <int spacedim, typename Number>
HMatrix<spacedim, Number>::HMatrix(
  const BlockClusterTree<spacedim, Number> &bct)
  : type(UndefinedMatrixType)
  , submatrices(0)
  , rkmatrix(nullptr)
  , fullmatrix(nullptr)
  , bc_node(nullptr)
  , row_indices(nullptr)
  , col_indices(nullptr)
  , m(0)
  , n(0)
{
  InitAndCreateHMatrixChildren(this, bct.get_root());
}


template <int spacedim, typename Number>
HMatrix<spacedim, Number>::HMatrix(
  typename BlockClusterTree<spacedim, Number>::node_const_pointer_type bc_node)
  : type(UndefinedMatrixType)
  , submatrices(0)
  , rkmatrix(nullptr)
  , fullmatrix(nullptr)
  , bc_node(nullptr)
  , row_indices(nullptr)
  , col_indices(nullptr)
  , m(0)
  , n(0)
{
  InitAndCreateHMatrixChildren(this, bc_node);
}


template <int spacedim, typename Number>
HMatrix<spacedim, Number>::HMatrix(
  const BlockClusterTree<spacedim, Number> &bct,
  const FullMatrix<Number> &                M)
  : type(UndefinedMatrixType)
  , submatrices(0)
  , rkmatrix(nullptr)
  , fullmatrix(nullptr)
  , bc_node(nullptr)
  , row_indices(nullptr)
  , col_indices(nullptr)
  , m(0)
  , n(0)
{}


template <int spacedim, typename Number>
HMatrix<spacedim, Number>::HMatrix(
  typename BlockClusterTree<spacedim, Number>::node_const_pointer_type bc_node,
  const FullMatrix<Number> &                                           M)
  : type(UndefinedMatrixType)
  , submatrices(0)
  , rkmatrix(nullptr)
  , fullmatrix(nullptr)
  , bc_node(nullptr)
  , row_indices(nullptr)
  , col_indices(nullptr)
  , m(0)
  , n(0)
{}


template <int spacedim, typename Number>
HMatrix<spacedim, Number>::~HMatrix()
{
  if (rkmatrix != nullptr)
    {
      delete rkmatrix;
    }

  if (fullmatrix != nullptr)
    {
      delete fullmatrix;
    }

  for (auto submatrix : submatrices)
    {
      /**
       * The deletion of \p submatrix will call the destructor of this
       * sub-HMatrix, which will further recursively call the destructor of the
       * submatrices of this sub-HMatrix. Hence, this destructor is
       * intrinsically recursive.
       */
      delete submatrix;
    }
}

#endif /* INCLUDE_HMATRIX_H_ */
