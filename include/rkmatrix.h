/**
 * \file rkmatrix.h
 * \brief Definition of rank-k matrix.
 * \ingroup hierarchical_matrices
 * \date 2021-06-06
 * \author Jihuan Tian
 */

#ifndef INCLUDE_RKMATRIX_H_
#define INCLUDE_RKMATRIX_H_

#include <deal.II/base/types.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/lapack_support.h>
#include <deal.II/lac/lapack_templates.h>

#include "lapack_full_matrix_ext.h"

using namespace dealii;

template <typename Number = double>
class RkMatrix
{
public:
  /**
   * Declare the type for container size.
   */
  using size_type = std::make_unsigned<types::blas_int>::type;

  template <typename Number1>
  friend void
  print_rkmatrix_to_mat(std::ostream &           out,
                        const std::string &      name,
                        const RkMatrix<Number1> &values,
                        const unsigned int       precision,
                        const bool               scientific,
                        const unsigned int       width,
                        const char *             zero_string,
                        const double             denominator,
                        const double             threshold);

  /**
   * Default constructor.
   */
  RkMatrix();

  /**
   * Construct an empty rank-k matrix with the specified matrix dimension and
   * rank.
   * @param m
   * @param n
   * @param fixed_rank_k
   */
  RkMatrix(const size_type m, const size_type n, const size_type fixed_rank_k);

  /**
   * Construct an rank-k matrix from a specified block cluster in a full
   * matrix \p M.
   * @param tau
   * @param sigma
   * @param fixed_rank_k
   * @param M
   */
  template <typename SrcMatrixType>
  RkMatrix(const std::vector<types::global_dof_index> &tau,
           const std::vector<types::global_dof_index> &sigma,
           const size_type                             fixed_rank_k,
           const SrcMatrixType &                       M);

  /**
   * Construct an rank-k matrix from a full matrix \p M.
   * @param fixed_rank_k
   * @param M
   */
  template <typename SrcMatrixType>
  RkMatrix(const size_type fixed_rank_k, SrcMatrixType &M);

  /**
   * Construct an rank-k matrix from two component matrices.
   * @param A
   * @param B
   */
  RkMatrix(const LAPACKFullMatrixExt<Number> &A,
           const LAPACKFullMatrixExt<Number> &B);

  /**
   * Copy constructor.
   */
  RkMatrix(const RkMatrix<Number> &matrix);

  /**
   * Convert an HMatrix to a full matrix.
   * @param matrix
   */
  template <typename MatrixType>
  void
  convertToFullMatrix(MatrixType &matrix) const;

  /**
   * Print a RkMatrix.
   * @param out
   * @param precision
   * @param scientific
   * @param width
   * @param zero_string
   * @param denominator
   * @param threshold
   */
  void
  print_formatted(std::ostream &     out,
                  const unsigned int precision   = 3,
                  const bool         scientific  = true,
                  const unsigned int width       = 0,
                  const char *       zero_string = "0",
                  const double       denominator = 1.,
                  const double       threshold   = 0.) const;

  /**
   * Truncate the RkMatrix to \p new_rank.
   * @param new_rank
   */
  void
  truncate_to_rank(size_type new_rank);

  /**
   * Calculate matrix-vector multiplication as \f$y = A B^T x\f$ or \f$y =
   * y + A B^T x\f$.
   * @param y
   * @param x
   * @param adding
   */
  void
  vmult(Vector<Number> &      y,
        const Vector<Number> &x,
        const bool            adding = false) const;

  /**
   * Calculate matrix-vector multiplication as \f$y = B A^T x\f$ or \f$y =
   * y + B A^T x\f$.
   * @param y
   * @param x
   * @param adding
   */
  void
  Tvmult(Vector<Number> &      y,
         const Vector<Number> &x,
         const bool            adding = false) const;

private:
  LAPACKFullMatrixExt<Number> A;
  LAPACKFullMatrixExt<Number> B;

  /**
   * Matrix rank.
   */
  size_type rank;

  /**
   * Total number of rows.
   */
  size_type m;

  /**
   * Total number of columns
   */
  size_type n;
};


template <typename Number>
RkMatrix<Number>::RkMatrix()
  : A(0, 0)
  , B(0, 0)
  , rank(0)
  , m(0)
  , n(0)
{}


template <typename Number>
RkMatrix<Number>::RkMatrix(const size_type m,
                           const size_type n,
                           const size_type fixed_rank_k)
  : A()
  , B()
  , rank(0)
  , m(m)
  , n(n)
{
  const size_type min_dim        = std::min(m, n);
  const size_type effective_rank = std::min(min_dim, fixed_rank_k);

  A.reinit(m, effective_rank);
  B.reinit(n, effective_rank);
  rank = effective_rank;
}


template <typename Number>
template <typename SrcMatrixType>
RkMatrix<Number>::RkMatrix(const std::vector<types::global_dof_index> &tau,
                           const std::vector<types::global_dof_index> &sigma,
                           const size_type      fixed_rank_k,
                           const SrcMatrixType &M)
  : A()
  , B()
  , rank()
  , m(tau.size())
  , n(sigma.size())
{
  /**
   * Extract the data for the submatrix block \f$b = \tau \times \sigma\f$ in
   * the original matrix \p M.
   */
  LAPACKFullMatrixExt<Number> M_b(m, n);

  for (size_type i = 0; i < m; i++)
    {
      for (size_type j = 0; j < n; j++)
        {
          M_b(i, j) = M(tau.at(i), sigma.at(j));
        }
    }

  /**
   * Convert the matrix block \p M_b in full matrix format to rank-k
   * format.
   */
  rank = M_b.rank_k_decompose(fixed_rank_k, A, B, true);
}


template <typename Number>
template <typename SrcMatrixType>
RkMatrix<Number>::RkMatrix(const size_type fixed_rank_k, SrcMatrixType &M)
  : A()
  , B()
  , rank()
  , m(M.m())
  , n(M.n())
{
  rank = M.rank_k_decompose(fixed_rank_k, A, B, true);
}


template <typename Number>
RkMatrix<Number>::RkMatrix(const LAPACKFullMatrixExt<Number> &A,
                           const LAPACKFullMatrixExt<Number> &B)
  : A(A)
  , B(B)
  , rank(A.n())
  , m(A.m())
  , n(B.m())
{
  AssertDimension(A.n(), B.n());
}


template <typename Number>
RkMatrix<Number>::RkMatrix(const RkMatrix<Number> &matrix)
  : A(matrix.A)
  , B(matrix.B)
  , rank(matrix.rank)
  , m(matrix.m)
  , n(matrix.n)
{}


template <typename Number>
template <typename MatrixType>
void
RkMatrix<Number>::convertToFullMatrix(MatrixType &matrix) const
{
  matrix.reinit(m, n);
  A.mTmult(matrix, B);
}


template <typename Number>
void
RkMatrix<Number>::print_formatted(std::ostream &     out,
                                  const unsigned int precision,
                                  const bool         scientific,
                                  const unsigned int width,
                                  const char *       zero_string,
                                  const double       denominator,
                                  const double       threshold) const
{
  out << "RkMatrix.dim=(" << m << "," << n << ")\n";
  out << "RkMatrix.rank=" << rank << "\n";
  out << "RkMatrix.A=\n";
  A.print_formatted(
    out, precision, scientific, width, zero_string, denominator, threshold);
  out << "RkMatrix.B=\n";
  B.print_formatted(
    out, precision, scientific, width, zero_string, denominator, threshold);
}


template <typename Number>
void
RkMatrix<Number>::truncate_to_rank(size_type new_rank)
{
  Assert(new_rank > 0, ExcLowerRange(new_rank, 1));

  if (new_rank >= rank)
    {
      /**
       * Do nothing.
       */
    }
  else
    {
      A.keep_first_n_columns(new_rank, true);
      B.keep_first_n_columns(new_rank, true);
      rank = new_rank;
    }
}


template <typename Number>
void
RkMatrix<Number>::vmult(Vector<Number> &      y,
                        const Vector<Number> &x,
                        const bool            adding) const
{
  /**
   * The vector storing \f$B^T x\f$
   */
  Vector<Number> z(rank);

  B.Tvmult(z, x);
  A.vmult(y, z, adding);
}


template <typename Number>
void
RkMatrix<Number>::Tvmult(Vector<Number> &      y,
                         const Vector<Number> &x,
                         const bool            adding) const
{
  /**
   * The vector storing \f$B^T x\f$
   */
  Vector<Number> z(rank);

  A.Tvmult(z, x);
  B.vmult(y, z, adding);
}


/**
 * Print an RkMatrix in Octave text data format.
 * @param out
 * @param name
 * @param values
 * @param precision
 * @param scientific
 * @param width
 * @param zero_string
 * @param denominator
 * @param threshold
 */
template <typename Number>
void
print_rkmatrix_to_mat(std::ostream &          out,
                      const std::string &     name,
                      const RkMatrix<Number> &values,
                      const unsigned int      precision   = 8,
                      const bool              scientific  = true,
                      const unsigned int      width       = 0,
                      const char *            zero_string = "0",
                      const double            denominator = 1.,
                      const double            threshold   = 0.)
{
  out << "# name: " << name << "\n"
      << "# type: scalar struct\n"
      << "# ndims: 2\n"
      << "1 1\n"
      << "# length: 2\n";

  out << "# name: A\n"
      << "# type: matrix\n"
      << "# rows: " << values.A.m() << "\n"
      << "# columns: " << values.A.n() << "\n";

  values.A.print_formatted(
    out, precision, scientific, width, zero_string, denominator, threshold);

  out << "\n\n";

  out << "# name: B\n"
      << "# type: matrix\n"
      << "# rows: " << values.B.m() << "\n"
      << "# columns: " << values.B.n() << "\n";

  values.B.print_formatted(
    out, precision, scientific, width, zero_string, denominator, threshold);

  out << "\n\n";
}

#endif /* INCLUDE_RKMATRIX_H_ */
