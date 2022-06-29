/**
 * @file verify-fe-poly-dof-numbering.cc
 * @brief Verify different orderings of dof support points related to finite
 * element polynomials.
 *
 * @date 2022-02-14
 * @author Jihuan Tian
 */

// Log handling
#include <deal.II/base/logstream.h>
// H1-conforming finite element
#include <deal.II/fe/fe_q.h>
// L2-conforming finite element
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_tools.h>

// Linear algebra related
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <fstream>

#include "debug_tools.h"

using namespace dealii;

int
main()
{
  deallog.pop();
  deallog.depth_console(2);

  const unsigned int    spacedim = 3;
  const unsigned int    dim      = 2;
  const unsigned int    fe_order = 3;
  FE_Q<dim, spacedim>   fe_q(fe_order);
  FE_DGQ<dim, spacedim> fe_dgq(fe_order);

  print_polynomial_space_numbering(std::cout, fe_q, "fe_q");
  print_mapping_between_lexicographic_and_hierarchic_numberings(std::cout,
                                                                fe_q,
                                                                "fe_q");

  print_polynomial_space_numbering(std::cout, fe_dgq, "fe_dgq");
  //  print_mapping_between_lexicographic_and_hierarchic_numberings(std::cout,
  //                                                                fe_dgq,
  //                                                                "fe_dgq");
}
