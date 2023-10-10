#include <deal.II/base/logstream.h>
#include <laplace_bem.h>

#include "data_output.h"

int main()
{
  deallog.pop();
  deallog.depth_console(2);

  FE_Q<2, 3> fe(3);

  std::vector<unsigned int>
  forward_dof_numbering_from_0(HierBEM::generate_forward_dof_permutation(fe, 0));
  deallog << "Forward dof numbering starting from corner #0..." << std::endl;
  HierBEM::print_vector(deallog.get_console(), forward_dof_numbering_from_0, std::string(", "));

  std::vector<unsigned int>
  forward_dof_numbering_from_1(HierBEM::generate_forward_dof_permutation(fe, 1));
  deallog << "Forward dof numbering starting from corner #1..." << std::endl;
  HierBEM::print_vector(deallog.get_console(), forward_dof_numbering_from_1, std::string(", "));

  std::vector<unsigned int>
  forward_dof_numbering_from_2(HierBEM::generate_forward_dof_permutation(fe, 2));
  deallog << "Forward dof numbering starting from corner #2..." << std::endl;
  HierBEM::print_vector(deallog.get_console(), forward_dof_numbering_from_2, std::string(", "));

  std::vector<unsigned int>
  forward_dof_numbering_from_3(HierBEM::generate_forward_dof_permutation(fe, 3));
  deallog << "Forward dof numbering starting from corner #3..." << std::endl;
  HierBEM::print_vector(deallog.get_console(), forward_dof_numbering_from_3, std::string(", "));

  std::vector<unsigned int>
  backward_dof_numbering_from_0(HierBEM::generate_backward_dof_permutation(fe, 0));
  deallog << "Backward dof numbering starting from corner #0..." << std::endl;
  HierBEM::print_vector(deallog.get_console(), backward_dof_numbering_from_0, std::string(", "));

  std::vector<unsigned int>
  backward_dof_numbering_from_1(HierBEM::generate_backward_dof_permutation(fe, 1));
  deallog << "Backward dof numbering starting from corner #1..." << std::endl;
  HierBEM::print_vector(deallog.get_console(), backward_dof_numbering_from_1, std::string(", "));

  std::vector<unsigned int>
  backward_dof_numbering_from_2(HierBEM::generate_backward_dof_permutation(fe, 2));
  deallog << "Backward dof numbering starting from corner #2..." << std::endl;
  HierBEM::print_vector(deallog.get_console(), backward_dof_numbering_from_2, std::string(", "));

  std::vector<unsigned int>
  backward_dof_numbering_from_3(HierBEM::generate_backward_dof_permutation(fe, 3));
  deallog << "Backward dof numbering starting from corner #3..." << std::endl;
  HierBEM::print_vector(deallog.get_console(), backward_dof_numbering_from_3, std::string(", "));

  return 0;
}
