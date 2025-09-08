// Copyright (C) 2021-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

#include <deal.II/base/logstream.h>

#include "laplace/laplace_bem.h"
#include "utilities/data_output.h"

int
main()
{
  deallog.pop();
  deallog.depth_console(2);

  FE_Q<2, 3> fe(3);

  std::vector<unsigned int> forward_dof_numbering_from_0(
    HierBEM::generate_forward_dof_permutation(fe, 0));
  deallog << "Forward dof numbering starting from corner #0..." << std::endl;
  HierBEM::print_vector(deallog.get_console(),
                        forward_dof_numbering_from_0,
                        std::string(", "));

  std::vector<unsigned int> forward_dof_numbering_from_1(
    HierBEM::generate_forward_dof_permutation(fe, 1));
  deallog << "Forward dof numbering starting from corner #1..." << std::endl;
  HierBEM::print_vector(deallog.get_console(),
                        forward_dof_numbering_from_1,
                        std::string(", "));

  std::vector<unsigned int> forward_dof_numbering_from_2(
    HierBEM::generate_forward_dof_permutation(fe, 2));
  deallog << "Forward dof numbering starting from corner #2..." << std::endl;
  HierBEM::print_vector(deallog.get_console(),
                        forward_dof_numbering_from_2,
                        std::string(", "));

  std::vector<unsigned int> forward_dof_numbering_from_3(
    HierBEM::generate_forward_dof_permutation(fe, 3));
  deallog << "Forward dof numbering starting from corner #3..." << std::endl;
  HierBEM::print_vector(deallog.get_console(),
                        forward_dof_numbering_from_3,
                        std::string(", "));

  std::vector<unsigned int> backward_dof_numbering_from_0(
    HierBEM::generate_backward_dof_permutation(fe, 0));
  deallog << "Backward dof numbering starting from corner #0..." << std::endl;
  HierBEM::print_vector(deallog.get_console(),
                        backward_dof_numbering_from_0,
                        std::string(", "));

  std::vector<unsigned int> backward_dof_numbering_from_1(
    HierBEM::generate_backward_dof_permutation(fe, 1));
  deallog << "Backward dof numbering starting from corner #1..." << std::endl;
  HierBEM::print_vector(deallog.get_console(),
                        backward_dof_numbering_from_1,
                        std::string(", "));

  std::vector<unsigned int> backward_dof_numbering_from_2(
    HierBEM::generate_backward_dof_permutation(fe, 2));
  deallog << "Backward dof numbering starting from corner #2..." << std::endl;
  HierBEM::print_vector(deallog.get_console(),
                        backward_dof_numbering_from_2,
                        std::string(", "));

  std::vector<unsigned int> backward_dof_numbering_from_3(
    HierBEM::generate_backward_dof_permutation(fe, 3));
  deallog << "Backward dof numbering starting from corner #3..." << std::endl;
  HierBEM::print_vector(deallog.get_console(),
                        backward_dof_numbering_from_3,
                        std::string(", "));

  return 0;
}
