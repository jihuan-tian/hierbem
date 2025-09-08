// Copyright (C) 2021-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * \file octave-read-vector.cc
 * \brief Verify reading a vector from a file saved from Octave in text format,
 * i.e. saved with the option \p -text.
 *
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2021-10-20
 */

#include <fstream>

#include "utilities/read_octave_data.h"

using namespace dealii;

int
main()
{
  Vector<double> V1, V2;

  std::ifstream in("input.dat");

  read_vector_from_octave(in, "V1", V1);
  read_vector_from_octave(in, "V2", V2);

  std::cout << "V1=\n";
  V1.print(std::cout, 8, false, true);
  std::cout << "V2=\n";
  V2.print(std::cout, 8, false, true);

  return 0;
}
