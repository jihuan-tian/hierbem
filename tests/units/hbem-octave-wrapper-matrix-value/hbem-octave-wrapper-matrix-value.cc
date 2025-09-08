// Copyright (C) 2023 Xiaozhe Wang <chaoslawful@gmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file hbem-octave-wrapper-matrix-value.cc
 * @brief Generate an Octave matrix and load it into @p LAPACKFullMatrixExt.
 *
 * @ingroup test_cases octave
 * @author Jihuan Tian
 * @date 2023-11-10
 */

#include <iostream>

#include "hbem_octave_wrapper.h"
#include "linear_algebra/lapack_full_matrix_ext.h"

using namespace std;
using namespace HierBEM;

int
main()
{
  HBEMOctaveWrapper &inst    = HBEMOctaveWrapper::get_instance();
  auto               oct_val = inst.eval_string("reshape(1:15, 3, 5)");
  unsigned int       m, n;
  vector<double>     values;
  oct_val.matrix_value(values, m, n);

  LAPACKFullMatrixExt<double> mat;
  LAPACKFullMatrixExt<double>::Reshape(m, n, values, mat);
  mat.print_formatted_to_mat(cout, "A", 8, false, 16);

  return 0;
}
