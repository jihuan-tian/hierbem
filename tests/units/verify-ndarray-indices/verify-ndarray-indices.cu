// Copyright (C) 2023-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file verify-CUDAWrappers::ndarray-indices.cc
 * @brief Verify the conversion from multi-dimensional array indices into a
 * linearized index.
 *
 * @ingroup test_cases
 * @author Jihuan Tian
 * @date 2023-01-29
 */

#include <iostream>

#include "linear_algebra/cu_table_indices.hcu"

using namespace dealii;
using namespace std;
using namespace HierBEM::CUDAWrappers;

using size_type = std::size_t;

int
main()
{
  {
    CUDATableIndices<1> table_size(6);
    CUDATableIndices<1> indices(5);
    CUDATableIndices<1> indices_from_linear_index;

    cout << "Table size (6), indices (5), C style linear index="
         << ndarray_indices_to_linear_index(indices, table_size) << endl;

    // Convert the linear index back to multi-dimensional indices using C style.
    linear_index_to_ndarray_indices(
      indices_from_linear_index,
      table_size,
      ndarray_indices_to_linear_index(indices, table_size));
    cout << "ND-indices calculated from the linear index (C style): "
         << indices_from_linear_index << endl;

    cout << "Table size (6), indices (5), Fortran style linear index="
         << ndarray_indices_to_linear_index(indices, table_size, false) << endl;

    // Convert the linear index back to multi-dimensional indices using Fortran
    // style.
    linear_index_to_ndarray_indices(indices_from_linear_index,
                                    table_size,
                                    ndarray_indices_to_linear_index(indices,
                                                                    table_size,
                                                                    false),
                                    false);
    cout << "ND-indices calculated from the linear index (Fortran style): "
         << indices_from_linear_index << endl;
  }

  {
    CUDATableIndices<2> table_size(5, 5);
    CUDATableIndices<2> indices(2, 3);
    CUDATableIndices<2> indices_from_linear_index;

    cout << "Table size (5, 5), indices (2, 3), C style linear index="
         << ndarray_indices_to_linear_index(indices, table_size) << endl;

    // Convert the linear index back to multi-dimensional indices using C style.
    linear_index_to_ndarray_indices(
      indices_from_linear_index,
      table_size,
      ndarray_indices_to_linear_index(indices, table_size));
    cout << "ND-indices calculated from the linear index (C style): "
         << indices_from_linear_index << endl;

    cout << "Table size (5, 5), indices (2, 3), Fortran style linear index="
         << ndarray_indices_to_linear_index(indices, table_size, false) << endl;

    // Convert the linear index back to multi-dimensional indices using Fortran
    // style.
    linear_index_to_ndarray_indices(indices_from_linear_index,
                                    table_size,
                                    ndarray_indices_to_linear_index(indices,
                                                                    table_size,
                                                                    false),
                                    false);
    cout << "ND-indices calculated from the linear index (Fortran style): "
         << indices_from_linear_index << endl;
  }

  {
    CUDATableIndices<3> table_size(5, 5, 5);
    CUDATableIndices<3> indices(4, 3, 3);
    CUDATableIndices<3> indices_from_linear_index;

    cout << "Table size (5, 5, 5), indices (4, 3, 3), C style linear index="
         << ndarray_indices_to_linear_index(indices, table_size) << endl;

    // Convert the linear index back to multi-dimensional indices using C style.
    linear_index_to_ndarray_indices(
      indices_from_linear_index,
      table_size,
      ndarray_indices_to_linear_index(indices, table_size));
    cout << "ND-indices calculated from the linear index (C style): "
         << indices_from_linear_index << endl;

    cout
      << "Table size (5, 5, 5), indices (4, 3, 3), Fortran style linear index="
      << ndarray_indices_to_linear_index(indices, table_size, false) << endl;

    // Convert the linear index back to multi-dimensional indices using Fortran
    // style.
    linear_index_to_ndarray_indices(indices_from_linear_index,
                                    table_size,
                                    ndarray_indices_to_linear_index(indices,
                                                                    table_size,
                                                                    false),
                                    false);
    cout << "ND-indices calculated from the linear index (Fortran style): "
         << indices_from_linear_index << endl;
  }

  {
    // Calculate special cases: linear index = 0.
    CUDATableIndices<3> table_size(5, 5, 5);
    CUDATableIndices<3> indices_from_linear_index;

    // Convert the linear index back to multi-dimensional indices using C style.
    linear_index_to_ndarray_indices(indices_from_linear_index, table_size, 0);
    cout << "ND-indices calculated from the linear index (C style): "
         << indices_from_linear_index << endl;

    // Convert the linear index back to multi-dimensional indices using Fortran
    // style.
    linear_index_to_ndarray_indices(indices_from_linear_index,
                                    table_size,
                                    0,
                                    false);
    cout << "ND-indices calculated from the linear index (Fortran style): "
         << indices_from_linear_index << endl;
  }

  {
    // Calculate special cases: linear index = 104.
    CUDATableIndices<3> table_size(5, 3, 7);
    CUDATableIndices<3> indices_from_linear_index;

    // Convert the linear index back to multi-dimensional indices using C style.
    linear_index_to_ndarray_indices(indices_from_linear_index, table_size, 104);
    cout << "ND-indices calculated from the linear index (C style): "
         << indices_from_linear_index << endl;

    // Convert the linear index back to multi-dimensional indices using Fortran
    // style.
    linear_index_to_ndarray_indices(indices_from_linear_index,
                                    table_size,
                                    104,
                                    false);
    cout << "ND-indices calculated from the linear index (Fortran style): "
         << indices_from_linear_index << endl;
  }
}
