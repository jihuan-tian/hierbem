// Copyright (C) 2022-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file hmatrix_support.cc
 * @brief Implementation of helper functions declared in @p hmatrix_support.h
 *
 * @date 2022-05-09
 * @author Jihuan Tian
 */

#include "hmatrix/hmatrix_support.h"

namespace HierBEM
{
  namespace HMatrixSupport
  {
    void
    infer_submatrix_block_types_from_parent_hmat(
      const BlockType                           parent_hmat_block_type,
      std::array<HMatrixSupport::BlockType, 4> &submatrix_block_types)
    {
      switch (parent_hmat_block_type)
        {
            case HMatrixSupport::diagonal_block: {
              submatrix_block_types[0] = HMatrixSupport::diagonal_block;
              submatrix_block_types[1] = HMatrixSupport::upper_triangular_block;
              submatrix_block_types[2] = HMatrixSupport::lower_triangular_block;
              submatrix_block_types[3] = HMatrixSupport::diagonal_block;

              break;
            }
            case HMatrixSupport::lower_triangular_block: {
              submatrix_block_types[0] = HMatrixSupport::lower_triangular_block;
              submatrix_block_types[1] = HMatrixSupport::lower_triangular_block;
              submatrix_block_types[2] = HMatrixSupport::lower_triangular_block;
              submatrix_block_types[3] = HMatrixSupport::lower_triangular_block;

              break;
            }
            case HMatrixSupport::upper_triangular_block: {
              submatrix_block_types[0] = HMatrixSupport::upper_triangular_block;
              submatrix_block_types[1] = HMatrixSupport::upper_triangular_block;
              submatrix_block_types[2] = HMatrixSupport::upper_triangular_block;
              submatrix_block_types[3] = HMatrixSupport::upper_triangular_block;

              break;
            }
            case HMatrixSupport::undefined_block: {
              submatrix_block_types[0] = HMatrixSupport::undefined_block;
              submatrix_block_types[1] = HMatrixSupport::undefined_block;
              submatrix_block_types[2] = HMatrixSupport::undefined_block;
              submatrix_block_types[3] = HMatrixSupport::undefined_block;

              break;
            }
            default: {
              Assert(false,
                     ExcMessage(std::string("Invalid H-matrix types: ") +
                                std::to_string(parent_hmat_block_type)));

              break;
            }
        }
    }
  } // namespace HMatrixSupport
} // namespace HierBEM
