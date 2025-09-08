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
 * @file read_octave_data.h
 * @brief Introduction of read_octave_data.h
 * @date 2021-10-20
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_UTILITIES_READ_OCTAVE_DATA_H_
#define HIERBEM_INCLUDE_UTILITIES_READ_OCTAVE_DATA_H_

#include <deal.II/base/numbers.h>

#include <deal.II/lac/vector.h>

#include <fstream>
#include <regex>
#include <string>

#include "config.h"
#include "linear_algebra/lapack_full_matrix_ext.h"

HBEM_NS_OPEN

template <typename MatrixType>
void
read_matrix_from_octave(std::ifstream     &in,
                        const std::string &name,
                        MatrixType        &matrix)
{
  in.seekg(0);

  std::string line_buf;

  /**
   * Iterate over each line of the file to search the desired variable.
   */
  while (std::getline(in, line_buf))
    {
      if (line_buf.compare(std::string("# name: ") + name) == 0)
        {
          /**
           * When the desired variable is found, read the next line to check
           * the
           * data type is \p matrix.
           */
          std::getline(in, line_buf);
          if constexpr (numbers::NumberTraits<
                          typename MatrixType::value_type>::is_complex)
            {
              Assert(
                line_buf.compare("# type: complex matrix") == 0,
                ExcMessage(
                  "Data type for the complex valued matrix to be read should be 'complex matrix'"));
            }
          else
            {
              Assert(
                line_buf.compare("# type: matrix") == 0,
                ExcMessage(
                  "Data type for the real valued matrix to be read should be 'matrix'"));
            }

          /**
           * Read a new line to extract the number of rows.
           */
          std::getline(in, line_buf);
          std::smatch sm;
          if (!std::regex_match(line_buf, sm, std::regex("# rows: (\\d+)")))
            {
              ExcMessage("Cannot get n_rows of the matrix!");
            }
          const unsigned int n_rows = std::stoi(sm.str(1));

          /**
           * Read a new line to extract the number of columns.
           */
          std::getline(in, line_buf);
          if (!std::regex_match(line_buf, sm, std::regex("# columns: (\\d+)")))
            {
              ExcMessage("Cannot get n_cols of the matrix!");
            }
          const unsigned int n_cols = std::stoi(sm.str(1));

          Assert(n_rows > 0, ExcMessage("Matrix to be read has no rows!"));
          Assert(n_cols > 0, ExcMessage("Matrix to be read has no columns!"));

          matrix.reinit(n_rows, n_cols);
          /**
           * Get each row of the matrix.
           */
          for (typename MatrixType::size_type i = 0; i < n_rows; i++)
            {
              std::getline(in, line_buf);
              std::istringstream line_buf_stream(line_buf);
              /**
               * Get each matrix element in a row.
               */
              for (typename MatrixType::size_type j = 0; j < n_cols; j++)
                {
                  line_buf_stream >> matrix(i, j);
                }
            }

          /**
           * After reading all matrix data, exit from the loop.
           */
          break;
        }
    }
}


template <typename Number>
void
read_vector_from_octave(std::ifstream          &in,
                        const std::string      &name,
                        dealii::Vector<Number> &vec)
{
  in.seekg(0);

  std::string line_buf;

  /**
   * Iterate over each line of the file to search the desired variable.
   */
  while (std::getline(in, line_buf))
    {
      if (line_buf.compare(std::string("# name: ") + name) == 0)
        {
          /**
           * When the desired variable is found, read the next line to check
           * the
           * data type is \p matrix.
           */
          std::getline(in, line_buf);
          if constexpr (numbers::NumberTraits<Number>::is_complex)
            {
              Assert(
                line_buf.compare("# type: complex matrix") == 0,
                ExcMessage(
                  "Data type for the complex valued vector to be read should be 'complex matrix'"));
            }
          else
            {
              Assert(
                line_buf.compare("# type: matrix") == 0,
                ExcMessage(
                  "Data type for the real valued vector to be read should be 'matrix'"));
            }

          /**
           * Read a new line to extract the number of rows.
           */
          std::getline(in, line_buf);
          std::smatch sm;
          if (!std::regex_match(line_buf, sm, std::regex("# rows: (\\d+)")))
            {
              ExcMessage("Cannot get n_rows of the matrix!");
            }
          const unsigned int n_rows = std::stoi(sm.str(1));

          /**
           * Read a new line to extract the number of columns.
           */
          std::getline(in, line_buf);
          if (!std::regex_match(line_buf, sm, std::regex("# columns: (\\d+)")))
            {
              ExcMessage("Cannot get n_cols of the matrix!");
            }
          const unsigned int n_cols = std::stoi(sm.str(1));

          Assert(n_rows > 0, ExcMessage("Matrix to be read has no rows!"));
          Assert(n_cols > 0, ExcMessage("Matrix to be read has no columns!"));
          Assert(
            n_rows == 1 || n_cols == 1,
            ExcMessage(
              "Either n_rows or n_cols should be 1 for initializing a vector!"));

          const unsigned int vector_length = std::max(n_rows, n_cols);
          vec.reinit(vector_length);

          if (n_rows == 1)
            {
              /**
               * The vector is a row vector.
               */
              std::getline(in, line_buf);
              std::istringstream line_buf_stream(line_buf);
              /**
               * Get each matrix element in a row.
               */
              for (typename dealii::Vector<Number>::size_type i = 0;
                   i < vector_length;
                   i++)
                {
                  line_buf_stream >> vec(i);
                }
            }
          else
            {
              /**
               * The vector is a column vector. Then get each row of the
               * vector.
               */
              for (typename dealii::Vector<Number>::size_type i = 0;
                   i < vector_length;
                   i++)
                {
                  std::getline(in, line_buf);
                  std::istringstream line_buf_stream(line_buf);
                  line_buf_stream >> vec(i);
                }
            }


          /**
           * After reading all matrix data, exit from the loop.
           */
          break;
        }
    }
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_UTILITIES_READ_OCTAVE_DATA_H_
