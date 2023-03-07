/**
 * \file read_octave_data.h
 * \brief Introduction of read_octave_data.h
 * \date 2021-10-20
 * \author Jihuan Tian
 */
#ifndef INCLUDE_READ_OCTAVE_DATA_H_
#define INCLUDE_READ_OCTAVE_DATA_H_


#include <deal.II/lac/vector.h>

#include <fstream>
#include <regex>
#include <string>

#include "lapack_full_matrix_ext.h"

namespace IdeoBEM
{
  template <typename MatrixType>
  void
  read_matrix_from_octave(std::ifstream     &in,
                          const std::string &name,
                          MatrixType        &matrix)
  {
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
            Assert(line_buf.compare("# type: matrix") == 0,
                   ExcMessage(
                     "Data type for the matrix to be read should be 'matrix'"));

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
            if (!std::regex_match(line_buf,
                                  sm,
                                  std::regex("# columns: (\\d+)")))
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
            Assert(line_buf.compare("# type: matrix") == 0,
                   ExcMessage(
                     "Data type for the matrix to be read should be 'matrix'"));

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
            if (!std::regex_match(line_buf,
                                  sm,
                                  std::regex("# columns: (\\d+)")))
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
} // namespace IdeoBEM

#endif /* INCLUDE_READ_OCTAVE_DATA_H_ */
