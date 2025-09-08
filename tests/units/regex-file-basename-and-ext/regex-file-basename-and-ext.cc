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
 * \file regex-file-basename-and-ext.cc
 * \brief Verify parsing file name into base name and extension using regular
 * expressions.
 *
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2022-03-16
 */

#include <iostream>

#include "utilities/regex_tools.h"

using namespace std;
using namespace RegexTools;

int
main()
{
  {
    //! File name with an extension
    string filename("mesh_file.msh");
    cout << "Base: " << file_basename(filename)
         << "\nExtension: " << file_ext(filename) << endl;
  }

  {
    //! File name with an extension
    string filename("./folder/mesh_file.msh");
    cout << "Base: " << file_basename(filename)
         << "\nExtension: " << file_ext(filename) << endl;
  }

  {
    //! File name with an extension
    string filename("../folder/mesh_file.msh");
    cout << "Base: " << file_basename(filename)
         << "\nExtension: " << file_ext(filename) << endl;
  }

  {
    //! File name with an extension
    string filename("../../folder/mesh_file.msh");
    cout << "Base: " << file_basename(filename)
         << "\nExtension: " << file_ext(filename) << endl;
  }

  {
    //! File name without an extension
    string filename("mesh_file");
    cout << "Base: " << file_basename(filename)
         << "\nExtension: " << file_ext(filename) << endl;
  }

  {
    //! File name without an extension
    string filename("./folder/mesh_file");
    cout << "Base: " << file_basename(filename)
         << "\nExtension: " << file_ext(filename) << endl;
  }

  {
    //! File name without an extension
    string filename("../folder/mesh_file");
    cout << "Base: " << file_basename(filename)
         << "\nExtension: " << file_ext(filename) << endl;
  }

  {
    //! File name without an extension
    string filename("../../folder/mesh_file");
    cout << "Base: " << file_basename(filename)
         << "\nExtension: " << file_ext(filename) << endl;
  }

  return 0;
}
