/**
 * \file regex-file-basename-and-ext.cc
 * \brief Verify parsing file name into base name and extension using regular
 * expressions.
 *
 * \ingroup testers
 * \author Jihuan Tian
 * \date 2022-03-16
 */

#include <iostream>

#include "regex_tools.h"

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
