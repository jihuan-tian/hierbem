/**
 * @file regex_tools.cc
 * @brief Introduction of regex_tools.cc
 *
 * @date 2022-03-16
 * @author Jihuan Tian
 */

#include "regex_tools.h"

#include <exception>

namespace HierBEM
{
  namespace RegexTools
  {
    using namespace std;

    const regex reg_for_file_base_and_ext1("^((../|./)*)(.+)$");
    const regex reg_for_file_base_and_ext2("^(.+)\\.(.+)$");

    string
    file_basename(const string &filename)
    {
      smatch m1, m2;

      regex_match(filename, m1, reg_for_file_base_and_ext1);
      string filename_without_dots(m1[3].str());

      if (regex_match(filename_without_dots, m2, reg_for_file_base_and_ext2))
        {
          return m1[1].str() + m2[1].str();
        }
      else
        {
          return m1[1].str() + m1[3].str();
        }
    }

    string
    file_ext(const string &filename)
    {
      smatch m1, m2;

      regex_match(filename, m1, reg_for_file_base_and_ext1);
      string filename_without_dots(m1[3].str());

      if (regex_match(filename_without_dots, m2, reg_for_file_base_and_ext2))
        {
          return m2[2].str();
        }
      else
        {
          return string("");
        }
    }
  } // namespace RegexTools
} // namespace HierBEM
