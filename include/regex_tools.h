/**
 * @file regex_tools.h
 * @brief Declaration of tools for regular expressions.
 *
 * @date 2022-03-16
 * @author Jihuan Tian
 */
#ifndef INCLUDE_REGEX_TOOLS_H_
#define INCLUDE_REGEX_TOOLS_H_


#include <regex>

namespace RegexTools
{
  using namespace std;

  extern const regex reg_for_file_base_and_ext1;
  extern const regex reg_for_file_base_and_ext2;

  string
  file_basename(const string &filename);

  string
  file_ext(const string &filename);
} // namespace RegexTools


#endif /* INCLUDE_REGEX_TOOLS_H_ */
