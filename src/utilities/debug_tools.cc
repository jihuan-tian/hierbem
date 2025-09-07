/**
 * @file debug_tools.cc
 * @brief Introduction of debug_tools.cc
 *
 * @date 2023-03-09
 * @author Jihuan Tian
 */

#include "utilities/debug_tools.h"

#include <fstream>
#include <string>

namespace HierBEM
{
  void
  print_wall_time(LogStream         &log,
                  const Timer       &timer,
                  const std::string &activity_name)
  {
    log << "Elapsed wall time for " << activity_name << " is "
        << timer.last_wall_time() << "s, total elapsed wall time is "
        << timer.wall_time() << "s" << std::endl;
  }


  void
  print_wall_time(std::ostream      &out,
                  const Timer       &timer,
                  const std::string &activity_name)
  {
    out << "Elapsed wall time for " << activity_name << " is "
        << timer.last_wall_time() << "s, total elapsed wall time is "
        << timer.wall_time() << "s" << std::endl;
  }


  void
  print_cpu_time(LogStream         &log,
                 const Timer       &timer,
                 const std::string &activity_name)
  {
    log << "Elapsed cpu time for " << activity_name << " is "
        << timer.last_cpu_time() << "s, total elapsed cpu time is "
        << timer.cpu_time() << "s" << std::endl;
  }


  void
  print_cpu_time(std::ostream      &out,
                 const Timer       &timer,
                 const std::string &activity_name)
  {
    out << "Elapsed cpu time for " << activity_name << " is "
        << timer.last_cpu_time() << "s, total elapsed cpu time is "
        << timer.cpu_time() << "s" << std::endl;
  }


  void
  read_file_lines(const std::string &file_name, std::vector<std::string> &lines)
  {
    std::ifstream in(file_name);

    std::string line;
    while (std::getline(in, line))
      lines.push_back(line);

    in.close();
  }
} // namespace HierBEM
