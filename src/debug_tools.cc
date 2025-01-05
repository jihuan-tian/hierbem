/**
 * @file debug_tools.cc
 * @brief Introduction of debug_tools.cc
 *
 * @date 2023-03-09
 * @author Jihuan Tian
 */

#include "debug_tools.h"

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
} // namespace HierBEM
