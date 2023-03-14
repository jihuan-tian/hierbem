/**
 * \file general_exceptions.h
 * \brief Definition of self-defined general exceptions.
 * \date 2021-06-10
 * \author Jihuan Tian
 */
#ifndef INCLUDE_GENERAL_EXCEPTIONS_H_
#define INCLUDE_GENERAL_EXCEPTIONS_H_

#include <deal.II/base/exceptions.h>

namespace IdeoBEM
{
  using namespace dealii;

  DeclException3(ExcOpenIntervalRange,
                 double,
                 double,
                 double,
                 << arg1 << " is not in the range (" << arg2 << ", " << arg3
                 << ").");

  DeclException3(ExcLeftOpenIntervalRange,
                 double,
                 double,
                 double,
                 << arg1 << " is not in the range (" << arg2 << ", " << arg3
                 << "].");

  DeclException3(ExcRightOpenIntervalRange,
                 double,
                 double,
                 double,
                 << arg1 << " is not in the range [" << arg2 << ", " << arg3
                 << ").");

  DeclException3(ExcClosedIntervalRange,
                 double,
                 double,
                 double,
                 << arg1 << " is not in the range [" << arg2 << ", " << arg3
                 << "].");
} // namespace IdeoBEM

#endif /* INCLUDE_GENERAL_EXCEPTIONS_H_ */
