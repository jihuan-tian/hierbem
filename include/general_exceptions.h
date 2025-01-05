/**
 * \file general_exceptions.h
 * \brief Definition of self-defined general exceptions.
 * \date 2021-06-10
 * \author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_GENERAL_EXCEPTIONS_H_
#define HIERBEM_INCLUDE_GENERAL_EXCEPTIONS_H_


#include <deal.II/base/exceptions.h>

#include "config.h"

HBEM_NS_OPEN

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

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_GENERAL_EXCEPTIONS_H_
