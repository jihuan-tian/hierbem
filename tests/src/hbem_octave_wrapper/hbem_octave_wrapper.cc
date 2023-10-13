/**
 * \file hbem_octave_wrapper.cc
 * \brief Simple singleton wrapper around minimal Octave APIs to workaround
 * naming conflictions between Octave and CUDA and ease interacting
 * with Octave in testcases.
 *
 * \author Xiaozhe Wang
 * \date 2023-10-12
 */

#include "hbem_octave_wrapper.h"

#include <octave/builtin-defun-decls.h>
#include <octave/interpreter.h>
#include <octave/oct.h>
#include <octave/octave.h>
#include <octave/parse.h>

namespace HierBEM
{
  struct HBEMOctaveValueImpl
  {
    ::octave_value oct_val;
  };

  HBEMOctaveValue::HBEMOctaveValue()
    : m_impl(new HBEMOctaveValueImpl)
  {}

  HBEMOctaveValue::HBEMOctaveValue(::octave_value const &ov)
    : m_impl(new HBEMOctaveValueImpl)
  {
    m_impl->oct_val = ov;
  }
  HBEMOctaveValue::HBEMOctaveValue(HBEMOctaveValue const &other)
    : m_impl(new HBEMOctaveValueImpl)
  {
    m_impl->oct_val = other.m_impl->oct_val;
  }

  void
  HBEMOctaveValue::operator=(HBEMOctaveValue const &other)
  {
    m_impl->oct_val = other.m_impl->oct_val;
  }

  HBEMOctaveValue::~HBEMOctaveValue()
  {
    delete m_impl;
  }

  int
  HBEMOctaveValue::int_value() const
  {
    return m_impl->oct_val.int_value();
  }

  double
  HBEMOctaveValue::double_value() const
  {
    return m_impl->oct_val.double_value();
  }

  struct HBEMOctaveWrapperImpl
  {
    octave::interpreter interp;
  };

  HBEMOctaveWrapper::HBEMOctaveWrapper()
    : m_impl(new HBEMOctaveWrapperImpl)
  {
    int status = m_impl->interp.execute();
    if (status != 0)
      {
        throw std::runtime_error("failed to startup Octave interpreter");
      }
  }

  HBEMOctaveWrapper::~HBEMOctaveWrapper()
  {
#if OCTAVE_MAJOR_VERSION == 6
    // need to call interpreter's shutdown() method to prevent segfault on exit
    // @see: https://savannah.gnu.org/bugs/?60334
    m_impl->interp.shutdown();
#endif
    delete m_impl;
  }

  HBEMOctaveValue
  HBEMOctaveWrapper::eval_string(const std::string &eval_str)
  {
    int  parse_status;
    auto ret = m_impl->interp.eval_string(eval_str, true, parse_status);
    if (parse_status != 0)
      {
        throw std::runtime_error("parse error");
      }
    return HBEMOctaveValue(ret);
  }

  void
  HBEMOctaveWrapper::source_file(const std::string &file_name)
  {
    eval_string("source('" + file_name + "')");
  }

  void
  HBEMOctaveWrapper::add_path(const std::string &path)
  {
    ::octave_value_list in;
    in(0) = path;
    Faddpath(m_impl->interp, in);
  }

  HBEMOctaveWrapper &
  HBEMOctaveWrapper::get_instance()
  {
    static thread_local HBEMOctaveWrapper
      instance; // Guarantteed to be destroyed
    return instance;
  }

} // namespace HierBEM
