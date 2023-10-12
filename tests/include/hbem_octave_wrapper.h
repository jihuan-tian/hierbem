/**
 * \file hbem_octave_wrapper.h
 * \brief Simple singleton wrapper around minimal Octave APIs to workaround
 * naming conflictions between Octave and CUDA and ease interacting
 * with Octave in testcases.
 *
 * \warning This header includes `octave/ov.h` for `octave_value` definition,
 * but it contains much more Octave types (Matrix and SparseMatrix eg) which
 * may conflict with HierBEM. If it is  the case, you have to extract all
 * HierBEM related logics into another source file and wrap them into a function
 * to be called by Octave testbase.
 *
 * \author Xiaozhe Wang
 * \date 2023-10-12
 */
#ifndef HBEM_OCTAVE_WRAPPER_H_
#define HBEM_OCTAVE_WRAPPER_H_

#include <octave/ov.h>

namespace HierBEM
{
  // Hide Octave related details
  class HBEMOctaveWrapperImpl;

  class HBEMOctaveWrapper
  {
  public:
    static HBEMOctaveWrapper &
    get_instance();

    // Deleting default copy methods to keep instance uncopyable
    HBEMOctaveWrapper(HBEMOctaveWrapper const &) = delete;
    void
    operator=(HBEMOctaveWrapper const &) = delete;

    ~HBEMOctaveWrapper();

    octave_value
    eval_string(const std::string &eval_str);
    void
    source_file(const std::string &file_name);
    void
    add_path(const std::string &path);

  private:
    HBEMOctaveWrapper();

    HBEMOctaveWrapperImpl *m_impl;
  };
} // namespace HierBEM

#endif
