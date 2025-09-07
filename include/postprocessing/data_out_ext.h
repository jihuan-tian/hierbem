#ifndef HIERBEM_INCLUDE_POSTPROCESSING_DATA_OUT_EXT_H_
#define HIERBEM_INCLUDE_POSTPROCESSING_DATA_OUT_EXT_H_

#include <deal.II/base/numbers.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/numerics/data_out.h>

#include <complex>
#include <memory>
#include <string>

#include "config.h"
#include "linear_algebra/linalg.h"

HBEM_NS_OPEN

using namespace dealii;

/**
 * Generate real part, imaginary part, amplitude and angle for a vector of
 * complex values, which are to be added into a @p DataOut object for
 * visualization.
 */
template <template <typename> typename VectorType, typename real_type>
class ComplexOutputDataVector
{
public:
  template <int dim,
            int spacedim,
            template <typename>
            typename VectorType1,
            typename real_type1>
  friend void
  add_complex_data_vector(
    DataOut<dim, spacedim>                                 &data_out,
    const DoFHandler<dim, spacedim>                        &dof_handler,
    const ComplexOutputDataVector<VectorType1, real_type1> &complex_data,
    const std::string                                       variable_name);

  ComplexOutputDataVector(const VectorType<std::complex<real_type>> &vec)
    : real_part(vec.size())
    , imag_part(vec.size())
    , amplitude(vec.size())
    , angle(vec.size())
  {
    LinAlg::get_vector_real_part(real_part, vec);
    LinAlg::get_vector_imag_part(imag_part, vec);
    LinAlg::get_vector_amplitude(amplitude, vec);
    LinAlg::get_vector_angle(angle, vec);
  }

private:
  VectorType<real_type> real_part;
  VectorType<real_type> imag_part;
  VectorType<real_type> amplitude;
  VectorType<real_type> angle;
};


template <int dim,
          int spacedim,
          template <typename>
          typename VectorType,
          typename real_type>
void
add_complex_data_vector(
  DataOut<dim, spacedim>                               &data_out,
  const DoFHandler<dim, spacedim>                      &dof_handler,
  const ComplexOutputDataVector<VectorType, real_type> &complex_data,
  const std::string                                     variable_name)
{
  data_out.add_data_vector(dof_handler,
                           complex_data.real_part,
                           variable_name + std::string("_real"));
  data_out.add_data_vector(dof_handler,
                           complex_data.imag_part,
                           variable_name + std::string("_imag"));
  data_out.add_data_vector(dof_handler,
                           complex_data.amplitude,
                           variable_name + std::string("_amplitude"));
  data_out.add_data_vector(dof_handler,
                           complex_data.angle,
                           variable_name + std::string("_angle"));
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_POSTPROCESSING_DATA_OUT_EXT_H_
