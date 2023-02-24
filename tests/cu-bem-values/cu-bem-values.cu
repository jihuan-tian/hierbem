/**
 * @file cu-bem-values.cu
 * @brief Verify the initialization of BEMValues on GPU.
 *
 * @ingroup testers
 * @author Jihuan Tian
 * @date 2023-02-01
 */

#include <iostream>

#include "cu_bem_values.hcu"
#include "laplace_bem.h"

using namespace dealii;
using namespace IdeoBEM;

template <int dim>
void
print_qgauss(std::ostream &out, const QGauss<dim> &quad_cpu)
{
  for (unsigned int q = 0; q < quad_cpu.size(); q++)
    {
      out << "#" << q << ": points=(";

      for (unsigned int i = 0; i < dim; i++)
        {
          if (i == dim - 1)
            {
              out << quad_cpu.point(q)(i) << "), weights=";
            }
          else
            {
              out << quad_cpu.point(q)(i) << ",";
            }
        }

      out << quad_cpu.weight(q) << std::endl;
    }
}


/**
 * N.B. The points stored in the table adopt the C style indexing.
 *
 * @param out
 * @param points The first dimension is the quadrature point number. The second
 * dimension is the coordinate component. The index for the second dimension
 * runs faster.
 * @param weights
 */
void
print_qgauss(std::ostream           &out,
             const Table<2, double> &points,
             const double *const     weights)
{
  const unsigned int quad_num = points.size(0);
  const unsigned int dim      = points.size(1);

  for (unsigned int q = 0; q < quad_num; q++)
    {
      out << "#" << q << ": points=(";

      for (unsigned int i = 0; i < dim; i++)
        {
          if (i == dim - 1)
            {
              out << points(q, i) << "), weights=";
            }
          else
            {
              out << points(q, i) << ",";
            }
        }

      out << weights[q] << std::endl;
    }
}


/**
 * Check the equality of the quadrature objects on CPU and GPU.
 *
 * @tparam dim
 * @param quad_cpu
 * @param quad_gpu
 * @return
 */
template <int dim>
bool
is_equal_qgauss(const QGauss<dim>                            &quad_cpu,
                const IdeoBEM::CUDAWrappers::CUDAQGauss<dim> &quad_gpu)
{
  if (quad_cpu.size() == quad_gpu.size())
    {
      const unsigned int quad_num = quad_cpu.size();
      // Extract quadrature points and weights from the device for comparison.
      Table<2, double> quad_points_from_gpu(quad_num, dim);
      quad_gpu.get_points().copy_to_host(quad_points_from_gpu);
      double *weights_from_gpu = new double[quad_num];
      cudaMemcpy(weights_from_gpu,
                 quad_gpu.get_weights(),
                 sizeof(double) * quad_num,
                 cudaMemcpyDeviceToHost);

      for (unsigned int q = 0; q < quad_num; q++)
        {
          for (unsigned int i = 0; i < dim; i++)
            {
              if (quad_points_from_gpu(q, i) != quad_cpu.point(q)(i))
                {
                  std::cout << "(" << quad_cpu.point(q)(i) << ","
                            << quad_points_from_gpu(q, i) << ")" << std::endl;
                  delete weights_from_gpu;
                  return false;
                }
            }

          if (*(weights_from_gpu + q) != quad_cpu.weight(q))
            {
              std::cout << "(" << quad_cpu.weight(q) << ","
                        << *(weights_from_gpu + q) << ")" << std::endl;
              delete weights_from_gpu;
              return false;
            }
        }

      delete weights_from_gpu;
      return true;
    }
  else
    {
      std::cout << "(" << quad_cpu.size() << "," << quad_gpu.size() << ")"
                << std::endl;
      return false;
    }
}


template <typename T>
bool
is_equal_shape_value_tables(
  const Table<2, T>                            &table_cpu,
  const IdeoBEM::CUDAWrappers::CUDATable<2, T> &table_gpu)
{
  const unsigned int N = 2;

  // Make a copy of the GPU table on the host.
  TableIndices<N> table_sizes;
  IdeoBEM::CUDAWrappers::copy_table_indices(table_sizes, table_gpu.size());
  Table<N, T> table_copied_from_gpu(table_sizes[0], table_sizes[1]);
  table_gpu.copy_to_host(table_copied_from_gpu);

  // Check the equality of number of elements.
  if (table_cpu.n_elements() == table_copied_from_gpu.n_elements())
    {
      // Get the pointers to the first element in the two tables.
      const T *table_cpu_ptr      = &(table_cpu(TableIndices<N>()));
      const T *table_from_gpu_ptr = &(table_copied_from_gpu(TableIndices<N>()));

      for (std::size_t i = 0; i < table_cpu.n_elements(); i++)
        {
          if (*(table_cpu_ptr + i) != *(table_from_gpu_ptr + i))
            {
              return false;
            }
        }

      return true;
    }
  else
    {
      return false;
    }
}


template <typename T>
bool
is_equal_shape_value_tables(
  const Table<3, T>                            &table_cpu,
  const IdeoBEM::CUDAWrappers::CUDATable<3, T> &table_gpu)
{
  const unsigned int N = 3;

  // Make a copy of the GPU table on the host.
  TableIndices<N> table_sizes;
  IdeoBEM::CUDAWrappers::copy_table_indices(table_sizes, table_gpu.size());
  Table<N, T> table_copied_from_gpu(table_sizes[0],
                                    table_sizes[1],
                                    table_sizes[2]);
  table_gpu.copy_to_host(table_copied_from_gpu);

  // Check the equality of number of elements.
  if (table_cpu.n_elements() == table_copied_from_gpu.n_elements())
    {
      // Get the pointers to the first element in the two tables.
      const T *table_cpu_ptr      = &(table_cpu(TableIndices<N>()));
      const T *table_from_gpu_ptr = &(table_copied_from_gpu(TableIndices<N>()));

      for (std::size_t i = 0; i < table_cpu.n_elements(); i++)
        {
          if (*(table_cpu_ptr + i) != *(table_from_gpu_ptr + i))
            {
              return false;
            }
        }

      return true;
    }
  else
    {
      return false;
    }
}


template <typename T>
bool
is_equal_shape_grad_matrix_tables(
  const Table<2, LAPACKFullMatrixExt<T>>       &table_cpu,
  const IdeoBEM::CUDAWrappers::CUDATable<4, T> &table_gpu)
{
  const unsigned int N = 4;

  // Make a copy of the GPU table on the host.
  TableIndices<N> table_sizes;
  IdeoBEM::CUDAWrappers::copy_table_indices(table_sizes, table_gpu.size());
  Table<N, T> table_copied_from_gpu(table_sizes[0],
                                    table_sizes[1],
                                    table_sizes[2],
                                    table_sizes[3]);
  table_gpu.copy_to_host(table_copied_from_gpu);

  // Get the gradient matrix sizes.
  const unsigned int m = table_cpu(0, 0).m();
  const unsigned int n = table_cpu(0, 0).n();

  // Check the equality of number of elements.
  if (table_cpu.n_elements() * m * n == table_copied_from_gpu.n_elements())
    {
      // Get table sizes.
      const unsigned int k3_terms = table_cpu.n_rows();
      const unsigned int quad_num = table_cpu.n_cols();

      std::size_t counter         = 0;
      const T *table_from_gpu_ptr = &(table_copied_from_gpu(TableIndices<N>()));

      for (unsigned int k = 0; k < k3_terms; k++)
        {
          for (unsigned int q = 0; q < quad_num; q++)
            {
              for (unsigned int j = 0; j < n; j++)
                {
                  for (unsigned int i = 0; i < m; i++)
                    {
                      if (*(table_from_gpu_ptr + counter) !=
                          table_cpu(k, q)(i, j))
                        {
                          std::cout << "(" << table_cpu(k, q)(i, j) << ","
                                    << *(table_from_gpu_ptr + counter) << ")"
                                    << std::endl;
                          return false;
                        }
                      else
                        {
                          counter++;
                        }
                    }
                }
            }
        }

      return true;
    }
  else
    {
      std::cout << "(" << table_cpu.n_elements() * m * n << ","
                << table_copied_from_gpu.n_elements() << ")" << std::endl;
      return false;
    }
}


int
main()
{
  const unsigned     dim      = 2;
  const unsigned int spacedim = 3;

  FE_Q<dim, spacedim>   fe_for_dirichlet_space(3);
  FE_DGQ<dim, spacedim> fe_for_neumann_space(2);

  MappingQGenericExt<dim, spacedim> kx_mapping(1);
  MappingQGenericExt<dim, spacedim> ky_mapping(1);

  std::unique_ptr<typename MappingQGeneric<dim, spacedim>::InternalData>
    kx_mapping_data;
  std::unique_ptr<typename MappingQGeneric<dim, spacedim>::InternalData>
    ky_mapping_data;

  std::unique_ptr<typename Mapping<dim, spacedim>::InternalDataBase>
    kx_mapping_database = kx_mapping.get_data(update_default, QGauss<dim>(1));

  /**
   * Downcast the smart pointer of @p Mapping<dim, spacedim>::InternalDataBase to
   * @p MappingQGeneric<dim,spacedim>::InternalData by first unwrapping
   * the original smart pointer via @p static_cast then wrapping it again.
   */
  kx_mapping_data =
    std::unique_ptr<typename MappingQGeneric<dim, spacedim>::InternalData>(
      static_cast<typename MappingQGeneric<dim, spacedim>::InternalData *>(
        kx_mapping_database.release()));

  std::unique_ptr<typename Mapping<dim, spacedim>::InternalDataBase>
    ky_mapping_database = ky_mapping.get_data(update_default, QGauss<dim>(1));

  ky_mapping_data =
    std::unique_ptr<typename MappingQGeneric<dim, spacedim>::InternalData>(
      static_cast<typename MappingQGeneric<dim, spacedim>::InternalData *>(
        ky_mapping_database.release()));

  SauterQuadratureRule<dim> sauter_quad_rule(5, 4, 4, 3);

  BEMValues<dim, spacedim, double> bem_values_cpu(
    fe_for_dirichlet_space,
    fe_for_neumann_space,
    *kx_mapping_data,
    *ky_mapping_data,
    sauter_quad_rule.quad_rule_for_same_panel,
    sauter_quad_rule.quad_rule_for_common_edge,
    sauter_quad_rule.quad_rule_for_common_vertex,
    sauter_quad_rule.quad_rule_for_regular);

  bem_values_cpu.fill_shape_function_value_tables();

  IdeoBEM::CUDAWrappers::CUDABEMValues<dim, spacedim> bem_values_gpu;
  bem_values_gpu.allocate_and_assign_from_host(bem_values_cpu);

  if (is_equal_qgauss(bem_values_cpu.quad_rule_for_same_panel,
                      bem_values_gpu.quad_rule_for_same_panel))
    {
      std::cout << "quad_rule_for_same_panel is equal" << std::endl;
    }
  else
    {
      std::cout << "quad_rule_for_same_panel is not equal" << std::endl;
    }

  if (is_equal_qgauss(bem_values_cpu.quad_rule_for_common_edge,
                      bem_values_gpu.quad_rule_for_common_edge))
    {
      std::cout << "quad_rule_for_common_edge is equal" << std::endl;
    }
  else
    {
      std::cout << "quad_rule_for_common_edge is not equal" << std::endl;
    }

  if (is_equal_qgauss(bem_values_cpu.quad_rule_for_common_vertex,
                      bem_values_gpu.quad_rule_for_common_vertex))
    {
      std::cout << "quad_rule_for_common_vertex is equal" << std::endl;
    }
  else
    {
      std::cout << "quad_rule_for_common_vertex is not equal" << std::endl;
    }

  if (is_equal_qgauss(bem_values_cpu.quad_rule_for_regular,
                      bem_values_gpu.quad_rule_for_regular))
    {
      std::cout << "quad_rule_for_regular is equal" << std::endl;
    }
  else
    {
      std::cout << "quad_rule_for_regular is not equal" << std::endl;
    }

  if (is_equal_shape_value_tables(
        bem_values_cpu.kx_shape_value_table_for_same_panel,
        bem_values_gpu.kx_shape_value_table_for_same_panel))
    {
      std::cout << "kx_shape_value_table_for_same_panel is equal" << std::endl;
    }
  else
    {
      std::cout << "kx_shape_value_table_for_same_panel is not equal"
                << std::endl;
    }

  if (is_equal_shape_value_tables(
        bem_values_cpu.ky_shape_value_table_for_same_panel,
        bem_values_gpu.ky_shape_value_table_for_same_panel))
    {
      std::cout << "ky_shape_value_table_for_same_panel is equal" << std::endl;
    }
  else
    {
      std::cout << "ky_shape_value_table_for_same_panel is not equal"
                << std::endl;
    }

  if (is_equal_shape_value_tables(
        bem_values_cpu.kx_shape_value_table_for_common_edge,
        bem_values_gpu.kx_shape_value_table_for_common_edge))
    {
      std::cout << "kx_shape_value_table_for_common_edge is equal" << std::endl;
    }
  else
    {
      std::cout << "kx_shape_value_table_for_common_edge is not equal"
                << std::endl;
    }

  if (is_equal_shape_value_tables(
        bem_values_cpu.ky_shape_value_table_for_common_edge,
        bem_values_gpu.ky_shape_value_table_for_common_edge))
    {
      std::cout << "ky_shape_value_table_for_common_edge is equal" << std::endl;
    }
  else
    {
      std::cout << "ky_shape_value_table_for_common_edge is not equal"
                << std::endl;
    }

  if (is_equal_shape_value_tables(
        bem_values_cpu.kx_shape_value_table_for_common_vertex,
        bem_values_gpu.kx_shape_value_table_for_common_vertex))
    {
      std::cout << "kx_shape_value_table_for_common_vertex is equal"
                << std::endl;
    }
  else
    {
      std::cout << "kx_shape_value_table_for_common_vertex is not equal"
                << std::endl;
    }

  if (is_equal_shape_value_tables(
        bem_values_cpu.ky_shape_value_table_for_common_vertex,
        bem_values_gpu.ky_shape_value_table_for_common_vertex))
    {
      std::cout << "ky_shape_value_table_for_common_vertex is equal"
                << std::endl;
    }
  else
    {
      std::cout << "ky_shape_value_table_for_common_vertex is not equal"
                << std::endl;
    }

  if (is_equal_shape_value_tables(
        bem_values_cpu.kx_shape_value_table_for_regular,
        bem_values_gpu.kx_shape_value_table_for_regular))
    {
      std::cout << "kx_shape_value_table_for_regular is equal" << std::endl;
    }
  else
    {
      std::cout << "kx_shape_value_table_for_regular is not equal" << std::endl;
    }

  if (is_equal_shape_value_tables(
        bem_values_cpu.ky_shape_value_table_for_regular,
        bem_values_gpu.ky_shape_value_table_for_regular))
    {
      std::cout << "ky_shape_value_table_for_regular is equal" << std::endl;
    }
  else
    {
      std::cout << "ky_shape_value_table_for_regular is not equal" << std::endl;
    }

  if (is_equal_shape_value_tables(
        bem_values_cpu.kx_mapping_shape_value_table_for_same_panel,
        bem_values_gpu.kx_mapping_shape_value_table_for_same_panel))
    {
      std::cout << "kx_mapping_shape_value_table_for_same_panel is equal"
                << std::endl;
    }
  else
    {
      std::cout << "kx_mapping_shape_value_table_for_same_panel is not equal"
                << std::endl;
    }

  if (is_equal_shape_value_tables(
        bem_values_cpu.ky_mapping_shape_value_table_for_same_panel,
        bem_values_gpu.ky_mapping_shape_value_table_for_same_panel))
    {
      std::cout << "ky_mapping_shape_value_table_for_same_panel is equal"
                << std::endl;
    }
  else
    {
      std::cout << "ky_mapping_shape_value_table_for_same_panel is not equal"
                << std::endl;
    }

  if (is_equal_shape_value_tables(
        bem_values_cpu.kx_mapping_shape_value_table_for_common_edge,
        bem_values_gpu.kx_mapping_shape_value_table_for_common_edge))
    {
      std::cout << "kx_mapping_shape_value_table_for_common_edge is equal"
                << std::endl;
    }
  else
    {
      std::cout << "kx_mapping_shape_value_table_for_common_edge is not equal"
                << std::endl;
    }

  if (is_equal_shape_value_tables(
        bem_values_cpu.ky_mapping_shape_value_table_for_common_edge,
        bem_values_gpu.ky_mapping_shape_value_table_for_common_edge))
    {
      std::cout << "ky_mapping_shape_value_table_for_common_edge is equal"
                << std::endl;
    }
  else
    {
      std::cout << "ky_mapping_shape_value_table_for_common_edge is not equal"
                << std::endl;
    }

  if (is_equal_shape_value_tables(
        bem_values_cpu.kx_mapping_shape_value_table_for_common_vertex,
        bem_values_gpu.kx_mapping_shape_value_table_for_common_vertex))
    {
      std::cout << "kx_mapping_shape_value_table_for_common_vertex is equal"
                << std::endl;
    }
  else
    {
      std::cout << "kx_mapping_shape_value_table_for_common_vertex is not equal"
                << std::endl;
    }

  if (is_equal_shape_value_tables(
        bem_values_cpu.ky_mapping_shape_value_table_for_common_vertex,
        bem_values_gpu.ky_mapping_shape_value_table_for_common_vertex))
    {
      std::cout << "ky_mapping_shape_value_table_for_common_vertex is equal"
                << std::endl;
    }
  else
    {
      std::cout << "ky_mapping_shape_value_table_for_common_vertex is not equal"
                << std::endl;
    }

  if (is_equal_shape_value_tables(
        bem_values_cpu.kx_mapping_shape_value_table_for_regular,
        bem_values_gpu.kx_mapping_shape_value_table_for_regular))
    {
      std::cout << "kx_mapping_shape_value_table_for_regular is equal"
                << std::endl;
    }
  else
    {
      std::cout << "kx_mapping_shape_value_table_for_regular is not equal"
                << std::endl;
    }

  if (is_equal_shape_value_tables(
        bem_values_cpu.ky_mapping_shape_value_table_for_regular,
        bem_values_gpu.ky_mapping_shape_value_table_for_regular))
    {
      std::cout << "ky_mapping_shape_value_table_for_regular is equal"
                << std::endl;
    }
  else
    {
      std::cout << "ky_mapping_shape_value_table_for_regular is not equal"
                << std::endl;
    }

  if (is_equal_shape_grad_matrix_tables(
        bem_values_cpu.kx_shape_grad_matrix_table_for_same_panel,
        bem_values_gpu.kx_shape_grad_matrix_table_for_same_panel))
    {
      std::cout << "kx_shape_grad_matrix_table_for_same_panel is equal"
                << std::endl;
    }
  else
    {
      std::cout << "kx_shape_grad_matrix_table_for_same_panel is not equal"
                << std::endl;
    }

  if (is_equal_shape_grad_matrix_tables(
        bem_values_cpu.ky_shape_grad_matrix_table_for_same_panel,
        bem_values_gpu.ky_shape_grad_matrix_table_for_same_panel))
    {
      std::cout << "ky_shape_grad_matrix_table_for_same_panel is equal"
                << std::endl;
    }
  else
    {
      std::cout << "ky_shape_grad_matrix_table_for_same_panel is not equal"
                << std::endl;
    }

  if (is_equal_shape_grad_matrix_tables(
        bem_values_cpu.kx_shape_grad_matrix_table_for_common_edge,
        bem_values_gpu.kx_shape_grad_matrix_table_for_common_edge))
    {
      std::cout << "kx_shape_grad_matrix_table_for_common_edge is equal"
                << std::endl;
    }
  else
    {
      std::cout << "kx_shape_grad_matrix_table_for_common_edge is not equal"
                << std::endl;
    }

  if (is_equal_shape_grad_matrix_tables(
        bem_values_cpu.ky_shape_grad_matrix_table_for_common_edge,
        bem_values_gpu.ky_shape_grad_matrix_table_for_common_edge))
    {
      std::cout << "ky_shape_grad_matrix_table_for_common_edge is equal"
                << std::endl;
    }
  else
    {
      std::cout << "ky_shape_grad_matrix_table_for_common_edge is not equal"
                << std::endl;
    }

  if (is_equal_shape_grad_matrix_tables(
        bem_values_cpu.kx_shape_grad_matrix_table_for_common_vertex,
        bem_values_gpu.kx_shape_grad_matrix_table_for_common_vertex))
    {
      std::cout << "kx_shape_grad_matrix_table_for_common_vertex is equal"
                << std::endl;
    }
  else
    {
      std::cout << "kx_shape_grad_matrix_table_for_common_vertex is not equal"
                << std::endl;
    }

  if (is_equal_shape_grad_matrix_tables(
        bem_values_cpu.ky_shape_grad_matrix_table_for_common_vertex,
        bem_values_gpu.ky_shape_grad_matrix_table_for_common_vertex))
    {
      std::cout << "ky_shape_grad_matrix_table_for_common_vertex is equal"
                << std::endl;
    }
  else
    {
      std::cout << "ky_shape_grad_matrix_table_for_common_vertex is not equal"
                << std::endl;
    }

  if (is_equal_shape_grad_matrix_tables(
        bem_values_cpu.kx_shape_grad_matrix_table_for_regular,
        bem_values_gpu.kx_shape_grad_matrix_table_for_regular))
    {
      std::cout << "kx_shape_grad_matrix_table_for_regular is equal"
                << std::endl;
    }
  else
    {
      std::cout << "kx_shape_grad_matrix_table_for_regular is not equal"
                << std::endl;
    }

  if (is_equal_shape_grad_matrix_tables(
        bem_values_cpu.ky_shape_grad_matrix_table_for_regular,
        bem_values_gpu.ky_shape_grad_matrix_table_for_regular))
    {
      std::cout << "ky_shape_grad_matrix_table_for_regular is equal"
                << std::endl;
    }
  else
    {
      std::cout << "ky_shape_grad_matrix_table_for_regular is not equal"
                << std::endl;
    }

  if (is_equal_shape_grad_matrix_tables(
        bem_values_cpu.kx_mapping_shape_grad_matrix_table_for_same_panel,
        bem_values_gpu.kx_mapping_shape_grad_matrix_table_for_same_panel))
    {
      std::cout << "kx_mapping_shape_grad_matrix_table_for_same_panel is equal"
                << std::endl;
    }
  else
    {
      std::cout
        << "kx_mapping_shape_grad_matrix_table_for_same_panel is not equal"
        << std::endl;
    }

  if (is_equal_shape_grad_matrix_tables(
        bem_values_cpu.ky_mapping_shape_grad_matrix_table_for_same_panel,
        bem_values_gpu.ky_mapping_shape_grad_matrix_table_for_same_panel))
    {
      std::cout << "ky_mapping_shape_grad_matrix_table_for_same_panel is equal"
                << std::endl;
    }
  else
    {
      std::cout
        << "ky_mapping_shape_grad_matrix_table_for_same_panel is not equal"
        << std::endl;
    }

  if (is_equal_shape_grad_matrix_tables(
        bem_values_cpu.kx_mapping_shape_grad_matrix_table_for_common_edge,
        bem_values_gpu.kx_mapping_shape_grad_matrix_table_for_common_edge))
    {
      std::cout << "kx_mapping_shape_grad_matrix_table_for_common_edge is equal"
                << std::endl;
    }
  else
    {
      std::cout
        << "kx_mapping_shape_grad_matrix_table_for_common_edge is not equal"
        << std::endl;
    }

  if (is_equal_shape_grad_matrix_tables(
        bem_values_cpu.ky_mapping_shape_grad_matrix_table_for_common_edge,
        bem_values_gpu.ky_mapping_shape_grad_matrix_table_for_common_edge))
    {
      std::cout << "ky_mapping_shape_grad_matrix_table_for_common_edge is equal"
                << std::endl;
    }
  else
    {
      std::cout
        << "ky_mapping_shape_grad_matrix_table_for_common_edge is not equal"
        << std::endl;
    }

  if (is_equal_shape_grad_matrix_tables(
        bem_values_cpu.kx_mapping_shape_grad_matrix_table_for_common_vertex,
        bem_values_gpu.kx_mapping_shape_grad_matrix_table_for_common_vertex))
    {
      std::cout
        << "kx_mapping_shape_grad_matrix_table_for_common_vertex is equal"
        << std::endl;
    }
  else
    {
      std::cout
        << "kx_mapping_shape_grad_matrix_table_for_common_vertex is not equal"
        << std::endl;
    }

  if (is_equal_shape_grad_matrix_tables(
        bem_values_cpu.ky_mapping_shape_grad_matrix_table_for_common_vertex,
        bem_values_gpu.ky_mapping_shape_grad_matrix_table_for_common_vertex))
    {
      std::cout
        << "ky_mapping_shape_grad_matrix_table_for_common_vertex is equal"
        << std::endl;
    }
  else
    {
      std::cout
        << "ky_mapping_shape_grad_matrix_table_for_common_vertex is not equal"
        << std::endl;
    }

  if (is_equal_shape_grad_matrix_tables(
        bem_values_cpu.kx_mapping_shape_grad_matrix_table_for_regular,
        bem_values_gpu.kx_mapping_shape_grad_matrix_table_for_regular))
    {
      std::cout << "kx_mapping_shape_grad_matrix_table_for_regular is equal"
                << std::endl;
    }
  else
    {
      std::cout << "kx_mapping_shape_grad_matrix_table_for_regular is not equal"
                << std::endl;
    }

  if (is_equal_shape_grad_matrix_tables(
        bem_values_cpu.ky_mapping_shape_grad_matrix_table_for_regular,
        bem_values_gpu.ky_mapping_shape_grad_matrix_table_for_regular))
    {
      std::cout << "ky_mapping_shape_grad_matrix_table_for_regular is equal"
                << std::endl;
    }
  else
    {
      std::cout << "ky_mapping_shape_grad_matrix_table_for_regular is not equal"
                << std::endl;
    }

  bem_values_gpu.release();

  return 0;
}
