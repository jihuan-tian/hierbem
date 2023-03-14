/**
 * \file debug_tools.h
 * \brief This file includes a bunch of helper functions for printing out and
 * visualizing information about grid, DoFs, map, etc.
 * \ingroup toolbox
 * \date 2021-04-25
 * \author Jihuan Tian
 */
#ifndef INCLUDE_DEBUG_TOOLS_H_
#define INCLUDE_DEBUG_TOOLS_H_

#include <deal.II/base/logstream.h>
#include <deal.II/base/table.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/tria.h>

#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>

namespace IdeoBEM
{
  using namespace dealii;

  /**
   * Print the elapsed wall time and the total wall time to log stream.
   *
   * @param log
   * @param timer
   * @param activity_name
   */
  void
  print_wall_time(LogStream         &log,
                  const Timer       &timer,
                  const std::string &activity_name);


  /**
   * Print the elapsed wall time and the total wall time to output stream.
   *
   * @param out
   * @param timer
   * @param activity_name
   */
  void
  print_wall_time(std::ostream      &out,
                  const Timer       &timer,
                  const std::string &activity_name);


  /**
   * Print the elapsed CPU time and the total CPU time to log stream.
   *
   * @param log
   * @param timer
   * @param activity_name
   */
  void
  print_cpu_time(LogStream         &log,
                 const Timer       &timer,
                 const std::string &activity_name);


  /**
   * Print the elapsed CPU time and the total CPU time to output stream.
   *
   * @param out
   * @param timer
   * @param activity_name
   */
  void
  print_cpu_time(std::ostream      &out,
                 const Timer       &timer,
                 const std::string &activity_name);

  template <int dim, int spacedim>
  void
  print_mesh_info(const Triangulation<dim, spacedim> &triangulation)
  {
    std::cout << "=== Mesh info ===\n"
              << "Manifold dimension: " << dim << "\n"
              << "Space dimension: " << spacedim << "\n"
              << "No. of cells: " << triangulation.n_active_cells()
              << std::endl;

    {
      std::map<types::boundary_id, unsigned int> boundary_count;

      // Loop over each cell using TriaAccessor.
      for (auto &cell : triangulation.active_cell_iterators())
        {
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
               ++face)
            {
              if (cell->face(face)->at_boundary())
                boundary_count[cell->face(face)->boundary_id()]++;
            }
        }

      std::cout << "Boundary indicators: ";
      for (const std::pair<const types::boundary_id, unsigned int> &pair :
           boundary_count)
        {
          std::cout << pair.first << "(" << pair.second << " cells) ";
        }
      std::cout << std::endl;
    }
  }

  template <typename VectorType>
  void
  print_vector_values(std::ostream      &out,
                      const VectorType  &values,
                      const std::string &sep         = std::string(","),
                      bool               has_newline = true)
  {
    typename VectorType::const_iterator iter_copy;

    for (auto iter = values.begin(); iter != values.end();
         std::advance(iter, 1))
      {
        iter_copy = iter;
        std::advance(iter_copy, 1);

        if (iter_copy == values.end())
          if (has_newline)
            {
              out << (*iter) << std::endl;
            }
          else
            {
              out << (*iter);
            }
        else
          out << (*iter) << sep;
      }
  }


  template <typename VectorType>
  void
  print_vector_indices(std::ostream      &out,
                       const VectorType  &values,
                       const std::string &sep,
                       bool               index_starting_from_zero,
                       bool               has_newline = true)
  {
    typename VectorType::const_iterator iter_copy;

    for (auto iter = values.cbegin(); iter != values.cend();
         std::advance(iter, 1))
      {
        iter_copy = iter;
        std::advance(iter_copy, 1);

        if (iter_copy == values.cend())
          if (has_newline)
            {
              if (index_starting_from_zero)
                {
                  out << (*iter) << std::endl;
                }
              else
                {
                  out << (*iter) + 1 << std::endl;
                }
            }
          else
            {
              if (index_starting_from_zero)
                {
                  out << (*iter);
                }
              else
                {
                  out << (*iter) + 1;
                }
            }
        else
          {
            if (index_starting_from_zero)
              {
                out << (*iter) << sep;
              }
            else
              {
                out << (*iter) + 1 << sep;
              }
          }
      }
  }


  template <typename Number>
  void
  print_scalar_to_mat(std::ostream      &out,
                      const std::string &name,
                      const Number       value)
  {
    out << "# name: " << name << "\n"
        << "# type: scalar\n"
        << value << "\n";

    out << "\n\n";
  }


  template <typename VectorType>
  void
  print_vector_to_mat(std::ostream      &out,
                      const std::string &name,
                      const VectorType  &values,
                      bool               is_row_vector = false,
                      const unsigned int precision     = 6,
                      const unsigned int width         = 10)
  {
    out << "# name: " << name << "\n";
    out << "# type: matrix\n";
    if (is_row_vector)
      {
        out << "# rows: 1\n";
        out << "# columns: " << values.size() << "\n";
      }
    else
      {
        out << "# rows: " << values.size() << "\n";
        out << "# columns: 1\n";
      }

    for (auto iter = values.begin(); iter != values.end(); iter++)
      {
        out << std::setprecision(precision) << std::setw(width) << (*iter)
            << "\n";
      }

    out << "\n\n";
  }


  /**
   * Print a @p FullMatrix or a @p LAPACKFullMatrix into an Octave matrix.
   *
   * @param out
   * @param name
   * @param values
   * @param precision
   * @param scientific
   * @param width
   * @param zero_string
   * @param denominator
   * @param threshold
   */
  template <typename MatrixType>
  void
  print_matrix_to_mat(std::ostream      &out,
                      const std::string &name,
                      const MatrixType  &values,
                      const unsigned int precision   = 8,
                      const bool         scientific  = true,
                      const unsigned int width       = 0,
                      const char        *zero_string = "0",
                      const double       denominator = 1.,
                      const double       threshold   = 0.)
  {
    out << "# name: " << name << "\n";
    out << "# type: matrix\n";
    out << "# rows: " << values.m() << "\n";
    out << "# columns: " << values.n() << "\n";

    values.print_formatted(
      out, precision, scientific, width, zero_string, denominator, threshold);

    out << "\n\n";
  }

  template <typename T>
  void
  print_2d_table_to_mat(std::ostream      &out,
                        const std::string &name,
                        const Table<2, T> &values,
                        const unsigned int precision  = 8,
                        const bool         scientific = true,
                        const unsigned int width      = 0)
  {
    const typename TableBase<2, T>::size_type m = values.size(0);
    const typename TableBase<2, T>::size_type n = values.size(1);

    out << "# name: " << name << "\n";
    out << "# type: matrix\n";
    out << "# rows: " << m << "\n";
    out << "# columns: " << n << "\n";

    for (typename TableBase<2, T>::size_type i = 0; i < m; i++)
      {
        for (typename TableBase<2, T>::size_type j = 0; j < n; j++)
          {
            out << (scientific ? std::scientific : std::fixed)
                << std::setw(width) << std::setprecision(precision)
                << values(TableIndices<2>(i, j)) << " ";
          }
        out << std::endl;
      }

    out << "\n\n";
  }

  template <typename node_pointer_type>
  void
  print_vector_of_tree_node_pointer_values(
    std::ostream                         &out,
    const std::vector<node_pointer_type> &tree_node_pointers,
    const std::string                    &sep = std::string(","))
  {
    for (auto iter = tree_node_pointers.cbegin();
         iter != tree_node_pointers.cend();
         iter++)
      {
        if ((iter + 1) == tree_node_pointers.cend())
          out << (*iter)->get_data_reference() << std::endl;
        else
          out << (*iter)->get_data_reference() << sep;
      }
  }

  /**
   * \brief Generate a table of DoF indices associated with each support point.
   * \details The information in the table is defined in the given Mapping and
   * DoFHandler objects, which can be then visualized in Gnuplot by executing
   * the following command.
   * 1. For spacedim=2:
   * \code
   * plot "./data_file.gpl" using 1:2:3 with labels offset 1,1 point pt 1 lc rgb
   * \ "red" notitle \endcode
   * 2. For spacedim=3:
   * \code
   * splot "./data_file.gpl" using 1:2:3:4 with labels offset 1,1 point pt 1 lc
   * \ rgb "red" \endcode
   *
   * @param fe_system The given FiniteElement object will be checked if it has support points.
   * @param mapping
   * @param dof_handler
   * @param base_name
   */
  template <int dim, int spacedim>
  void
  print_support_point_info(const FiniteElement<dim, spacedim> &fe,
                           const Mapping<dim, spacedim>       &mapping,
                           const DoFHandler<dim, spacedim>    &dof_handler,
                           const std::string                  &base_name)
  {
    if (fe.has_support_points())
      {
        /**
         * Allocate memory for the vector storing support points.
         */
        std::map<types::global_dof_index, Point<spacedim>> support_points;
        /**
         * Get the list of support point coordinates for all DoFs. The DoFs are
         * in the default numbering starting from 0.
         */
        DoFTools::map_dofs_to_support_points(mapping,
                                             dof_handler,
                                             support_points);

        /**
         * Write the table of DoF indices for each support point into file.
         */
        std::ofstream gnuplot_file(base_name + ".gpl");
        DoFTools::write_gnuplot_dof_support_point_info(gnuplot_file,
                                                       support_points);
      }
  }


  /**
   * Generate a table of DoF indices associated with each support point. The DoF
   * indices are automatically assigned by starting from zero.
   *
   * @param support_points
   * @param base_name
   */
  template <int spacedim>
  void
  print_support_point_info(const std::vector<Point<spacedim>> &support_points,
                           const std::string                  &base_name)
  {
    /**
     * Allocate memory for the vector storing support points. The DoF numbering
     * will automatically be given by starting from zero.
     */
    std::map<types::global_dof_index, Point<spacedim>> support_points_map;

    for (unsigned int i = 0; i < support_points.size(); i++)
      {
        support_points_map[i] = support_points[i];
      }

    /**
     * Write the table of DoF indices for each support point into file.
     */
    std::ofstream gnuplot_file(base_name + ".gpl");
    DoFTools::write_gnuplot_dof_support_point_info(gnuplot_file,
                                                   support_points_map);
  }


  /**
   * \brief Generate a table of DoF indices associated with each support point.
   * \details The information in the table is defined in the given Mapping and
   * DoFHandler objects, which can be then visualized in Gnuplot by executing
   * the following command.
   * 1. For spacedim=2:
   * \code
   * plot "./data_file.gpl" using 1:2:3 with labels offset 1,1 point pt 1 lc rgb
   * \ "red" notitle \endcode
   * 2. For spacedim=3:
   * \code
   * splot "./data_file.gpl" using 1:2:3:4 with labels offset 1,1 point pt 1 lc
   * \ rgb "red" \endcode
   * @param fe_system The given FESystem object will be checked if it has support points.
   * @param mapping
   * @param dof_handler
   * @param base_name
   */
  template <int dim, int spacedim>
  void
  print_support_point_info(const FESystem<dim, spacedim>   &fe_system,
                           const Mapping<dim, spacedim>    &mapping,
                           const DoFHandler<dim, spacedim> &dof_handler,
                           const std::string               &base_name)
  {
    if (fe_system.has_support_points())
      {
        /**
         * Allocate memory for the vector storing support points.
         */
        std::map<types::global_dof_index, Point<spacedim>> support_points;
        /**
         * Get the list of support point coordinates for all DoFs. The DoFs are
         * in the default numbering starting from 0.
         */
        DoFTools::map_dofs_to_support_points(mapping,
                                             dof_handler,
                                             support_points);

        /**
         * Write the table of DoF indices for each support point into file.
         */
        std::ofstream gnuplot_file(base_name + ".gpl");
        DoFTools::write_gnuplot_dof_support_point_info(gnuplot_file,
                                                       support_points);
      }
  }


  template <int dim, int spacedim>
  void
  print_support_point_info(const MappingQGeneric<dim, spacedim> &mapping,
                           const DoFHandler<dim, spacedim>      &dof_handler,
                           const std::string                    &fe_name)
  {
    if (dof_handler.get_fe().has_support_points())
      {
        // Allocate memory for the vector storing support points.
        std::map<types::global_dof_index, Point<spacedim>> support_points;
        DoFTools::map_dofs_to_support_points(mapping,
                                             dof_handler,
                                             support_points);
        std::ofstream gnuplot_file(fe_name + ".gpl");
        DoFTools::write_gnuplot_dof_support_point_info(gnuplot_file,
                                                       support_points);
      }
  }


  /**
   * Print out the polynomial space numbering and inverse numbering of a finite
   * element in Octave format.
   *
   * @param out
   * @param fe
   * @param fe_name
   */
  template <typename FiniteElementType>
  void
  print_polynomial_space_numbering(std::ostream            &out,
                                   const FiniteElementType &fe,
                                   const std::string       &fe_name)
  {
    print_vector_to_mat(out,
                        fe_name + std::string("_poly_space_numbering"),
                        fe.get_poly_space_numbering(),
                        true);
    print_vector_to_mat(out,
                        fe_name + std::string("_poly_space_numbering_inverse"),
                        fe.get_poly_space_numbering_inverse(),
                        true);
  }


  /**
   * Print out the mapping between lexicographic numbering and hierarchic
   * numbering of a finite element in Octave format.
   *
   * @param out
   * @param fe
   * @param fe_name
   */
  template <int dim, int spacedim = dim>
  void
  print_mapping_between_lexicographic_and_hierarchic_numberings(
    std::ostream                       &out,
    const FiniteElement<dim, spacedim> &fe,
    const std::string                  &fe_name)
  {
    print_vector_to_mat(out,
                        fe_name + std::string("_lexi2hier"),
                        FETools::lexicographic_to_hierarchic_numbering<dim>(
                          fe.degree),
                        true);
    print_vector_to_mat(out,
                        fe_name + std::string("_hier2lexi"),
                        FETools::hierarchic_to_lexicographic_numbering<dim>(
                          fe.degree),
                        true);
  }


  template <int dim, int spacedim = dim>
  void
  print_triangulation_info(std::ostream                       &out,
                           const Triangulation<dim, spacedim> &triangulation)
  {
    out << "Number of cells: " << triangulation.n_active_cells() << "\n"
        << "Number of faces: " << triangulation.n_active_faces() << "\n"
        << "Number of lines: " << triangulation.n_active_lines() << "\n"
        << "Number of vertices: " << triangulation.n_vertices() << std::endl;
  }


  template <int dim, int spacedim = dim>
  void
  print_fe_info(std::ostream &out, const FiniteElement<dim, spacedim> &fe)
  {
    out << "Finite element: " << fe.get_name() << "\n"
        << "Has support points: " << (fe.has_support_points() ? "Yes" : "No")
        << "\n"
        << "dofs_per_vertex: " << fe.dofs_per_vertex << "\n"
        << "dofs_per_line: " << fe.dofs_per_line << "\n"
        << "dofs_per_quad: " << fe.dofs_per_quad << "\n"
        << "dofs_per_hex: " << fe.dofs_per_hex << "\n"
        << "dofs_per_face: " << fe.dofs_per_face << "\n"
        << "dofs_per_cell: " << fe.dofs_per_cell << "\n"
        << "components: " << fe.components << "\n"
        << "degree: " << fe.degree << std::endl;
  }
} // namespace IdeoBEM
#endif /* INCLUDE_DEBUG_TOOLS_H_ */
