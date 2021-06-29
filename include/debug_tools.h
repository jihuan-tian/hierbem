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

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping.h>

#include <iostream>
#include <string>

using namespace dealii;

template <typename VectorType>
void
print_vector_values(std::ostream &     out,
                    const VectorType & values,
                    const std::string &sep         = std::string(","),
                    bool               has_newline = true)
{
  for (auto iter = values.cbegin(); iter != values.cend(); iter++)
    {
      if ((iter + 1) == values.cend())
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
print_vector_to_mat(std::ostream &     out,
                    const std::string &name,
                    const VectorType & values)
{
  out << "# name: " << name << "\n";
  out << "# type: matrix\n";
  out << "# rows: " << values.size() << "\n";
  out << "# columns: 1\n";

  for (auto iter = values.begin(); iter != values.end(); iter++)
    {
      out << (*iter) << "\n";
    }

  out << "\n\n";
}


template <typename MatrixType>
void
print_matrix_to_mat(std::ostream &     out,
                    const std::string &name,
                    const MatrixType & values,
                    const unsigned int precision   = 8,
                    const bool         scientific  = true,
                    const unsigned int width       = 0,
                    const char *       zero_string = "0",
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


template <typename node_pointer_type>
void
print_vector_of_tree_node_pointer_values(
  std::ostream &                        out,
  const std::vector<node_pointer_type> &tree_node_pointers,
  const std::string &                   sep = std::string(","))
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
 * DoFHandler objects, which can be then visualized in Gnuplot by executing the
 * following command.
 * 1. For spacedim=2:
 * \code
 * plot "./data_file.gpl" using 1:2:3 with labels offset 1,1 point pt 1 lc rgb \
 * "red" notitle
 * \endcode
 * 2. For spacedim=3:
 * \code
 * splot "./data_file.gpl" using 1:2:3:4 with labels offset 1,1 point pt 1 lc \
 * rgb "red"
 * \endcode
 * @param fe_system The given FiniteElement object will be checked if it has support points.
 * @param mapping
 * @param dof_handler
 * @param base_name
 */
template <int dim, int spacedim>
void
print_support_point_info(const FiniteElement<dim, spacedim> &fe,
                         const Mapping<dim, spacedim> &      mapping,
                         const DoFHandler<dim, spacedim> &   dof_handler,
                         const std::string &                 base_name)
{
  if (fe.has_support_points())
    {
      /**
       * Allocate memory for the vector storing support points.
       */
      std::map<types::global_dof_index, Point<spacedim>> support_points;
      /**
       * Get the list of support point coordinates for all DoFs. The DoFs are in
       * the default numbering starting from 0.
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
 * \brief Generate a table of DoF indices associated with each support point.
 * \details The information in the table is defined in the given Mapping and
 * DoFHandler objects, which can be then visualized in Gnuplot by executing the
 * following command.
 * 1. For spacedim=2:
 * \code
 * plot "./data_file.gpl" using 1:2:3 with labels offset 1,1 point pt 1 lc rgb \
 * "red" notitle
 * \endcode
 * 2. For spacedim=3:
 * \code
 * splot "./data_file.gpl" using 1:2:3:4 with labels offset 1,1 point pt 1 lc \
 * rgb "red"
 * \endcode
 * @param fe_system The given FESystem object will be checked if it has support points.
 * @param mapping
 * @param dof_handler
 * @param base_name
 */
template <int dim, int spacedim>
void
print_support_point_info(const FESystem<dim, spacedim> &  fe_system,
                         const Mapping<dim, spacedim> &   mapping,
                         const DoFHandler<dim, spacedim> &dof_handler,
                         const std::string &              base_name)
{
  if (fe_system.has_support_points())
    {
      /**
       * Allocate memory for the vector storing support points.
       */
      std::map<types::global_dof_index, Point<spacedim>> support_points;
      /**
       * Get the list of support point coordinates for all DoFs. The DoFs are in
       * the default numbering starting from 0.
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

#endif /* INCLUDE_DEBUG_TOOLS_H_ */
