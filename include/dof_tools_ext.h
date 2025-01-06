/**
 * @file dof_tools_ext.h
 * @brief Introduction of dof_tools_ext.h
 *
 * @date 2022-11-16
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_DOF_TOOLS_EXT_H_
#define HIERBEM_INCLUDE_DOF_TOOLS_EXT_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/geometry_info.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/sparsity_pattern.h>

#include <algorithm>
#include <map>
#include <vector>

#include "config.h"
#include "dof_to_cell_topology.h"
#include "gmsh_manipulation.h"

HBEM_NS_OPEN

using namespace dealii;

namespace DoFToolsExt
{
  /**
   * N.B. This function appears here because the already instantiated deal.ii
   * versions only assume <code>dim==spacedim</code>. Meanwhile, the general
   * template version only appears in @p dof_tool.cc, which is not accessible
   * to the outside.
   *
   * @param dof_handler
   * @param subdomain_id
   * @param selected_dofs
   */
  template <int dim, int spacedim>
  void
  extract_subdomain_dofs(const DoFHandler<dim, spacedim>     &dof_handler,
                         const std::set<types::subdomain_id> &subdomain_ids,
                         std::vector<bool>                   &selected_dofs)
  {
    Assert(selected_dofs.size() == dof_handler.n_dofs(),
           ExcDimensionMismatch(selected_dofs.size(), dof_handler.n_dofs()));

    // preset all values by false
    std::fill_n(selected_dofs.begin(), dof_handler.n_dofs(), false);

    // Global DoF indices for the current cell.
    std::vector<types::global_dof_index> cell_dof_indices;
    cell_dof_indices.reserve(
      dof_handler.get_fe_collection().max_dofs_per_cell());

    // this function is similar to the make_sparsity_pattern function, see
    // there for more information
    typename DoFHandler<dim, spacedim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
    for (; cell != endc; ++cell)
      {
        // Find the current cell's subdomain id in the given list.
        auto found_iter = subdomain_ids.find(cell->subdomain_id());

        if (found_iter != subdomain_ids.end())
          {
            const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
            cell_dof_indices.resize(dofs_per_cell);
            cell->get_dof_indices(cell_dof_indices);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              selected_dofs[cell_dof_indices[i]] = true;
          }
      }
  }


  /**
   * Mark the DoFs in cells which have a material id belonging to the given
   * collection.
   *
   * @param dof_handler
   * @param material_ids
   * @param selected_dofs
   * @param reset_selectors_to_false If preset all selectors to false at the
   * beginning of this function.
   */
  template <int dim, int spacedim>
  void
  extract_material_domain_dofs(const DoFHandler<dim, spacedim>    &dof_handler,
                               const std::set<types::material_id> &material_ids,
                               std::vector<bool> &selected_dofs,
                               const bool reset_selectors_to_false = true)
  {
    Assert(selected_dofs.size() == dof_handler.n_dofs(),
           ExcDimensionMismatch(selected_dofs.size(), dof_handler.n_dofs()));

    if (reset_selectors_to_false)
      {
        // preset all values by false
        std::fill_n(selected_dofs.begin(), dof_handler.n_dofs(), false);
      }

    // Global DoF indices for the current cell.
    std::vector<types::global_dof_index> cell_dof_indices;
    cell_dof_indices.reserve(
      dof_handler.get_fe_collection().max_dofs_per_cell());

    // this function is similar to the make_sparsity_pattern function, see
    // there for more information
    typename DoFHandler<dim, spacedim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
    for (; cell != endc; ++cell)
      {
        // Find the current cell's material id in the given list.
        auto found_iter = material_ids.find(cell->material_id());

        if (found_iter != material_ids.end())
          {
            const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
            cell_dof_indices.resize(dofs_per_cell);
            cell->get_dof_indices(cell_dof_indices);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                selected_dofs[cell_dof_indices[i]] = true;
              }
          }
      }
  }


  /**
   * @brief Generate DoF selectors for those on the boundary condition.
   *
   * @tparam dim
   * @tparam spacedim
   * @param dof_handler
   * @param boundary_bc_definition
   * @param selected_dofs
   * @param reset_selectors_to_false
   */
  template <int dim, int spacedim>
  void
  extract_boundary_condition_dofs(
    const DoFHandler<dim, spacedim>                   &dof_handler,
    std::map<EntityTag, Function<spacedim, double> *> &boundary_bc_definition,
    std::vector<bool>                                 &selected_dofs,
    const bool reset_selectors_to_false = true)
  {
    Assert(selected_dofs.size() == dof_handler.n_dofs(),
           ExcDimensionMismatch(selected_dofs.size(), dof_handler.n_dofs()));

    if (reset_selectors_to_false)
      {
        // preset all values by false
        std::fill_n(selected_dofs.begin(), dof_handler.n_dofs(), false);
      }

    // Global DoF indices for the current cell.
    std::vector<types::global_dof_index> cell_dof_indices;
    cell_dof_indices.reserve(
      dof_handler.get_fe_collection().max_dofs_per_cell());

    // this function is similar to the make_sparsity_pattern function, see
    // there for more information
    typename DoFHandler<dim, spacedim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
    for (; cell != endc; ++cell)
      {
        // Find the current cell's material id in the given list.
        auto found_iter = boundary_bc_definition.find(cell->material_id());

        if (found_iter != boundary_bc_definition.end())
          {
            const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
            cell_dof_indices.resize(dofs_per_cell);
            cell->get_dof_indices(cell_dof_indices);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                selected_dofs[cell_dof_indices[i]] = true;
              }
          }
      }
  }


  template <int dim, int spacedim>
  void
  extract_material_domain_dofs_excluding_boundary_dofs(
    const DoFHandler<dim, spacedim>    &dof_handler,
    const std::set<types::material_id> &boundary_cell_material_ids,
    std::vector<bool>                  &selected_dofs)
  {
    Assert(selected_dofs.size() == dof_handler.n_dofs(),
           ExcDimensionMismatch(selected_dofs.size(), dof_handler.n_dofs()));

    // preset all values by true.
    std::fill_n(selected_dofs.begin(), dof_handler.n_dofs(), true);

    std::vector<types::global_dof_index> face_dof_indices(
      dof_handler.get_fe().dofs_per_face);

    typename DoFHandler<dim, spacedim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
    for (; cell != endc; ++cell)
      {
        // Find the current cell's material id in the list of ids for the
        // boundary cells.
        auto found_iter = boundary_cell_material_ids.find(cell->material_id());

        if (found_iter != boundary_cell_material_ids.end())
          {
            // Iterate over each face, which is actually a line in
            // this case, and check if it completely lies at boundary.
            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; f++)
              {
                // Get the face iterator as a @p DoFAccessor, but it
                // should be converted to a @p CellAccessor to call
                // @p at_boundary.
                if (cell->at_boundary(f))
                  {
                    // Get the face iterator as a @p DoFCellAccessor.
                    cell->face(f)->get_dof_indices(face_dof_indices);

                    for (unsigned int d = 0;
                         d < dof_handler.get_fe().dofs_per_face;
                         d++)
                      {
                        selected_dofs[face_dof_indices[d]] = false;
                      }
                  }
              }
          }
      }
  }


  /**
   * Return a list of support points for the local DoFs selected from the full
   * list of DoFs handled by the DoF handler.
   *
   * @pre
   * @post
   * @tparam dim
   * @tparam spacedim
   * @param mapping The mapping from the reference cell to the real cell on
   * which DoFs are defined.
   * @param dof_handler The DoF handler object, in which the finite element
   * object should provide support points.
   * @param map_from_local_to_full_dof_indices This is a vector which stores
   * a list of global DoF indices corresponding to the DoFs held in the
   * @p dof_handler, while the element indices of this vector play the role
   * of local DoF indices corresponding to the selected DoFs. Therefore, this
   * vector is used as a map from local to global DoF indices.
   * @param support_points A vector that stores the corresponding location of
   * the DoFs in real space coordinates. Before calling this function, this
   * vector should has the same size as the number of selected local DoFs,
   * which is also equal to the size of @p map_from_local_to_full_dof_indices.
   * Previous content of this object is deleted in this function.
   */
  template <int dim, int spacedim>
  void
  map_dofs_to_support_points(const Mapping<dim, spacedim>    &mapping,
                             const DoFHandler<dim, spacedim> &dof_handler,
                             const std::vector<types::global_dof_index>
                               &map_from_local_to_full_dof_indices,
                             std::vector<Point<spacedim>> &support_points)
  {
    const types::global_dof_index n_dofs =
      map_from_local_to_full_dof_indices.size();
    AssertDimension(n_dofs, support_points.size());

    std::vector<Point<spacedim>> full_support_points(dof_handler.n_dofs());
    DoFTools::map_dofs_to_support_points(mapping,
                                         dof_handler,
                                         full_support_points);

    for (types::global_dof_index d = 0; d < n_dofs; d++)
      {
        support_points[d] =
          full_support_points[map_from_local_to_full_dof_indices[d]];
      }
  }


  /**
   * @brief Return a list of support points for DoFs on the specified level in
   * the DoF handler.
   *
   * The result is a map from DoF indices to support points, which can be
   * passed to DoFTools::write_gnuplot_dof_support_point_info for visualizing
   * the distribution of support points.
   *
   * @tparam dim
   * @tparam spacedim
   * @param mapping
   * @param dof_handler
   * @param level
   * @param support_points
   */
  template <int dim, int spacedim>
  void
  map_mg_dofs_to_support_points(
    const Mapping<dim, spacedim>                       &mapping,
    const DoFHandler<dim, spacedim>                    &dof_handler,
    const unsigned int                                  level,
    std::map<types::global_dof_index, Point<spacedim>> &support_points)
  {
    support_points.clear();

    // Get the unit support point coordinates.
    const std::vector<Point<dim>> &unit_supports =
      dof_handler.get_fe().get_unit_support_points();

    std::vector<types::global_dof_index> dof_indices_in_cell(
      dof_handler.get_fe().dofs_per_cell);

    // Iterate over each cell on the specified level.
    for (const auto &cell : dof_handler.mg_cell_iterators_on_level(level))
      {
        // Get the DoF indices in the current cell.
        cell->get_mg_dof_indices(dof_indices_in_cell);

        // Transform each unit support point to real cell and added to the
        // result map.
        unsigned int i = 0;
        for (const auto &p : unit_supports)
          {
            support_points[dof_indices_in_cell[i]] =
              mapping.transform_unit_to_real_cell(cell, p);
            i++;
          }
      }
  }


  /**
   * @brief Return a list of support points for DoFs on the specified level in
   * the DoF handler.
   *
   * @tparam dim
   * @tparam spacedim
   * @param mapping
   * @param dof_handler
   * @param level
   * @param support_points
   */
  template <int dim, int spacedim>
  void
  map_mg_dofs_to_support_points(const Mapping<dim, spacedim>    &mapping,
                                const DoFHandler<dim, spacedim> &dof_handler,
                                const unsigned int               level,
                                std::vector<Point<spacedim>>    &support_points)
  {
    AssertDimension(dof_handler.n_dofs(level), support_points.size());

    // Get the unit support point coordinates.
    const std::vector<Point<dim>> &unit_supports =
      dof_handler.get_fe().get_unit_support_points();

    std::vector<types::global_dof_index> dof_indices_in_cell(
      dof_handler.get_fe().dofs_per_cell);

    // Iterate over each cell on the specified level.
    for (const auto &cell : dof_handler.mg_cell_iterators_on_level(level))
      {
        // Get the DoF indices in the current cell.
        cell->get_mg_dof_indices(dof_indices_in_cell);

        // Transform each unit support point to real cell.
        unsigned int i = 0;
        for (const auto &p : unit_supports)
          {
            support_points[dof_indices_in_cell[i]] =
              mapping.transform_unit_to_real_cell(cell, p);
            i++;
          }
      }
  }


  /**
   * Calculate the average cell sizes associated with those DoFs in the given
   * DoF handler object.
   *
   * \mynote{The doubled cell size will be used as an estimate for the
   * diameter of the support set of each DoF.}
   *
   * @param dof_handler DoF handler object.
   * @param dof_average_cell_size The returned list of average cell sizes
   * which corresponds to the DoFs held within the DoF handler object. The
   * memory for this vector should be preallocated and initialized to zero
   * before calling this function.
   */
  template <int dim, int spacedim, typename Number = double>
  void
  map_dofs_to_average_cell_size(const DoFHandler<dim, spacedim> &dof_handler,
                                std::vector<Number> &dof_average_cell_size)
  {
    const unsigned int n_dofs = dof_handler.n_dofs();
    AssertDimension(n_dofs, dof_average_cell_size.size());

    /**
     * Create the vector which stores the number of cells that share a common
     * DoF for each DoF.
     */
    std::vector<unsigned int> number_of_cells_sharing_dof(n_dofs, 0);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        /**
         * Get the diameter of the current cell.
         */
        Number cell_diameter = cell->diameter();

        /**
         * Get DoF indices local to this cell.
         */
        std::vector<types::global_dof_index> dof_indices(
          cell->get_fe().dofs_per_cell);
        cell->get_dof_indices(dof_indices);

        for (const auto index : dof_indices)
          {
            number_of_cells_sharing_dof.at(index) += 1;
            dof_average_cell_size.at(index) += cell_diameter;
          }
      }

    for (unsigned int i = 0; i < n_dofs; i++)
      {
        dof_average_cell_size.at(i) /= number_of_cells_sharing_dof.at(i);
      }
  }


  /**
   * @brief Calculate the average cell sizes associated with those DoFs on a
   * specific level in the given DoF handler object.
   *
   * @tparam Number
   * @tparam dim
   * @tparam spacedim
   * @param dof_handler
   * @param level
   * @param dof_average_cell_size
   */
  template <int dim, int spacedim, typename Number = double>
  void
  map_mg_dofs_to_average_cell_size(const DoFHandler<dim, spacedim> &dof_handler,
                                   const unsigned int               level,
                                   std::vector<Number> &dof_average_cell_size)
  {
    const unsigned int n_dofs = dof_handler.n_dofs(level);
    AssertDimension(n_dofs, dof_average_cell_size.size());

    /**
     * Create the vector which stores the number of cells that share a common
     * DoF for each DoF.
     */
    std::vector<unsigned int> number_of_cells_sharing_dof(n_dofs, 0);

    for (const auto &cell : dof_handler.mg_cell_iterators_on_level(level))
      {
        /**
         * Get the diameter of the current cell.
         */
        Number cell_diameter = cell->diameter();

        /**
         * Get DoF indices local to this cell.
         */
        std::vector<types::global_dof_index> dof_indices(
          cell->get_fe().dofs_per_cell);
        cell->get_mg_dof_indices(dof_indices);

        for (const auto index : dof_indices)
          {
            number_of_cells_sharing_dof.at(index) += 1;
            dof_average_cell_size.at(index) += cell_diameter;
          }
      }

    for (unsigned int i = 0; i < n_dofs; i++)
      {
        dof_average_cell_size.at(i) /= number_of_cells_sharing_dof.at(i);
      }
  }


  /**
   * Calculate the average cell sizes associated with a subset of DoFs
   * selected from all those DoFs in the DoF handler.
   *
   * \mynote{The doubled cell size will be used as an estimate for the
   * diameter of the support set of each DoF.}
   *
   * @pre
   * @post
   * @tparam dim
   * @tparam spacedim
   * @tparam Number
   * @param dof_handler DoF handler object.
   * @param map_from_local_to_full_dof_indices This is a vector which stores
   * a list of global DoF indices corresponding to the DoFs held in the
   * @p dof_handler, while the element indices of this vector play the role
   * of local DoF indices corresponding to the selected DoFs. Therefore, this
   * vector is used as a map from local to global DoF indices.
   * @param dof_average_cell_size The returned list of average cell sizes
   * which corresponds to the selected DoFs. The memory for this vector should
   * be preallocated before calling this function.
   */
  template <int dim, int spacedim, typename Number = double>
  void
  map_dofs_to_average_cell_size(const DoFHandler<dim, spacedim> &dof_handler,
                                const std::vector<types::global_dof_index>
                                  &map_from_local_to_full_dof_indices,
                                std::vector<Number> &dof_average_cell_size)
  {
    const types::global_dof_index n_dofs =
      map_from_local_to_full_dof_indices.size();
    AssertDimension(n_dofs, dof_average_cell_size.size());

    std::vector<Number> full_dof_average_cell_size(dof_handler.n_dofs());
    map_dofs_to_average_cell_size(dof_handler, full_dof_average_cell_size);

    for (types::global_dof_index d = 0; d < n_dofs; d++)
      {
        dof_average_cell_size[d] =
          full_dof_average_cell_size[map_from_local_to_full_dof_indices[d]];
      }
  }


  /**
   * @brief Calculate the average cell sizes associated with a subset of DoFs
   * selected from all those DoFs on a specific level in the DoF handler.
   *
   * @tparam Number
   * @tparam dim
   * @tparam spacedim
   * @param dof_handler
   * @param map_from_local_to_full_dof_indices
   * @param level
   * @param dof_average_cell_size
   */
  template <int dim, int spacedim, typename Number = double>
  void
  map_mg_dofs_to_average_cell_size(const DoFHandler<dim, spacedim> &dof_handler,
                                   const std::vector<types::global_dof_index>
                                     &map_from_local_to_full_dof_indices,
                                   const unsigned int   level,
                                   std::vector<Number> &dof_average_cell_size)
  {
    const types::global_dof_index n_dofs =
      map_from_local_to_full_dof_indices.size();
    AssertDimension(n_dofs, dof_average_cell_size.size());

    std::vector<Number> full_dof_average_cell_size(dof_handler.n_dofs(level));
    map_mg_dofs_to_average_cell_size(dof_handler,
                                     level,
                                     full_dof_average_cell_size);

    for (types::global_dof_index d = 0; d < n_dofs; d++)
      {
        dof_average_cell_size[d] =
          full_dof_average_cell_size[map_from_local_to_full_dof_indices[d]];
      }
  }


  /**
   * Calculate the maximum cell sizes associated with those DoFs in the given
   * DoF handler object.
   *
   * The value doubled is used as an estimate for the diameter of the support
   * set of each DoF.
   *
   * @param dof_handler
   * @param dof_max_cell_size The returned list of maximum cell sizes. The
   * memory for this vector should be preallocated and initialized to zero
   * before calling this function.
   */
  template <typename DoFHandlerType, typename Number = double>
  void
  map_dofs_to_max_cell_size(const DoFHandlerType &dof_handler,
                            std::vector<Number>  &dof_max_cell_size)
  {
    const unsigned int n_dofs = dof_handler.n_dofs();
    AssertDimension(n_dofs, dof_max_cell_size.size());

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        /**
         * Get the diameter of the current cell.
         */
        Number cell_diameter = cell->diameter();

        /**
         * Get DoF indices local to this cell.
         */
        std::vector<types::global_dof_index> dof_indices(
          cell->get_fe().dofs_per_cell);
        cell->get_dof_indices(dof_indices);

        for (auto index : dof_indices)
          {
            if (cell_diameter > dof_max_cell_size.at(index))
              {
                dof_max_cell_size.at(index) = cell_diameter;
              }
          }
      }
  }


  /**
   * Calculate the maximum cell sizes associated with those DoFs on a specific
   * level in the given DoF handler object.
   */
  template <typename DoFHandlerType, typename Number = double>
  void
  map_mg_dofs_to_max_cell_size(const DoFHandlerType &dof_handler,
                               const unsigned int    level,
                               std::vector<Number>  &dof_max_cell_size)
  {
    const unsigned int n_dofs = dof_handler.n_dofs(level);
    AssertDimension(n_dofs, dof_max_cell_size.size());

    for (const auto &cell : dof_handler.mg_cell_iterators_on_level(level))
      {
        /**
         * Get the diameter of the current cell.
         */
        Number cell_diameter = cell->diameter();

        /**
         * Get DoF indices local to this cell.
         */
        std::vector<types::global_dof_index> dof_indices(
          cell->get_fe().dofs_per_cell);
        cell->get_mg_dof_indices(dof_indices);

        for (const auto index : dof_indices)
          {
            if (cell_diameter > dof_max_cell_size.at(index))
              {
                dof_max_cell_size.at(index) = cell_diameter;
              }
          }
      }
  }


  /**
   * Calculate the maximum cell sizes associated with a subset of DoFs
   * selected from all those DoFs in the DoF handler.
   */
  template <int dim, int spacedim, typename Number = double>
  void
  map_dofs_to_max_cell_size(const DoFHandler<dim, spacedim> &dof_handler,
                            const std::vector<types::global_dof_index>
                              &map_from_local_to_full_dof_indices,
                            std::vector<Number> &dof_max_cell_size)
  {
    const types::global_dof_index n_dofs =
      map_from_local_to_full_dof_indices.size();
    AssertDimension(n_dofs, dof_max_cell_size.size());

    std::vector<Number> full_dof_max_cell_size(dof_handler.n_dofs());
    map_dofs_to_max_cell_size(dof_handler, full_dof_max_cell_size);

    for (types::global_dof_index d = 0; d < n_dofs; d++)
      {
        dof_max_cell_size[d] =
          full_dof_max_cell_size[map_from_local_to_full_dof_indices[d]];
      }
  }


  /**
   * @brief Calculate the maximum cell sizes associated with a subset of DoFs
   * selected from all those DoFs on a specific level in the DoF handler.
   *
   * @tparam Number
   * @tparam dim
   * @tparam spacedim
   * @param dof_handler
   * @param map_from_local_to_full_dof_indices
   * @param level
   * @param dof_max_cell_size
   */
  template <int dim, int spacedim, typename Number = double>
  void
  map_mg_dofs_to_max_cell_size(const DoFHandler<dim, spacedim> &dof_handler,
                               const std::vector<types::global_dof_index>
                                 &map_from_local_to_full_dof_indices,
                               const unsigned int   level,
                               std::vector<Number> &dof_max_cell_size)
  {
    const types::global_dof_index n_dofs =
      map_from_local_to_full_dof_indices.size();
    AssertDimension(n_dofs, dof_max_cell_size.size());

    std::vector<Number> full_dof_max_cell_size(dof_handler.n_dofs(level));
    map_mg_dofs_to_max_cell_size(dof_handler, level, full_dof_max_cell_size);

    for (types::global_dof_index d = 0; d < n_dofs; d++)
      {
        dof_max_cell_size[d] =
          full_dof_max_cell_size[map_from_local_to_full_dof_indices[d]];
      }
  }


  /**
   * Calculate the minimum cell sizes associated with those DoFs in the given
   * DoF handler object.
   *
   * The value doubled is used as an estimate for the diameter of the support
   * set of each DoF.
   *
   * @param dof_min_cell_size The returned list of average cell sizes. The
   * memory for this vector should be preallocated and initialized to zero
   * before calling this function.
   */
  template <typename DoFHandlerType, typename Number = double>
  void
  map_dofs_to_min_cell_size(const DoFHandlerType &dof_handler,
                            std::vector<Number>  &dof_min_cell_size)
  {
    const unsigned int n_dofs = dof_handler.n_dofs();
    AssertDimension(n_dofs, dof_min_cell_size.size());

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        /**
         * Get the diameter of the current cell.
         */
        Number cell_diameter = cell->diameter();

        /**
         * Get DoF indices local to this cell.
         */
        std::vector<types::global_dof_index> dof_indices(
          cell->get_fe().dofs_per_cell);
        cell->get_dof_indices(dof_indices);

        for (auto index : dof_indices)
          {
            if (cell_diameter < dof_min_cell_size.at(index) ||
                dof_min_cell_size.at(index) == 0)
              {
                dof_min_cell_size.at(index) = cell_diameter;
              }
          }
      }
  }


  /**
   * Calculate the minimum cell sizes associated with those DoFs on a specific
   * level in the given DoF handler object.
   */
  template <typename DoFHandlerType, typename Number = double>
  void
  map_mg_dofs_to_min_cell_size(const DoFHandlerType &dof_handler,
                               const unsigned int    level,
                               std::vector<Number>  &dof_min_cell_size)
  {
    const unsigned int n_dofs = dof_handler.n_dofs(level);
    AssertDimension(n_dofs, dof_min_cell_size.size());

    for (const auto &cell : dof_handler.mg_cell_iterators_on_level(level))
      {
        /**
         * Get the diameter of the current cell.
         */
        Number cell_diameter = cell->diameter();

        /**
         * Get DoF indices local to this cell.
         */
        std::vector<types::global_dof_index> dof_indices(
          cell->get_fe().dofs_per_cell);
        cell->get_mg_dof_indices(dof_indices);

        for (const auto index : dof_indices)
          {
            if (cell_diameter < dof_min_cell_size.at(index) ||
                dof_min_cell_size.at(index) == 0)
              {
                dof_min_cell_size.at(index) = cell_diameter;
              }
          }
      }
  }


  /**
   * @brief Calculate the minimum cell sizes associated with a subset of DoFs
   * selected from all those DoFs in the DoF handler.
   *
   * @tparam Number
   * @tparam dim
   * @tparam spacedim
   * @param dof_handler
   * @param map_from_local_to_full_dof_indices
   * @param dof_min_cell_size
   */
  template <int dim, int spacedim, typename Number = double>
  void
  map_dofs_to_min_cell_size(const DoFHandler<dim, spacedim> &dof_handler,
                            const std::vector<types::global_dof_index>
                              &map_from_local_to_full_dof_indices,
                            std::vector<Number> &dof_min_cell_size)
  {
    const types::global_dof_index n_dofs =
      map_from_local_to_full_dof_indices.size();
    AssertDimension(n_dofs, dof_min_cell_size.size());

    std::vector<Number> full_dof_min_cell_size(dof_handler.n_dofs());
    map_dofs_to_min_cell_size(dof_handler, full_dof_min_cell_size);

    for (types::global_dof_index d = 0; d < n_dofs; d++)
      {
        dof_min_cell_size[d] =
          full_dof_min_cell_size[map_from_local_to_full_dof_indices[d]];
      }
  }


  /**
   * @brief Calculate the minimum cell sizes associated with a subset of DoFs
   * selected from all those DoFs on a specific level in the DoF handler.
   *
   * @tparam Number
   * @tparam dim
   * @tparam spacedim
   * @param dof_handler
   * @param map_from_local_to_full_dof_indices
   * @param level
   * @param dof_min_cell_size
   */
  template <int dim, int spacedim, typename Number = double>
  void
  map_mg_dofs_to_min_cell_size(const DoFHandler<dim, spacedim> &dof_handler,
                               const std::vector<types::global_dof_index>
                                 &map_from_local_to_full_dof_indices,
                               const unsigned int   level,
                               std::vector<Number> &dof_min_cell_size)
  {
    const types::global_dof_index n_dofs =
      map_from_local_to_full_dof_indices.size();
    AssertDimension(n_dofs, dof_min_cell_size.size());

    std::vector<Number> full_dof_min_cell_size(dof_handler.n_dofs(level));
    map_mg_dofs_to_min_cell_size(dof_handler, level, full_dof_min_cell_size);

    for (types::global_dof_index d = 0; d < n_dofs; d++)
      {
        dof_min_cell_size[d] =
          full_dof_min_cell_size[map_from_local_to_full_dof_indices[d]];
      }
  }


  /**
   * @brief Print out support points for the list of selected DoFs, which are
   * used for visualization in Gnuplot.
   *
   * @tparam spacedim
   * @param out
   * @param support_points
   * @param selected_dofs
   * @param has_label
   */
  template <int spacedim>
  void
  write_gnuplot_dof_support_point_info(
    std::ostream                                             &out,
    const std::map<types::global_dof_index, Point<spacedim>> &support_points,
    const std::vector<bool>                                  &selected_dofs,
    const bool                                                has_label = true)
  {
    AssertDimension(support_points.size(), selected_dofs.size());

    AssertThrow(out, ExcIO());

    unsigned int counter = 0;
    for (const auto &p : support_points)
      {
        if (selected_dofs[counter])
          {
            if (has_label)
              {
                out << p.second << ' ' << '"' << p.first << "\"\n";
              }
            else
              {
                out << p.second << "\n";
              }
          }

        counter++;
      }

    out.flush();

    AssertThrow(out, ExcIO());
  }


  /**
   * @brief Copy the list of selected DoF values into the complete DoF value
   * list.
   *
   * @tparam Number
   * @param all_dof_values
   * @param selected_dof_values
   * @param map_from_local_to_full_dof_indices
   */
  template <typename Number>
  void
  extend_selected_dof_values_to_full_dofs(
    Vector<Number>       &all_dof_values,
    const Vector<Number> &selected_dof_values,
    const std::vector<types::global_dof_index>
      &map_from_local_to_full_dof_indices)
  {
    AssertDimension(selected_dof_values.size(),
                    map_from_local_to_full_dof_indices.size());

    for (types::global_dof_index i = 0; i < selected_dof_values.size(); i++)
      {
        all_dof_values(map_from_local_to_full_dof_indices[i]) =
          selected_dof_values(i);
      }
  }


  /**
   * Build the topology for "DoF support point-to-cell" relation.
   *
   * \mynote{2022-06-06 This topology is needed when the continuous finite
   * element such as @p FE_Q is adopted. For the discontinuous finite element
   * such as @p FE_DGQ, the DoFs in a cell are separated from those in other
   * cells. Hence, such point-to-cell topology is not necessary.}
   *
   * @param dof_to_cell_topo
   * @param dof_handler
   * @param fe_index
   */
  template <int dim, int spacedim>
  void
  build_dof_to_cell_topology(
    std::vector<
      std::vector<const typename DoFHandler<dim, spacedim>::cell_iterator *>>
      &dof_to_cell_topo,
    const std::vector<typename DoFHandler<dim, spacedim>::cell_iterator>
                                    &cell_iterators_in_dof_handler,
    const DoFHandler<dim, spacedim> &dof_handler,
    const unsigned int               fe_index = 0)
  {
    const types::global_dof_index        n_dofs = dof_handler.n_dofs();
    const FiniteElement<dim, spacedim>  &fe     = dof_handler.get_fe(fe_index);
    const unsigned int                   dofs_per_cell = fe.dofs_per_cell;
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    dof_to_cell_topo.resize(n_dofs);

    for (const auto &cell : cell_iterators_in_dof_handler)
      {
        cell->get_dof_indices(local_dof_indices);
        for (auto dof_index : local_dof_indices)
          {
            dof_to_cell_topo[dof_index].push_back(&cell);
          }
      }
  }


  /**
   * @brief Build the topology for "DoF support point-to-cell" relation.
   *
   * This function is similar to \ref
   * build_dof_to_cell_topology(std::vector<std::vector<const typename
   * DoFHandler<dim, spacedim>::cell_iterator *>> &, const
   * std::vector<typename DoFHandler<dim, spacedim>::cell_iterator> &, const
   * DoFHandler<dim, spacedim> &, const unsigned int), but the result is
   * returned in the object DoFToCellTopology, which also stores the maximum
   * number of cells associated with a DoF.
   *
   * @tparam dim
   * @tparam spacedim
   * @param dof_to_cell_topo
   * @param cell_iterators_in_dof_handler
   * @param dof_handler
   * @param fe_index
   */
  template <int dim, int spacedim>
  void
  build_dof_to_cell_topology(
    DoFToCellTopology<dim, spacedim> &dof_to_cell_topo,
    const std::vector<typename DoFHandler<dim, spacedim>::cell_iterator>
                                    &cell_iterators_in_dof_handler,
    const DoFHandler<dim, spacedim> &dof_handler,
    const unsigned int               fe_index = 0)
  {
    const types::global_dof_index        n_dofs = dof_handler.n_dofs();
    const FiniteElement<dim, spacedim>  &fe     = dof_handler.get_fe(fe_index);
    const unsigned int                   dofs_per_cell = fe.dofs_per_cell;
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    dof_to_cell_topo.topology.resize(n_dofs);
    dof_to_cell_topo.max_cells_per_dof = 0;

    for (const auto &cell : cell_iterators_in_dof_handler)
      {
        cell->get_dof_indices(local_dof_indices);
        for (auto dof_index : local_dof_indices)
          {
            dof_to_cell_topo.topology[dof_index].push_back(&cell);
          }
      }

    for (const auto &dof_to_cells : dof_to_cell_topo.topology)
      {
        if (dof_to_cells.size() > dof_to_cell_topo.max_cells_per_dof)
          {
            dof_to_cell_topo.max_cells_per_dof = dof_to_cells.size();
          }
      }
  }


  /**
   * @brief Build the topology for "DoF support point-to-cell" relation. The
   * DoFs are on a specific level in the multigrid.
   *
   * @tparam dim
   * @tparam spacedim
   * @param dof_to_cell_topo
   * @param mg_cell_iterators_in_dof_handler
   * @param dof_handler
   * @param level
   * @param fe_index
   */
  template <int dim, int spacedim>
  void
  build_mg_dof_to_cell_topology(
    std::vector<
      std::vector<const typename DoFHandler<dim, spacedim>::cell_iterator *>>
      &dof_to_cell_topo,
    const std::vector<typename DoFHandler<dim, spacedim>::cell_iterator>
                                    &mg_cell_iterators_in_dof_handler,
    const DoFHandler<dim, spacedim> &dof_handler,
    const unsigned int               level    = 0,
    const unsigned int               fe_index = 0)
  {
    const types::global_dof_index        n_dofs = dof_handler.n_dofs(level);
    const FiniteElement<dim, spacedim>  &fe     = dof_handler.get_fe(fe_index);
    const unsigned int                   dofs_per_cell = fe.dofs_per_cell;
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    dof_to_cell_topo.resize(n_dofs);

    for (const auto &cell : mg_cell_iterators_in_dof_handler)
      {
        cell->get_mg_dof_indices(local_dof_indices);
        for (auto dof_index : local_dof_indices)
          {
            dof_to_cell_topo[dof_index].push_back(&cell);
          }
      }
  }


  /**
   * @brief Build the topology for "DoF support point-to-cell" relation. The
   * DoFs are on a specific level in the multigrid.
   *
   * This function is similar to \ref
   * build_dof_to_cell_topology(std::vector<std::vector<const typename
   * DoFHandler<dim, spacedim>::cell_iterator *>> &, const
   * std::vector<typename DoFHandler<dim, spacedim>::cell_iterator> &, const
   * DoFHandler<dim, spacedim> &, const unsigned int, const unsigned int), but
   * the result is returned in the object DoFToCellTopology, which also stores
   * the maximum number of cells associated with a DoF.
   *
   * @tparam dim
   * @tparam spacedim
   * @param dof_to_cell_topo
   * @param mg_cell_iterators_in_dof_handler
   * @param dof_handler
   * @param level
   * @param fe_index
   */
  template <int dim, int spacedim>
  void
  build_mg_dof_to_cell_topology(
    DoFToCellTopology<dim, spacedim> &dof_to_cell_topo,
    const std::vector<typename DoFHandler<dim, spacedim>::cell_iterator>
                                    &mg_cell_iterators_in_dof_handler,
    const DoFHandler<dim, spacedim> &dof_handler,
    const unsigned int               level    = 0,
    const unsigned int               fe_index = 0)
  {
    const types::global_dof_index        n_dofs = dof_handler.n_dofs(level);
    const FiniteElement<dim, spacedim>  &fe     = dof_handler.get_fe(fe_index);
    const unsigned int                   dofs_per_cell = fe.dofs_per_cell;
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    dof_to_cell_topo.topology.resize(n_dofs);
    dof_to_cell_topo.max_cells_per_dof = 0;

    for (const auto &cell : mg_cell_iterators_in_dof_handler)
      {
        cell->get_mg_dof_indices(local_dof_indices);
        for (auto dof_index : local_dof_indices)
          {
            dof_to_cell_topo.topology[dof_index].push_back(&cell);
          }
      }

    for (const auto &dof_to_cells : dof_to_cell_topo.topology)
      {
        if (dof_to_cells.size() > dof_to_cell_topo.max_cells_per_dof)
          {
            dof_to_cell_topo.max_cells_per_dof = dof_to_cells.size();
          }
      }
  }


  /**
   * @brief Make a sparsity pattern where the row indices of the matrix should
   * be converted from external DoF numbering to internal.
   */
  template <int dim, int spacedim, typename SparsityPatternType>
  void
  make_sparsity_pattern(
    const DoFHandler<dim, spacedim>            &dof_row,
    const std::vector<types::global_dof_index> &dof_row_e2i_numbering,
    const DoFHandler<dim, spacedim>            &dof_col,
    SparsityPatternType                        &sparsity)
  {
    const types::global_dof_index n_dofs_row = dof_row.n_dofs();
    const types::global_dof_index n_dofs_col = dof_col.n_dofs();
    (void)n_dofs_row;
    (void)n_dofs_col;

    Assert(sparsity.n_rows() == n_dofs_row,
           ExcDimensionMismatch(sparsity.n_rows(), n_dofs_row));
    Assert(sparsity.n_cols() == n_dofs_col,
           ExcDimensionMismatch(sparsity.n_cols(), n_dofs_col));

    using cell_iterator = typename DoFHandler<dim, spacedim>::cell_iterator;
    std::list<std::pair<cell_iterator, cell_iterator>> cell_list =
      GridTools::get_finest_common_cells(dof_row, dof_col);

    for (const auto &cell_pair : cell_list)
      {
        const cell_iterator cell_row = cell_pair.first;
        const cell_iterator cell_col = cell_pair.second;

        if (cell_row->is_active() && cell_col->is_active())
          {
            const unsigned int dofs_per_cell_row =
              cell_row->get_fe().n_dofs_per_cell();
            const unsigned int dofs_per_cell_col =
              cell_col->get_fe().n_dofs_per_cell();
            std::vector<types::global_dof_index> local_dof_indices_row(
              dofs_per_cell_row);
            std::vector<types::global_dof_index> local_dof_indices_col(
              dofs_per_cell_col);
            cell_row->get_dof_indices(local_dof_indices_row);
            cell_col->get_dof_indices(local_dof_indices_col);
            for (unsigned int i = 0; i < dofs_per_cell_row; ++i)
              sparsity.add_entries(
                dof_row_e2i_numbering[local_dof_indices_row[i]],
                local_dof_indices_col.begin(),
                local_dof_indices_col.end());
          }
      }
  }
} // namespace DoFToolsExt

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_DOF_TOOLS_EXT_H_