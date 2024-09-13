/**
 * @file dof_tools_ext.h
 * @brief Introduction of dof_tools_ext.h
 *
 * @date 2022-11-16
 * @author Jihuan Tian
 */
#ifndef INCLUDE_DOF_TOOLS_EXT_H_
#define INCLUDE_DOF_TOOLS_EXT_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/geometry_info.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <algorithm>
#include <map>
#include <vector>

#include "gmsh_manipulation.h"

using namespace dealii;

namespace HierBEM
{
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
    extract_material_domain_dofs(
      const DoFHandler<dim, spacedim>    &dof_handler,
      const std::set<types::material_id> &material_ids,
      std::vector<bool>                  &selected_dofs,
      const bool                          reset_selectors_to_false = true)
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
          auto found_iter =
            boundary_cell_material_ids.find(cell->material_id());

          if (found_iter != boundary_cell_material_ids.end())
            {
              // Iterate over each face, which is actually a line in
              // this case, and check if it completely lies at boundary.
              for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell;
                   f++)
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
     * Calculate the average cell sizes associated with those DoFs handled by
     * the given DoF handler object.
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

          for (auto index : dof_indices)
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
     * selected from all those DoFs held within the DoF handler.
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
     * Calculate the maximum cell sizes associated with those DoFs handled by
     * the given DoF handler object.
     *
     * The value doubled is used as an estimate for the diameter of the support
     * set of each DoF.
     *
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
     * Calculate the minimum cell sizes associated with those DoFs handled by
     * the given DoF handler object.
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


    template <int spacedim>
    void
    write_gnuplot_dof_support_point_info(
      std::ostream                                             &out,
      const std::map<types::global_dof_index, Point<spacedim>> &support_points,
      const std::vector<bool>                                  &selected_dofs,
      const bool has_label = true)
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
  } // namespace DoFToolsExt
} // namespace HierBEM
#endif /* INCLUDE_DOF_TOOLS_EXT_H_ */
