// Copyright (C) 2022-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file triangulation_tools.h
 * @brief Introduction of triangulation_tools.h
 *
 * @date 2022-11-17
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_GRID_TRIANGULATION_TOOLS_H_
#define HIERBEM_INCLUDE_GRID_TRIANGULATION_TOOLS_H_

#include <deal.II/grid/tria.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <set>
#include <vector>

#include "config.h"

HBEM_NS_OPEN

using namespace dealii;

/**
 * Build the vertex-to-cell topology.
 *
 * @param vertex_to_cell_topo
 * @param triangulation
 */
template <int dim, int spacedim>
void
build_vertex_to_cell_topology(
  std::vector<std::vector<unsigned int>> &vertex_to_cell_topo,
  const Triangulation<dim, spacedim>     &triangulation)
{
  vertex_to_cell_topo.resize(triangulation.n_vertices());

  // Iterate over each cell to build the topology.
  for (const auto &cell : triangulation.active_cell_iterators())
    {
      for (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell; v++)
        {
          vertex_to_cell_topo[cell->vertex_index(v)].push_back(
            cell->active_cell_index());
        }
    }
}


/**
 * Print out the topological information about vertex-to-cell
 * relation.
 *
 * @param vertex_to_cell_topo
 */
template <typename IndexType>
void
print_vertex_to_cell_topology(
  std::ostream                              &out,
  const std::vector<std::vector<IndexType>> &vertex_to_cell_topo)
{
  unsigned int counter = 0;
  for (const auto &vertex_to_cell : vertex_to_cell_topo)
    {
      out << counter << ": ";
      for (auto cell_index : vertex_to_cell)
        {
          out << cell_index << " ";
        }
      out << std::endl;
      counter++;
    }
}


/**
 * Mark the interfacial cells between two collection of material domains.
 *
 * \mynote{For handling mixed boundary value problem in BEM, the first domain
 * collection is assigned with Dirichlet boundary condition, while the second
 * domain collection is assigned with Neumann boundary condition. Those cells
 * neighboring the Dirichlet domain and lying within the Neumann domain will
 * be marked with its original material id plus a large shift.}
 *
 * \myalert{After extracting the surface mesh from volume mesh, the original
 * boundary id becomes material id.}
 *
 * @param triangulation
 * @param vertex_to_cell_topo
 * @param material_domain1
 * @param material_domain2
 */
template <int dim, int spacedim>
void
mark_interfacial_cells_between_materials(
  Triangulation<dim, spacedim> &surface_triangulation,
  const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                 typename Triangulation<dim + 1, spacedim>::face_iterator>
                                               &map_from_surface_to_volume_mesh,
  const std::vector<std::vector<unsigned int>> &vertex_to_cell_topo,
  const std::set<types::material_id>           &domain_collection1,
  const std::set<types::material_id>           &domain_collection2,
  std::set<types::material_id>                 &interfacial_domain_material_ids,
  const types::material_id                      material_id_shift = 10000)
{
  interfacial_domain_material_ids.clear();

  // Get the starting cell iterator.
  auto cell_iter = surface_triangulation.begin_active();

  // Iterate over each vertex in the vertex-to-cell topology.
  for (const auto &associated_cell_indices : vertex_to_cell_topo)
    {
      bool is_domain_collection1_hit = false;
      bool is_domain_collection2_hit = false;

      // Vector of flags indicating which domain collection the cells belong
      // to.
      std::vector<unsigned int> collection_flags(
        associated_cell_indices.size());

      // Iterate over each cell related to the current vertex and label it to
      // which domain collection it belongs.
      unsigned int counter = 0;
      for (auto cell_index : associated_cell_indices)
        {
          cell_iter = surface_triangulation.begin_active();
          std::advance(cell_iter, cell_index);
          // Get the material id.
          types::material_id mat_id = cell_iter->material_id();

          // Test if the current cell belongs to the first domain collection.
          auto found_pos = domain_collection1.find(
            mat_id >= material_id_shift ? mat_id - material_id_shift : mat_id);
          if (found_pos != domain_collection1.cend())
            {
              // The current cell belongs to the first domain collection.
              is_domain_collection1_hit = true;

              collection_flags[counter] = 1;
            }
          else
            {
              // Test if the current cell belongs to the second domain
              // collection.
              auto found_pos = domain_collection2.find(
                mat_id >= material_id_shift ? mat_id - material_id_shift :
                                              mat_id);
              if (found_pos != domain_collection2.cend())
                {
                  // The current cell belongs to the second domain collection.
                  is_domain_collection2_hit = true;

                  collection_flags[counter] = 2;
                }
              else
                {
                  // This case should not appear, since the cell should at
                  // least belong to one of the domain collections.
                  Assert(false, ExcInternalError());
                }
            }

          counter++;
        }

      if (is_domain_collection1_hit && is_domain_collection2_hit)
        {
          // The current vertex is found to be shared by cells from two domain
          // collections. Then iterate over each cell belonging to the second
          // domain collection and shift their material ids.
          for (unsigned int c = 0; c < associated_cell_indices.size(); c++)
            {
              if (collection_flags[c] == 2)
                {
                  cell_iter = surface_triangulation.begin_active();
                  std::advance(cell_iter, associated_cell_indices[c]);
                  types::material_id mat_id = cell_iter->material_id();

                  if (mat_id < material_id_shift)
                    {
                      // Set the material id of the current cell in the
                      // surface mesh.
                      cell_iter->set_material_id(mat_id + material_id_shift);

                      // Set the boundary id of the corresponding original
                      // face in the volume mesh.
                      auto face_iter_pos =
                        map_from_surface_to_volume_mesh.find(cell_iter);
                      Assert(
                        face_iter_pos != map_from_surface_to_volume_mesh.cend(),
                        ExcMessage(
                          "The associated face iterator in the volume triangulation cannot be found for the cell iterator in the boundary mesh!"));

                      auto face_iter = face_iter_pos->second;
                      face_iter->set_boundary_id(mat_id + material_id_shift);

                      // Insert the new id into the interfacial domain set.
                      interfacial_domain_material_ids.insert(mat_id +
                                                             material_id_shift);
                    }
                }
            }
        }
    }
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_GRID_TRIANGULATION_TOOLS_H_
