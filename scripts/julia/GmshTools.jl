using Meshes, Gmsh

function read_msh(file_name)
    Gmsh.initialize()

    mesh = gmsh.open(file_name)

    # Extract all vertices.
    nodeTags, nodes = gmsh.model.mesh.get_nodes()
    node_num = length(nodeTags)
    spacedim = 3
    nodes = transpose(reshape(nodes, (spacedim, node_num)))

    # N.B. The nodes and their tags directly read via Gmsh API may not be
    # ordered. We need to reorder them.
    perm_indices = sortperm(nodeTags)
    nodeTags = nodeTags[perm_indices]
    nodes = nodes[perm_indices, :]

    # Extract surface mesh, where @p element_node_indices stores the node indices for all cells,
    # the value type of which is UInt64.
    elementTypes, elementTags, element_node_indices = gmsh.model.mesh.get_elements(2)
    # Now convert the node indices 
    cell_num = length(elementTags[1])
    node_num_per_cell = 4
    cells = transpose(reshape(Int64.(element_node_indices[1]), (node_num_per_cell, cell_num)))

    Gmsh.finalize()

    # Connect indices into a quadrangles for visualization in Meshes.
    connectivity = connect.([(cells[i, 1], cells[i, 2], cells[i, 3], cells[i, 4]) for i = 1:cell_num], Quadrangle)
    points = Meshes.Point.(nodes[:, 1], nodes[:, 2], nodes[:, 3])
    mesh_viz = SimpleMesh(points, connectivity);

    return points, connectivity, mesh_viz
end
