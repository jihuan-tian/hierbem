using Colors
using Meshes, Gmsh, GLMakie
using CSV, DataFrames

# Load the data.
Cd = CSV.read("Cd.dat", DataFrame; delim=' ', header=false, ignorerepeated=true)
support_points_in_dual_mesh = CSV.read("support-points-in-dual-space-on-dual-mesh.dat", DataFrame; delim=' ', header=false)
support_points_in_refined_mesh = CSV.read("support-points-in-dual-space-on-refined-mesh.dat", DataFrame; delim=' ', header=false)

# Read the mesh
Gmsh.initialize()

mesh = gmsh.open("refined-mesh.msh")

# Extract all vertices.
nodeTags, nodes = gmsh.model.mesh.get_nodes()
node_num = length(nodeTags)
spacedim = 3
nodes = transpose(reshape(nodes, (spacedim, node_num)))

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

fig = Figure(size = (1920,1080))
ax = Axis3(fig[1, 1], azimuth=-π / 2, elevation=π / 2)
viz!(ax, mesh_viz, showsegments=true, color=:burlywood)
display(fig)

scatter!(ax, Matrix(support_points_in_dual_mesh[:, 2:4]), color=:red, markersize=10)
text!(ax, Matrix(support_points_in_dual_mesh[:, 2:4]), text=string.(support_points_in_dual_mesh[:,1]), align=(:left, :bottom), fontsize=20, color=:black)

scatter!(ax, Matrix(support_points_in_refined_mesh[:, 2:4]), color=:green, markersize=10)
text!(ax, Matrix(support_points_in_refined_mesh[:, 2:4]), text=string.(support_points_in_refined_mesh[:,1]), align=(:left, :bottom), fontsize=15, color=:blue)

# Generate a dictionary for the support points in dual mesh.
support_points_in_dual_mesh_dict = Dict(row[1] => row[2:4] for row in eachrow(support_points_in_dual_mesh))

# Generate text strings for DoFs in the refined mesh associated with each DoF in the dual mesh.
for m = 1:size(Cd)[1]
    associated_dof_index_string = ""
    for n = 1:size(Cd)[2]
        if (Cd[m,n] > 0)
            associated_dof_index_string = associated_dof_index_string * "(#" * string(n-1) * "," * string(Cd[m,n]) * ")"
        end
        text!(ax, support_points_in_dual_mesh_dict[m-1][1], support_points_in_dual_mesh_dict[m-1][2], Float64(support_points_in_dual_mesh_dict[m-1][3]), text=associated_dof_index_string, align=(:left, :top), fontsize=15, color=:black)
    end
end

GLMakie.save("verify-Cd.png", fig)
