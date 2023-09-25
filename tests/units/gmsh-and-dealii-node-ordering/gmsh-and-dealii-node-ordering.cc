#include <deal.II/base/logstream.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>

#include <fstream>

using namespace dealii;

int
main(int argc, char *argv[])
{
  (void)argc;

  deallog.depth_console(2);
  deallog.pop();

  std::string         mesh_file_name(argv[1]);
  Triangulation<2, 3> triangulation;
  GridIn<2, 3>        grid_in;
  grid_in.attach_triangulation(triangulation);
  std::fstream mesh_file(mesh_file_name);
  grid_in.read_msh(mesh_file);

  unsigned int cell_counter = 0;
  deallog << "Cell_index Vertex_index X Y Z" << std::endl;
  for (const auto &cell : triangulation.active_cell_iterators())
    {
      for (unsigned int i = 0; i < GeometryInfo<2>::vertices_per_cell; i++)
        {
          deallog << cell_counter << " " << cell->vertex_index(i) << " "
                  << cell->vertex(i) << " " << std::endl;
        }

      cell_counter++;
    }
}
