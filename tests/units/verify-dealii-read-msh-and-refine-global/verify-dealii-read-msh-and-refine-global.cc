/**
 * @file verify-dealii-read-msh-and-refine-global.cc
 * @brief
 *
 * @ingroup testers
 * @author
 * @date 2024-12-09
 */

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <boost/program_options.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>

#include "grid_in_ext.h"
#include "grid_out_ext.h"
#include "hbem_test_config.h"

using namespace HierBEM;
using namespace dealii;
using namespace std;
namespace po = boost::program_options;

enum ReadMSHFunction
{
  DEAL_II_FILENAME,
  DEAL_II_FILESTREAM,
  HIERBEM_FILENAME,
  HIERBEM_FILESTREAM,
  HIERBEM_FILESTREAM_WITH_CORRECT_OPTIONS // i.e.
                                          // read_lines_as_subcelldata=false,
                                          // reorder_cell_vertices=true,
                                          // check_cell_orientation=false
};

struct CmdOpts
{
  ReadMSHFunction func;
  std::string     mesh_file;

  // This option is only used for @p HIERBEM_FILESTREAM.
  bool read_lines_as_subcelldata;
  // This option is only used for @p HIERBEM_FILESTREAM.
  bool reorder_cell_vertices;
  // This option is used for both @p HIERBEM_FILESTREAM
  // and @p HIERBEM_FILENAME.
  bool check_cell_orientation;
};

CmdOpts
parse_cmdline(int argc, const char *argv[])
{
  CmdOpts                 opts;
  po::options_description desc("Allowed options");

  // clang-format off
  desc.add_options()
    ("help,h", "show help message")
    ("function,f", po::value<std::string>()->default_value("hierbem-filestream-with-correct-options"), "Function for reading MSH file")
    ("mesh-file,m", po::value<std::string>()->default_value(HBEM_TEST_MODEL_DIR "sphere2d-quasi-structured-quad.msh"), "Mesh file")
    ("read-lines,l", po::bool_switch(&opts.read_lines_as_subcelldata), "Read lines as subcelldata")
    ("reorder-vertices,v", po::bool_switch(&opts.reorder_cell_vertices), "Reorder cell vertices to make lines aligned")
    ("check-orient,o", po::bool_switch(&opts.check_cell_orientation), "Check cell orientation in manifold");
  // clang-format on

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help"))
    {
      std::cout << desc << std::endl;
      std::exit(EXIT_SUCCESS);
    }

  const std::string func = vm["function"].as<std::string>();
  if (func == "hierbem-filename")
    {
      opts.func = ReadMSHFunction::HIERBEM_FILENAME;
    }
  else if (func == "hierbem-filestream")
    {
      opts.func = ReadMSHFunction::HIERBEM_FILESTREAM;
    }
  else if (func == "hierbem-filestream-with-correct-options")
    {
      opts.func = ReadMSHFunction::HIERBEM_FILESTREAM_WITH_CORRECT_OPTIONS;
    }
  else if (func == "dealii-filename")
    {
      opts.func = ReadMSHFunction::DEAL_II_FILENAME;
    }
  else if (func == "dealii-filestream")
    {
      opts.func = ReadMSHFunction::DEAL_II_FILESTREAM;
    }
  else
    {
      std::cerr << "Invalid function specified for reading MSH file: " << func
                << std::endl;
      std::exit(EXIT_FAILURE);
    }

  if (vm.count("mesh-file"))
    {
      opts.mesh_file = vm["mesh-file"].as<std::string>();
    }
  else
    {
      std::cerr << "Please specify the mesh file to be read!" << std::endl;
      std::exit(EXIT_FAILURE);
    }

  // Check the existence of the mesh file.
  if (!std::filesystem::exists(opts.mesh_file))
    {
      std::cerr << "Mesh file " << opts.mesh_file << " does not exist!"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }

  return opts;
}

int
main(int argc, const char *argv[])
{
  CmdOpts opts = parse_cmdline(argc, argv);

  Triangulation<2, 3> tria(
    Triangulation<2, 3>::MeshSmoothing::limit_level_difference_at_vertices);

  switch (opts.func)
    {
        case ReadMSHFunction::DEAL_II_FILENAME: {
          GridIn<2, 3> grid_in;
          grid_in.attach_triangulation(tria);
          grid_in.read_msh(opts.mesh_file);

          break;
        }
        case ReadMSHFunction::DEAL_II_FILESTREAM: {
          GridIn<2, 3> grid_in;
          grid_in.attach_triangulation(tria);
          ifstream in(opts.mesh_file);
          grid_in.read_msh(in);

          break;
        }
        case ReadMSHFunction::HIERBEM_FILENAME: {
          read_msh(opts.mesh_file, tria, opts.check_cell_orientation);

          break;
        }
        case ReadMSHFunction::HIERBEM_FILESTREAM: {
          ifstream in(opts.mesh_file);
          read_msh(in,
                   tria,
                   opts.read_lines_as_subcelldata,
                   opts.reorder_cell_vertices,
                   opts.check_cell_orientation);

          break;
        }
        case ReadMSHFunction::HIERBEM_FILESTREAM_WITH_CORRECT_OPTIONS: {
          ifstream in(opts.mesh_file);
          read_msh(in, tria);

          break;
        }
        default: {
          AssertThrow(false, ExcInternalError());
        }
    }

  // Rewrite the original mesh to see if deal.ii has modified the node order.
  ofstream out("original-export.msh");
  write_msh_correct(tria, out);
  out.close();

  tria.refine_global();

  out.open("refined.msh");
  write_msh_correct(tria, out);

  return 0;
}
