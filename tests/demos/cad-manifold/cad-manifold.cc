/**
 * @file cad-manifold.cc
 * @brief Test for reading STEP file and generating mesh
 *
 * @ingroup testers
 * @date 2025-02-25
 */

#include <deal.II/base/point.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_in_ext.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/opencascade/manifold_lib.h>
#include <deal.II/opencascade/utilities.h>

#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <catch2/catch_all.hpp>

#include <fstream>
#include <iostream>

using namespace Catch::Matchers;

TEST_CASE("Extract TopoDS_Shape from CAD file", "[cad][demo]")
{
  // Read STEP file using deal.II interface
  TopoDS_Shape shape =
    dealii::OpenCASCADE::read_STEP(SOURCE_DIR "/test_model.stp");
  REQUIRE(!shape.IsNull());

  // Count solids in the shape
  int solid_count = 0;
  for (TopExp_Explorer exp(shape, TopAbs_SOLID); exp.More(); exp.Next())
    {
      solid_count++;
    }

  std::cout << "Number of solids: " << solid_count << std::endl;
  REQUIRE(solid_count == 2);
}

TEST_CASE("Generate mesh from STEP file", "[cad][demo][mesh]")
{
  // Read STEP file using deal.II interface
  TopoDS_Shape shape =
    dealii::OpenCASCADE::read_STEP(SOURCE_DIR "/test_model.stp");
  REQUIRE(!shape.IsNull());

  try
    {
      // 创建表面网格
      dealii::Triangulation<2, 3> triangulation;

      // 读取网格
      dealii::GridIn<2, 3> grid_in;
      grid_in.attach_triangulation(triangulation);

      try
        {
          std::ifstream mesh_in(SOURCE_DIR "/test_model.msh");
          read_msh(mesh_in, triangulation);

          // 应用CAD流形
          dealii::OpenCASCADE::ArclengthProjectionLineManifold<2, 3>
            line_manifold(shape);
          dealii::OpenCASCADE::NormalToMeshProjectionManifold<2, 3>
            surface_manifold(shape);

          triangulation.set_manifold(1, line_manifold);
          triangulation.set_manifold(2, surface_manifold);

          // 输出原始网格
          std::ofstream   out("test_model.vtk");
          dealii::GridOut grid_out;
          grid_out.write_vtk(triangulation, out);

          // 细化网格
          triangulation.refine_global(1);

          // 输出细化后的网格
          std::ofstream out_refined("test_model_refined.vtk");
          grid_out.write_vtk(triangulation, out_refined);
        }
      catch (const std::exception &e)
        {
          std::cerr << "Failed to read mesh: " << e.what() << std::endl;
          REQUIRE(false);
        }
    }
  catch (const std::exception &e)
    {
      std::cerr << "Exception: " << e.what() << std::endl;
      REQUIRE(false);
    }
}
