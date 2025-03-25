/**
 * @file cad-manifold.cc
 * @brief Test for reading STEP file and generating mesh
 *
 * @ingroup testers
 * @date 2025-02-25
 */

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/opencascade/manifold_lib.h>
#include <deal.II/opencascade/utilities.h>

#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <catch2/catch_all.hpp>
#include <fmt/core.h>

#include <fstream>

#include "grid_in_ext.h"
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

  fmt::print("Number of solids: {}\n", solid_count);
}

TEST_CASE("Generate mesh from STEP file", "[cad][demo][mesh]")
{
  // Read STEP file
  TopoDS_Shape shape =
    dealii::OpenCASCADE::read_STEP(SOURCE_DIR "/test_model.stp");
  REQUIRE(!shape.IsNull());

  dealii::Triangulation<2, 3> tria;
  dealii::GridOut             grid_out;

  try
    {
      // Read mesh file
      HierBEM::read_msh(SOURCE_DIR "/test_model.msh", tria);

      fmt::print("Original mesh has {} cells\n", tria.n_active_cells());
      REQUIRE(tria.n_active_cells() > 0);

      // Output original mesh
      std::ofstream out_original("test_model_original.vtk");
      grid_out.write_vtk(tria, out_original);
      fmt::print("Simple refined mesh has {} cells\n", tria.n_active_cells());

      // Refine mesh without manifold information
      tria.refine_global(1);
      std::ofstream out_refined_no_manifold(
        "test_model_refined_no_manifold.vtk");
      grid_out.write_vtk(tria, out_refined_no_manifold);

      // Reset triangulation
      tria.clear();

      // Reread mesh file
      HierBEM::read_msh(SOURCE_DIR "/test_model.msh", tria);

      // Add CAD manifold
      dealii::OpenCASCADE::NormalToMeshProjectionManifold<2, 3> manifold(shape);
      tria.set_manifold(0, manifold);

      // Iterating all active cells and set their manifold_ids
      for (const auto &cell : tria.active_cell_iterators())
        {
          cell->set_manifold_id(0);
        }

      // Refine mesh with manifold information
      tria.refine_global(1);
      std::ofstream out_refined_with_manifold(
        "test_model_refined_with_manifold.vtk");
      grid_out.write_vtk(tria, out_refined_with_manifold);
    }
  catch (const std::exception &e)
    {
      fmt::print("Exception: {}\n", e.what());
      REQUIRE(false);
    }
}
