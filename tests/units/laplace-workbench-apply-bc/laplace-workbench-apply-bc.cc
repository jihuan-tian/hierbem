// Copyright (C) 2026 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file laplace-workbench-apply-bc.cc
 * @brief Apply distinct Dirichlet or Neumann data to the two-sphere model and
 * check the interpolated boundary values.
 *
 * The configuration files are the same kind consumed by the Laplace workbench.
 * Boundary functions are created with @p create_function_from_value_spec and
 * assigned through @p LaplaceBEM, which is what @p setupBoundaryConds does.
 * Each sphere keeps its own constant value or muparser expression. The
 * interpolated vector is written to VTK and to a vector file.
 *
 * @author Jihuan Tian
 * @date 2026-09-23
 */

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/manifold_lib.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_interpolate.templates.h>

#include <catch2/catch_all.hpp>
#include <rfl.hpp>
#include <rfl/toml.hpp>

#include <complex>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "config_file/config_file.h"
#include "config_file/config_structs.h"
#include "grid/grid_in_ext.h"
#include "hbem_test_config.h"
#include "laplace/laplace_bem.h"
#include "postprocessing/data_out_ext.h"
#include "utilities/debug_tools.h"

using namespace dealii;
using namespace HierBEM;
using namespace Catch::Matchers;

namespace
{
  template <typename Number>
  void
  check_boundary_value(const Number &actual,
                       const Number &expected,
                       const double  abs_error = 1e-15,
                       const double  rel_error = 1e-15)
  {
    if constexpr (std::is_same_v<Number, std::complex<double>>)
      {
        REQUIRE_THAT(actual.real(),
                     WithinAbs(expected.real(), abs_error) ||
                       WithinRel(expected.real(), rel_error));
        REQUIRE_THAT(actual.imag(),
                     WithinAbs(expected.imag(), abs_error) ||
                       WithinRel(expected.imag(), rel_error));
      }
    else
      {
        REQUIRE_THAT(actual,
                     WithinAbs(expected, abs_error) ||
                       WithinRel(expected, rel_error));
      }
  }

  ProblemType
  problemTypeLiteralToEnum(const ProblemTypeLiteral &literal)
  {
    switch (literal.value())
      {
        case ProblemTypeLiteral::value_of<"neumann">():
          return ProblemType::NeumannBCProblem;
        case ProblemTypeLiteral::value_of<"dirichlet">():
          return ProblemType::DirichletBCProblem;
        case ProblemTypeLiteral::value_of<"mixed">():
          return ProblemType::MixedBCProblem;
        case ProblemTypeLiteral::value_of<"robin">():
          return ProblemType::RobinBCProblem;
        default:
          throw std::runtime_error("Unknown problem type");
      }
  }

  /**
   * Load one workbench configuration, assign boundary conditions on each
   * sphere, interpolate it with the specified mapping order, and write the
   * field.
   */
  template <typename Number>
  void
  run_two_sphere_bc_case(const std::string &config_path,
                         const bool         is_dirichlet)
  {
    constexpr bool solver_is_complex =
      std::is_same_v<Number, std::complex<double>>;

    auto conf =
      rfl::toml::load<ConfHierBEM, rfl::DefaultIfMissing>(config_path).value();

    // Append the test model path before the bare mesh and cad file names.
    conf.project.mesh_file =
      std::string(HBEM_TEST_MODEL_DIR) + conf.project.mesh_file;
    conf.project.cad_file =
      std::string(HBEM_TEST_MODEL_DIR) + conf.project.cad_file;

    REQUIRE((conf.bem.problem_type == "dirichlet") == is_dirichlet);
    REQUIRE((conf.bem.problem_type == "neumann") == !is_dirichlet);

    // Full-matrix constructor: this test only assigns and interpolates
    // boundary data, so the H-matrix runtime is not needed.
    ConfLaplaceBEM                   bem_params{conf.bem.mesh_refinement,
                              conf.bem.fe_order_for_dirichlet_space,
                              conf.bem.fe_order_for_neumann_space,
                              problemTypeLiteralToEnum(conf.bem.problem_type),
                              conf.bem.is_interior_problem};
    LaplaceBEM<2, 3, Number, double> bem(bem_params,
                                         conf.sauter_quad,
                                         conf.linear_solver);
    bem.set_project_name(conf.project.project_name.value());

    {
      std::ifstream mesh_in(conf.project.mesh_file);
      REQUIRE(mesh_in.good());
      read_msh(mesh_in, bem.get_triangulation());
      // bem.get_triangulation().refine_global(1);
    }

    // Create and assign manifolds.
    unsigned int manifold_id = 0;
    for (const auto &manifold : conf.manifolds)
      {
        manifold.visit([&](const auto &alternative) {
          using T = std::decay_t<decltype(alternative)>;
          if constexpr (std::is_same_v<T, ConfSphericalManifold>)
            {
              for (const EntityTag tag : alternative.surface_tags)
                bem.get_manifold_description()[tag] = manifold_id;

              bem.get_manifolds()[manifold_id] =
                new SphericalManifold<2, 3>(alternative.center);
              bem.get_manifold_id_to_mapping_order()[manifold_id] =
                alternative.mapping_order;
              ++manifold_id;
            }
          else
            {
              throw std::runtime_error(
                "This test only configures spherical manifolds");
            }
        });
      }
    bem.initialize_manifolds_from_manifold_description();
    bem.initialize_mappings();

    // Same assignment sequence as LaplaceWorkbench::setupBoundaryConds.
    // The Function objects must outlive interpolation because LaplaceBEM
    // stores raw pointers.
    std::vector<std::unique_ptr<Function<3, Number>>> functions;

    for (const auto &bc : conf.boundary_conditions)
      {
        bc.visit([&](const auto &alternative) {
          using T_bc = std::decay_t<decltype(alternative)>;
          REQUIRE(is_complex_value_spec(alternative.value_spec) ==
                  solver_is_complex);

          auto func =
            create_function_from_value_spec<3, Number>(alternative.value_spec);

          if constexpr (std::is_same_v<T_bc, ConfBEMDirichletBC>)
            {
              REQUIRE(is_dirichlet);
              bem.assign_dirichlet_bc(*func, alternative.surface_tags);
            }
          else if constexpr (std::is_same_v<T_bc, ConfBEMNeumannBC>)
            {
              REQUIRE_FALSE(is_dirichlet);
              bem.assign_neumann_bc(*func, alternative.surface_tags);
            }
          else
            throw std::runtime_error(
              "Robin boundary condition is not covered by this test");

          functions.push_back(std::move(func));
        });
      }

    const auto &definition = is_dirichlet ? bem.get_dirichlet_bc_definition() :
                                            bem.get_neumann_bc_definition();
    // Only two boundary conditions are defined.
    REQUIRE(definition.size() == 2);

    const unsigned int fe_order = is_dirichlet ?
                                    conf.bem.fe_order_for_dirichlet_space :
                                    conf.bem.fe_order_for_neumann_space;

    DoFHandler<2, 3> dof_handler(bem.get_triangulation());
    const std::unique_ptr<FiniteElement<2, 3>> fe =
      is_dirichlet ? std::unique_ptr<FiniteElement<2, 3>>(
                       std::make_unique<FE_Q<2, 3>>(fe_order)) :
                     std::unique_ptr<FiniteElement<2, 3>>(
                       std::make_unique<FE_DGQ<2, 3>>(fe_order));
    dof_handler.distribute_dofs(*fe);

    // Interpolate boundary conditions.
    Vector<Number> values(dof_handler.n_dofs());
    std::map<types::material_id, const Function<3, Number> *> function_map;
    for (const auto &bc : definition)
      {
        function_map[static_cast<types::material_id>(bc.first)] = bc.second;
        auto it = bem.get_material_id_to_mapping_index().find(bc.first);
        VectorTools::interpolate_based_on_material_id(
          bem
            .get_mappings()[it != bem.get_material_id_to_mapping_index().end() ?
                              it->second :
                              0]
            ->get_mapping(),
          dof_handler,
          function_map,
          values);
      }

    std::vector<types::material_id>      dof_material(dof_handler.n_dofs(),
                                                 numbers::invalid_material_id);
    std::vector<Point<3>>                support_points(dof_handler.n_dofs());
    std::vector<types::global_dof_index> local_dofs(fe->dofs_per_cell);
    const auto &unit_support_points = fe->get_unit_support_points();
    std::set<types::material_id> materials;
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell->get_dof_indices(local_dofs);
        materials.insert(cell->material_id());

        const auto mapping_it =
          bem.get_material_id_to_mapping_index().find(cell->material_id());
        const Mapping<2, 3> &mapping =
          bem
            .get_mappings()[mapping_it !=
                                bem.get_material_id_to_mapping_index().end() ?
                              mapping_it->second :
                              0]
            ->get_mapping();

        for (unsigned int d = 0; d < local_dofs.size(); ++d)
          {
            dof_material[local_dofs[d]] = cell->material_id();
            support_points[local_dofs[d]] =
              mapping.transform_unit_to_real_cell(cell, unit_support_points[d]);
          }
      }

    // Only two types of materials are defined in the CAD model.
    REQUIRE(materials.count(1) == 1);
    REQUIRE(materials.count(2) == 1);

    // Compare interpolated values at DoF support points and the value obtained
    // from function evaluation.
    for (unsigned int i = 0; i < values.size(); ++i)
      {
        const auto function_it = function_map.find(dof_material[i]);
        REQUIRE(function_it != function_map.end());
        check_boundary_value(values[i],
                             function_it->second->value(support_points[i]));
      }

    const std::filesystem::path out_dir =
      std::filesystem::path("laplace-workbench-apply-bc") /
      conf.project.project_name.value();
    std::filesystem::create_directories(out_dir);

    // Output interpolated boundary DoF data for visualization.
    const std::string field_name =
      is_dirichlet ? "dirichlet_data" : "neumann_data";
    DataOut<2, 3> data_out;
    if constexpr (solver_is_complex)
      {
        ComplexOutputDataVector<Vector, double> complex_values(values);
        add_complex_data_vector(data_out,
                                dof_handler,
                                complex_values,
                                field_name);
      }
    else
      data_out.add_data_vector(dof_handler, values, field_name);

    data_out.build_patches();

    const auto vtk_path =
      out_dir / (conf.project.project_name.value() + ".vtk");
    {
      std::ofstream vtk_out(vtk_path);
      REQUIRE(vtk_out.good());
      data_out.write_vtk(vtk_out);
    }
    REQUIRE(std::filesystem::file_size(vtk_path) > 0);

    const auto vector_path =
      out_dir / (conf.project.project_name.value() + ".output");
    {
      std::ofstream vector_out(vector_path);
      REQUIRE(vector_out.good());
      print_vector_to_mat(vector_out, field_name, values, false);
    }
    REQUIRE(std::filesystem::file_size(vector_path) > 0);

    {
      std::ifstream     vtk_in(vtk_path);
      const std::string vtk_text(std::istreambuf_iterator<char>(vtk_in),
                                 (std::istreambuf_iterator<char>()));
      REQUIRE_THAT(vtk_text, ContainsSubstring(field_name));
    }
  }
} // namespace

TEST_CASE("Apply distinct boundary data on the two-sphere model",
          "[laplace-workbench][boundary-condition]")
{
  const auto test_case =
    GENERATE(std::make_tuple("dirichlet-real.toml", false, true),
             std::make_tuple("dirichlet-complex.toml", true, true),
             std::make_tuple("dirichlet-real-expression.toml", false, true),
             std::make_tuple("dirichlet-complex-expression.toml", true, true),
             std::make_tuple("neumann-real.toml", false, false),
             std::make_tuple("neumann-complex.toml", true, false),
             std::make_tuple("neumann-real-expression.toml", false, false),
             std::make_tuple("neumann-complex-expression.toml", true, false));

  const std::string config_file  = std::get<0>(test_case);
  const bool        is_complex   = std::get<1>(test_case);
  const bool        is_dirichlet = std::get<2>(test_case);
  const std::string config_path  = std::string(SOURCE_DIR) + "/" + config_file;

  if (is_complex)
    run_two_sphere_bc_case<std::complex<double>>(config_path, is_dirichlet);
  else
    run_two_sphere_bc_case<double>(config_path, is_dirichlet);
}
