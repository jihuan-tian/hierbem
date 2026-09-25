// Copyright (C) 2025 Xiaozhe Wang <chaoslawful@gmail.com>
// Copyright (C) 2026 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

#ifndef HIERBEM_INCLUDE_CONFIG_FILE_CONFIG_FILE_H_
#define HIERBEM_INCLUDE_CONFIG_FILE_CONFIG_FILE_H_

#include <deal.II/base/function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>

#include <rfl.hpp>
#include <rfl/toml.hpp>

#include <array>
#include <complex>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <vector>

#include "cad_mesh/gmsh_manipulation.h"
#include "config.h"
#include "config_file/complex_function_parser.h"
#include "config_file/config_file.h"
#include "config_file/function_parser_template_ret.h"
#include "config_structs.h"

/**
 * Custom rfl parsers
 */
namespace rfl
{
  template <typename Number>
  struct Reflector<std::complex<Number>>
  {
    struct ReflType
    {
      Number real_part;
      Number imag_part;
    };

    static std::complex<Number>
    to(const ReflType &refl)
    {
      return std::complex<Number>(refl.real_part, refl.imag_part);
    }

    static ReflType
    from(const std::complex<Number> &value)
    {
      return ReflType{value.real(), value.imag()};
    }
  };

  template <int dim, typename Number>
  struct Reflector<dealii::Point<dim, Number>>
  {
    using ReflType = std::array<Number, dim>;

    static dealii::Point<dim, Number>
    to(const ReflType &refl)
    {
      dealii::Point<dim, Number> pt;

      for (unsigned int i = 0; i < dim; i++)
        pt[i] = refl[i];

      return pt;
    }

    static ReflType
    from(const dealii::Point<dim, Number> &pt)
    {
      ReflType refl{};

      for (unsigned int i = 0; i < dim; i++)
        refl[i] = pt[i];

      return refl;
    }
  };

  template <int rank, int dim, typename Number>
  struct Reflector<dealii::Tensor<rank, dim, Number>>
  {
    using ReflType =
      std::array<Number,
                 dealii::Tensor<rank, dim, Number>::n_independent_components>;

    static dealii::Tensor<rank, dim, Number>
    to(const ReflType &refl)
    {
      dealii::Tensor<rank, dim, Number> ts;

      for (unsigned int i = 0;
           i < dealii::Tensor<rank, dim, Number>::n_independent_components;
           i++)
        ts[dealii::Tensor<rank, dim, Number>::unrolled_to_component_indices(
          i)] = refl[i];

      return ts;
    }

    static ReflType
    from(const dealii::Tensor<rank, dim, Number> &ts)
    {
      ReflType refl{};

      for (unsigned int i = 0;
           i < dealii::Tensor<rank, dim, Number>::n_independent_components;
           i++)
        refl[i] =
          ts[dealii::Tensor<rank, dim, Number>::unrolled_to_component_indices(
            i)];

      return refl;
    }
  };
} // namespace rfl

HBEM_NS_OPEN

// Project name pattern
using ProjectName = rfl::Pattern<R"(([a-zA-Z0-9_.=\-]*)?)", "ProjectName">;
// BEM problem type
using ProblemTypeLiteral =
  rfl::Literal<"dirichlet", "neumann", "mixed", "robin">;
// Preconditioner type
using PreconditionerTypeLiteral = rfl::
  Literal<"factorization", "operator", "identity", "jacobi", "block_jacobi">;
// Type of H-matrix/vector multiplication, called by iterative solver
using VmultTypeLiteral =
  rfl::Literal<"serial_recursive", "serial_iterative", "task_parallel">;
// BEM space dimension (currently only 3 is supported)
using SpaceDim =
  rfl::Validator<std::uint32_t, rfl::AllOf<rfl::Minimum<3>, rfl::Maximum<3>>>;
// BEM boundary dimension (currently only 2 is supported)
using BoundaryDim =
  rfl::Validator<std::uint32_t, rfl::AllOf<rfl::Minimum<2>, rfl::Maximum<2>>>;

/**
 * Configuration for a simulation project
 */
struct ConfProj
{
  ProjectName project_name;          // The name of the project
  std::string mesh_file;             // The input mesh file path
  std::string cad_file;              // The input CAD file path
  std::string output_dir = "output"; // The output directory
};

/**
 * Configuration for spherical manifold
 */
struct ConfSphericalManifold
{
  using Tag = rfl::Literal<"spherical">;
  std::vector<EntityTag>   surface_tags;
  dealii::Point<3, double> center;
  std::uint32_t            mapping_order = 2;
};

/**
 * Configuration for cylindrical manifold, whose axis is one of the Cartesian
 * axes.
 */
struct ConfCylindricalManifoldCartesianAxis
{
  using Tag = rfl::Literal<"cylindrical_cartesian_axis">;
  std::vector<EntityTag> surface_tags;
  // Axis = 0, 1, 2 represent x, y and z axes respectively.
  std::uint32_t axis;
  double        tolerance     = 1e-10;
  std::uint32_t mapping_order = 2;
};

/**
 * Configuration for cylindrical manifold with an arbitrary axis.
 */
struct ConfCylindricalManifold
{
  using Tag = rfl::Literal<"cylindrical">;
  std::vector<EntityTag>       surface_tags;
  dealii::Tensor<1, 3, double> direction;
  dealii::Point<3, double>     point_on_axis;
  double                       tolerance     = 1e-10;
  std::uint32_t                mapping_order = 2;
};

/**
 * Configuration for torus manifold, centered at the origin.
 */
struct ConfTorusManifold
{
  using Tag = rfl::Literal<"torus">;
  std::vector<EntityTag> surface_tags;
  double                 R;
  double                 r;
  std::uint32_t          mapping_order = 2;
};

/**
 * Configuration for OCC normal projection manifold.
 */
struct ConfOCCNormalProjectionManifold
{
  using Tag = rfl::Literal<"occ_normal_projection">;
  std::vector<EntityTag> surface_tags;
  double                 tolerance     = 1e-7;
  std::uint32_t          mapping_order = 2;
};

/**
 * Configuration for OCC directional projection manifold.
 */
struct ConfOCCDirectionalProjectionManifold
{
  using Tag = rfl::Literal<"occ_directional_projection">;
  std::vector<EntityTag>       surface_tags;
  dealii::Tensor<1, 3, double> direction;
  double                       tolerance     = 1e-7;
  std::uint32_t                mapping_order = 2;
};

/**
 * Configuration for OCC normal-to-mesh projection manifold.
 */
struct ConfOCCNormalToMeshProjectionManifold
{
  using Tag = rfl::Literal<"occ_normal2mesh_projection">;
  std::vector<EntityTag> surface_tags;
  double                 tolerance     = 1e-7;
  std::uint32_t          mapping_order = 2;
};

/**
 * Configuration for OCC NURBS patch manifold. N.B. For each surface tag, an
 * independent NURBS patch manifold should be created.
 */
struct ConfOCCNURBSPatchManifold
{
  using Tag = rfl::Literal<"occ_nurbs_patch">;
  std::vector<EntityTag> surface_tags;
  double                 tolerance     = 1e-7;
  std::uint32_t          mapping_order = 2;
};

using ConfManifold = rfl::TaggedUnion<"kind",
                                      ConfSphericalManifold,
                                      ConfCylindricalManifoldCartesianAxis,
                                      ConfCylindricalManifold,
                                      ConfTorusManifold,
                                      ConfOCCNormalProjectionManifold,
                                      ConfOCCDirectionalProjectionManifold,
                                      ConfOCCNormalToMeshProjectionManifold,
                                      ConfOCCNURBSPatchManifold>;

/**
 * Configuration for BEM boundary conditions
 */
using ScalarValue = rfl::Variant<double, std::complex<double>>;
struct ConfConstantValue
{
  using Tag = rfl::Literal<"constant">;
  ScalarValue value;
};

struct ConfExpression
{
  using Tag = rfl::Literal<"expression">;
  std::string expr;
};

struct ConfExpressionComplex
{
  using Tag = rfl::Literal<"expression_complex">;
  std::string real_part_expr;
  std::string imag_part_expr;
};

/**
 * Value specification which can be used for boundary condition and source
 * term.
 */
using ConfValueSpec = rfl::
  TaggedUnion<"type", ConfConstantValue, ConfExpression, ConfExpressionComplex>;

/**
 * Check if a @p ConfValueSpec object is complex valued.
 */
bool
is_complex_value_spec(const ConfValueSpec &spec);

/**
 * Check if the solver is complex valued by inspecting the @p value_spec
 * property of the first boundary condition object. All other boundary
 * conditions are assumed to have the same value type as the first one.
 */
bool
is_solver_complex_valued();

template <int spacedim, typename Number>
std::unique_ptr<Function<spacedim, Number>>
create_function_from_value_spec(const ConfValueSpec &spec)
{
  // Constants and variables used in muparser expressions.
  const std::map<std::string, double> expr_constants{{"pi", numbers::PI},
                                                     {"e", numbers::E}};
  const std::string                   expr_variables = "x,y,z";

  return spec.visit([&](const auto &alternative)
                      -> std::unique_ptr<Function<spacedim, Number>> {
    using T = std::decay_t<decltype(alternative)>;
    if constexpr (std::is_same_v<T, ConfConstantValue>)
      {
        const Number value = rfl::get<Number>(alternative.value);
        return std::make_unique<Functions::ConstantFunction<spacedim, Number>>(
          value);
      }
    else if constexpr (std::is_same_v<T, ConfExpression>)
      {
        if constexpr (std::is_same_v<Number, double>)
          {
            auto func =
              std::make_unique<FunctionParserTemplateRet<spacedim, Number>>(1);
            func->initialize(expr_variables, alternative.expr, expr_constants);
            return func;
          }
        else
          {
            throw std::runtime_error(
              "Real valued expression cannot be used in a complex valued solver");
          }
      }
    else if constexpr (std::is_same_v<T, ConfExpressionComplex>)
      {
        if constexpr (std::is_same_v<Number, std::complex<double>>)
          {
            auto func =
              std::make_unique<ComplexFunctionParser<spacedim, Number>>(1);
            func->initialize(expr_variables,
                             alternative.real_part_expr,
                             alternative.imag_part_expr,
                             expr_constants);
            return func;
          }
        else
          {
            throw std::runtime_error(
              "Complex valued expression cannot be used in a real valued solver");
          }
      }
    else
      throw std::runtime_error("Invalid value specification type");
  });
}

/**
 * Configuration for Dirichlet boundary condition
 */
struct ConfBEMDirichletBC
{
  using Tag = rfl::Literal<"dirichlet">;
  std::vector<EntityTag> surface_tags;
  ConfValueSpec          value_spec = ConfConstantValue{.value = 0.0};
};

/**
 * Configuration for Neumann boundary condition
 */
struct ConfBEMNeumannBC
{
  using Tag = rfl::Literal<"neumann">;
  std::vector<EntityTag> surface_tags;
  ConfValueSpec          value_spec = ConfConstantValue{.value = 0.0};
};

/**
 * Configuration for Robin boundary condition
 *
 * The Robin boundary condition has the formulation \f$ \gamma_1 u(x) + a(x)
 * \gamma_0 u(x) = g(x) \f$, where @p coeff_spec is \f$a(x)\f$ and @p value_spec
 * is \f$g(x)\f$.
 */
struct ConfBEMRobinBC
{
  using Tag = rfl::Literal<"robin">;
  std::vector<EntityTag> surface_tags;
  ConfValueSpec          coeff_spec = ConfConstantValue{.value = 0.0};
  ConfValueSpec          value_spec = ConfConstantValue{.value = 0.0};
};

using ConfBEMBoundaryCondition = rfl::
  TaggedUnion<"kind", ConfBEMDirichletBC, ConfBEMNeumannBC, ConfBEMRobinBC>;

/**
 * Configuration for BEM algorithm
 */
struct ConfBEM
{
  BoundaryDim   boundary_dim    = 2; // The dimension of model boundary
  SpaceDim      space_dim       = 3; // The dimension of embeding space
  std::uint32_t mesh_refinement = 0; // Number of global mesh refinement
  std::uint32_t fe_order_for_dirichlet_space =
    1;                                          // FE order for Dirichlet space
  std::uint32_t fe_order_for_neumann_space = 0; // FE order for Neumann space
  ProblemTypeLiteral problem_type =
    ProblemTypeLiteral::make<"dirichlet">(); // The type of BEM problem
  bool is_interior_problem = false;          // Whether the problem is interior
  PreconditionerTypeLiteral precond_type =
    PreconditionerTypeLiteral::make<"operator">();
  VmultTypeLiteral vmult_type = VmultTypeLiteral::make<"serial_iterative">();
};

/**
 * Top-level configuration for pure BEM solver
 */
struct ConfHierBEM
{
  ConfProj                              project;
  ConfBEM                               bem;
  std::vector<ConfBEMBoundaryCondition> boundary_conditions;
  std::vector<ConfManifold>             manifolds;
  ConfHMatrix                           hmatrix;
  ConfHMatrix                           hmatrix_precond;
  ConfSauterQuad                        sauter_quad;
  ConfSauterQuad                        sauter_quad_precond;
  ConfLinearSolver                      linear_solver;
  ConfOperatorPreconditioner            op_precond;
  ConfParallelization                   parallel;
};

/**
 * Global configuration singleton
 */
class ConfigFile
{
public:
  static ConfigFile &
  instance()
  {
    static ConfigFile instance;
    return instance;
  }

  void
  initialize(const std::string &file_path)
  {
    std::lock_guard<std::mutex> lock(lock_);
    if (!initialized_)
      {
        conf_ = loadConf(file_path);
        validateConf(conf_);
        initialized_ = true;
      }
  }

  const ConfHierBEM &
  getConfig() const
  {
    if (!initialized_)
      {
        throw std::runtime_error("ConfigFile not initialized");
      }
    return conf_;
  }

private:
  ConfigFile()
    : initialized_(false)
  {}
  ConfigFile(const ConfigFile &) = delete;
  ConfigFile &
  operator=(const ConfigFile &) = delete;

  ConfHierBEM
  loadConf(const std::string &file_path)
  {
    // rfl::NoExtraFields should be removed since TaggedUnion is used, which
    // leaves keys in the config file that do not correspond to fields in config
    // structs.
    auto conf =
      rfl::toml::load<ConfHierBEM, rfl::DefaultIfMissing>(file_path).value();
    return conf;
  }

  void
  validateConf(const ConfHierBEM &conf)
  {
    if (conf.bem.space_dim.value() != 3 || conf.bem.boundary_dim.value() != 2)
      {
        throw std::runtime_error("Currently only 3D/2D problems are supported");
      }

    if (conf.bem.space_dim.value() <= conf.bem.boundary_dim.value())
      {
        throw std::runtime_error(
          "'space_dim' must be greater than 'boundary_dim'");
      }

    // Validate boundary conditions.
    if (conf.boundary_conditions.empty())
      {
        throw std::runtime_error("At least one boundary condition is required");
      }

    bool has_dirichlet_bc = false;
    bool has_neumann_bc   = false;
    bool has_robin_bc     = false;
    for (const auto &bc : conf.boundary_conditions)
      {
        bc.visit([&](const auto &alternative) {
          using T = std::decay_t<decltype(alternative)>;
          if (alternative.surface_tags.empty())
            {
              throw std::runtime_error(
                "Surface tags should not be empty for a boundary condition");
            }

          if constexpr (std::is_same_v<T, ConfBEMDirichletBC>)
            has_dirichlet_bc = true;
          else if constexpr (std::is_same_v<T, ConfBEMNeumannBC>)
            has_neumann_bc = true;
          else if constexpr (std::is_same_v<T, ConfBEMRobinBC>)
            has_robin_bc = true;
          else
            throw std::runtime_error("Invalid boundary condition kind");
        });
      }

    if (has_dirichlet_bc && !has_neumann_bc && !has_robin_bc)
      {
        // Dirichlet problem
        if (conf.bem.problem_type != "dirichlet")
          {
            throw std::runtime_error(
              "'problem_type' must be 'dirichlet' when only Dirichlet boundary condition is specified");
          }
      }
    else if (!has_dirichlet_bc && has_neumann_bc && !has_robin_bc)
      {
        // Neumann problem
        if (conf.bem.problem_type != "neumann")
          {
            throw std::runtime_error(
              "'problem_type' must be 'neumann' when only Neumann boundary condition is specified");
          }
      }
    else if (!has_dirichlet_bc && !has_neumann_bc && has_robin_bc)
      {
        // Robin problem
        if (conf.bem.problem_type != "robin")
          {
            throw std::runtime_error(
              "'problem_type' must be 'robin' when only Robin boundary condition is specified");
          }
      }
    else if (has_dirichlet_bc && has_neumann_bc && !has_robin_bc)
      {
        // At the moment, we only consider mixed boundary conditions which
        // include Dirichlet and Neumann boundary conditions.
        if (conf.bem.problem_type != "mixed")
          {
            throw std::runtime_error(
              "'problem_type' must be 'mixed' when both Dirichlet and Neumann boundary conditions are specified");
          }
      }
    else
      {
        throw std::runtime_error(
          "Robin boundary condition is not allowed in mixed problem at the moment");
      }

    // Validate manifolds.
    std::set<EntityTag> surface_tags_assigned_manifolds;
    for (const auto &manifold : conf.manifolds)
      {
        manifold.visit([&](const auto &alternative) {
          using T = std::decay_t<decltype(alternative)>;
          if (alternative.surface_tags.empty())
            {
              throw std::runtime_error(
                "Surface tags should not be empty for a manifold");
            }

          for (EntityTag tag : alternative.surface_tags)
            if (!surface_tags_assigned_manifolds.insert(tag).second)
              throw std::runtime_error(
                std::string("The surface tag ") + std::to_string(tag) +
                std::string(" is assigned more than one manifold"));

          if constexpr (std::is_same_v<T, ConfCylindricalManifoldCartesianAxis>)
            {
              if (alternative.axis > 2)
                throw std::runtime_error("'axis' should be 0, 1 or 2");
            }
          else if constexpr (std::is_same_v<T, ConfCylindricalManifold>)
            {
              if (alternative.direction.norm() < 1e-12)
                {
                  throw std::runtime_error(
                    "'direction' should not be a zero vector");
                }
            }
          else if constexpr (std::is_same_v<T, ConfTorusManifold>)
            {
              if (!(alternative.R > alternative.r && alternative.r > 0.0))
                {
                  throw std::runtime_error(
                    "Torus radius should satisfy R > r > 0");
                }
            }
          else if constexpr (std::is_same_v<T, ConfOCCNormalProjectionManifold>)
            {
              if (conf.project.cad_file.empty())
                {
                  throw std::runtime_error(
                    "CAD file is needed for defining OCC related manifolds");
                }
              else
                {
                  if (conf.project.cad_file.rfind(std::string(".geo")) !=
                      std::string::npos)
                    throw std::runtime_error(
                      "CAD file should not be *.geo file");
                }
            }
          else if constexpr (std::is_same_v<
                               T,
                               ConfOCCDirectionalProjectionManifold>)
            {
              if (conf.project.cad_file.empty())
                {
                  throw std::runtime_error(
                    "CAD file is needed for defining OCC related manifolds");
                }
              else
                {
                  if (conf.project.cad_file.rfind(std::string(".geo")) !=
                      std::string::npos)
                    throw std::runtime_error(
                      "CAD file should not be *.geo file");
                }

              if (alternative.direction.norm() < 1e-12)
                {
                  throw std::runtime_error(
                    "'direction' should not be a zero vector");
                }
            }
          else if constexpr (std::is_same_v<
                               T,
                               ConfOCCNormalToMeshProjectionManifold>)
            {
              if (conf.project.cad_file.empty())
                {
                  throw std::runtime_error(
                    "CAD file is needed for defining OCC related manifolds");
                }
              else
                {
                  if (conf.project.cad_file.rfind(std::string(".geo")) !=
                      std::string::npos)
                    throw std::runtime_error(
                      "CAD file should not be *.geo file");
                }
            }
          else if constexpr (std::is_same_v<T, ConfOCCNURBSPatchManifold>)
            {
              if (conf.project.cad_file.empty())
                {
                  throw std::runtime_error(
                    "CAD file is needed for defining OCC related manifolds");
                }
              else
                {
                  if (conf.project.cad_file.rfind(std::string(".geo")) !=
                      std::string::npos)
                    throw std::runtime_error(
                      "CAD file should not be *.geo file");
                }
            }

          if (alternative.mapping_order < 1 || alternative.mapping_order > 3)
            {
              throw std::runtime_error(
                "'mapping_order' adopted for a manifold should be 1, 2 or 3");
            }
        });
      }
  }

  ConfHierBEM conf_;
  bool        initialized_;
  std::mutex  lock_;
};

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_CONFIG_FILE_CONFIG_FILE_H_
