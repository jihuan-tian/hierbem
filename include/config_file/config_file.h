// Copyright (C) 2025 Xiaozhe Wang <chaoslawful@gmail.com>
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

#include <rfl.hpp>
#include <rfl/toml.hpp>

#include <cstdint>
#include <string>

#include "config.h"
#include "config_structs.h"

HBEM_NS_OPEN

// Project name pattern
using ProjectName =
  rfl::Pattern<R"(([a-zA-Z][a-zA-Z0-9_.=\-]*)?)", "ProjectName">;
// BEM problem type
using ProblemTypeLiteral = rfl::Literal<"dirichlet", "neumann", "mixed">;
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
  std::string cad_file   = "";       // The input CAD file path (optional)
  std::string output_dir = "output"; // The output directory
};

/**
 * Configuration for boundary manifold reconstruction
 */
struct ConfManifold
{
  // TBD
};

struct ConfBoundaryCondition
{
  // TBD
};

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
 * Top-level configuration
 */
struct ConfHierBEM
{
  ConfProj                   project;
  ConfBEM                    bem;
  ConfHMatrix                hmatrix;
  ConfHMatrix                hmatrix_precond;
  ConfSauterQuad             sauter_quad;
  ConfSauterQuad             sauter_quad_precond;
  ConfLinearSolver           linear_solver;
  ConfOperatorPreconditioner op_precond;
  ConfParallelization        parallel;
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
    auto conf =
      rfl::toml::load<ConfHierBEM, rfl::DefaultIfMissing, rfl::NoExtraFields>(
        file_path)
        .value();
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
  }

  ConfHierBEM conf_;
  bool        initialized_;
  std::mutex  lock_;
};

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_CONFIG_FILE_CONFIG_FILE_H_
