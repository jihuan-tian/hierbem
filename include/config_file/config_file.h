#ifndef HIERBEM_INCLUDE_CONFIG_FILE_CONFIG_FILE_H_
#define HIERBEM_INCLUDE_CONFIG_FILE_CONFIG_FILE_H_

#include <rfl.hpp>
#include <rfl/toml.hpp>

#include <cstdint>
#include <string>

#include "config.h"

HBEM_NS_OPEN

// Project name pattern
using ProjectName = rfl::Pattern<R"(([a-zA-Z][a-zA-Z0-9_]*)?)", "ProjectName">;
// BEM problem type
using ProblemType = rfl::Literal<"neumann", "dirichlet", "mixed">;
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
  std::string input_mesh;            // The input mesh file path
  std::string output_dir = "output"; // The output directory
};

/**
 * Configuration for BEM algorithm
 */
struct ConfBEM
{
  BoundaryDim   boundary_dim = 2; // The dimension of model boundary
  SpaceDim      space_dim    = 3; // The dimension of embeding space
  std::uint32_t fe_order_for_dirichlet_space =
    1;                                          // FE order for Dirichlet space
  std::uint32_t fe_order_for_neumann_space = 0; // FE order for Neumann space
  ProblemType   problem_type =
    ProblemType::make<"dirichlet">(); // The type of BEM problem
  bool is_interior_problem = false;   // Whether the problem is interior
};

/**
 * Configuration for 1-D/2-D cluster tree
 */
struct ConfClusterTree
{
  std::uint32_t n_min_for_ct  = 4;  // n_min for ClusterTree
  std::uint32_t n_min_for_bct = 32; // n_min for BlockClusterTree
};

/**
 * Configuration for H-matrix construction
 */
struct ConfHmatrix
{
  double        eta      = 0.8; // eta for H-matrix
  std::uint32_t max_rank = 5;   // max_rank for H-matrix
  double        aca_relative_err =
    0.01; // for ACA: max relative error while assembling H-matrix
};

/**
 * Configuration for preconditioner H-matrix construction
 */
struct ConfPrecond
{
  double        eta              = 1.0; // eta for preconditioner H-matrix
  std::uint32_t max_rank         = 2;   // max_rank for preconditioner H-matrix
  double        aca_relative_err = 0.1; // for ACA: max relative error while
                                        // assembling preconditioner H-matrix
};

/**
 * Miscellaneous configurations
 */
struct ConfMisc
{
  std::int32_t aca_thread_num = -1; // number of threads for ACA algorithm,
                                    // -1 means using all available threads
};

/**
 * Top-level configuration
 */
struct ConfHierBEM
{
  ConfProj        project;
  ConfBEM         bem;
  ConfClusterTree cluster_tree;
  ConfHmatrix     hmatrix;
  ConfPrecond     precond;
  ConfMisc        misc;
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
        conf_        = load_config(file_path);
        initialized_ = true;
      }
  }

  const ConfHierBEM &
  get_config() const
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
  load_config(const std::string &file_path)
  {
    auto &conf =
      rfl::toml::load<ConfHierBEM, rfl::DefaultIfMissing, rfl::NoExtraFields>(
        file_path)
        .value();

    validate_conf(conf);
    return conf;
  }

  void
  validate_conf(const ConfHierBEM &conf)
  {
    // Space dimension must be greater than boundary dimension
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
