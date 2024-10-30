#ifndef INCLUDE_CONFIG_FILE_H_
#define INCLUDE_CONFIG_FILE_H_

#include <rfl.hpp>
#include <rfl/toml.hpp>

#include <cstdint>
#include <string>

namespace HierBEM
{
  // Project name pattern
  using ProjectName =
    rfl::Pattern<R"(([a-zA-Z][a-zA-Z0-9_]*)?)", "ProjectName">;
  // BEM problem type
  using ProblemType = rfl::Literal<"neumann", "dirichlet", "mixed">;

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
    std::uint32_t boundary_dim = 2; // The dimension of model boundary
    std::uint32_t space_dim    = 3; // The dimension of embeding space
    std::uint32_t fe_order_for_dirichlet_space =
      1; // FE order for Dirichlet space
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
    std::uint32_t max_rank         = 2; // max_rank for preconditioner H-matrix
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
      std::lock_guard<std::mutex> lock(_mutex);
      if (!_initialized)
        {
          _config      = load_config(file_path);
          _initialized = true;
        }
    }

    const ConfHierBEM &
    get_config() const
    {
      if (!_initialized)
        {
          throw std::runtime_error("ConfigFile not initialized");
        }
      return _config;
    }

  private:
    ConfigFile()
      : _initialized(false)
    {}
    ConfigFile(const ConfigFile &) = delete;
    ConfigFile &
    operator=(const ConfigFile &) = delete;

    ConfHierBEM
    load_config(const std::string &file_path)
    {
      return rfl::toml::
        load<ConfHierBEM, rfl::DefaultIfMissing, rfl::NoExtraFields>(file_path)
          .value();
    }

    ConfHierBEM _config;
    bool        _initialized;
    std::mutex  _mutex;
  };

} // namespace HierBEM

#endif /* INCLUDE_CONFIG_FILE_H_ */
