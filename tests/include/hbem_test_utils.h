// Copyright (C) 2024 Xiaozhe Wang <chaoslawful@gmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

#ifndef HBEM_TEST_UTILS_H_
#define HBEM_TEST_UTILS_H_

#include <catch2/catch_all.hpp>

#include <filesystem>
#include <iostream>
#include <string>

namespace HierBEM
{

  /**
   * Scoped working directory to isolate testcases that generating intermediate
   * data files, to prevent multiple testcases parallelly writing to the same
   * files.
   *
   * @note This class is not thread-safe.
   * @note The instance of this class must be declared as `volatile` to prevent
   * compiler from optimizing unused instances.
   */
  class HBEMTestScopedDirectory
  {
  public:
    HBEMTestScopedDirectory(const std::string &prefix = "")
    {
      auto test_name = Catch::getResultCapture().getCurrentTestName();
      auto new_cwd =
        prefix + std::to_string(std::hash<std::string>{}(test_name));
      this->set_current_path(new_cwd);
    }

    ~HBEMTestScopedDirectory()
    {
      this->restore_current_path();
    }

  protected:
    void
    set_current_path(const std::string &new_cwd)
    {
      this->old_cwd_ = std::filesystem::current_path();

      if (!std::filesystem::exists(new_cwd))
        std::filesystem::create_directory(new_cwd);
      std::filesystem::current_path(new_cwd);

      std::cout << "Previous path: " << this->old_cwd_ << std::endl;
      std::cout << "Current path changed to: "
                << std::filesystem::current_path() << std::endl;
    }

    void
    restore_current_path()
    {
      std::filesystem::current_path(this->old_cwd_);
      std::cout << "Current path restored to: " << this->old_cwd_ << std::endl;
    }

    std::string old_cwd_;
  };

} // namespace HierBEM

#endif
