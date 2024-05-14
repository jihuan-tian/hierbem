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
