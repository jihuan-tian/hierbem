#include <fmt/core.h>

#include "config_file/config_file.h"

using namespace HierBEM;

int
main(int argc, char **argv)
{
  if (argc != 2)
    {
      std::cerr << "Usage: " << argv[0] << " <config file>" << std::endl;
      return 1;
    }
  ConfigFile::instance().initialize(argv[1]);

  const auto &conf_inst = ConfigFile::instance().get_config();
  fmt::println("{}", conf_inst.project.project_name.value());
  fmt::println("{}", conf_inst.project.input_mesh);
  fmt::println("{}", conf_inst.project.output_dir);
  fmt::println("{}", conf_inst.misc.aca_thread_num);
  return 0;
}
