/**
 * @file reflectcpp-usage.cc
 * @brief
 *
 * @ingroup testers
 * @author
 * @date 2024-05-23
 */
#include <catch2/catch_all.hpp>

// XXX This is a workaround to suppress warnings from reflect-cpp headers
// #pragma GCC diagnostic push
// #pragma GCC diagnostic ignored "-Wunused-parameter"
// #pragma GCC diagnostic ignored "-Wtype-limits"
// #pragma GCC diagnostic ignored "-Wmissing-braces"
#include <rfl.hpp>
#include <rfl/json.hpp>
#include <rfl/toml.hpp>
// #pragma GCC diagnostic pop

#include <cstdint>
#include <optional>
#include <string>

#include "hbem_test_config.h"
#include "hbem_test_utils.h"

using namespace Catch::Matchers;

struct ConfigProject
{
  std::string name;
  std::string working_dir = "hierbem_output";
};

struct ConfigBEM
{
  std::uint32_t order_dirichlet         = 1;
  std::uint32_t order_neumann           = 0;
  std::uint32_t mapping_order_dirichlet = 1;
  std::uint32_t mapping_order_neumann   = 1;
  bool          is_interior_problem     = false;
};

struct ConfigHierBEM
{
  ConfigProject project = ConfigProject{};
  ConfigBEM     bem     = ConfigBEM{};
};

TEST_CASE("Sanity: json schema desc", "[json][demo]")
{
  struct Config
  {
    rfl::Description<"The project name should contains only alphanumerics",
                     std::string>
      proj_name;
    rfl::Description<"The finite space dimension, should be 1~3", std::uint32_t>
      space_dim;
  };
  auto json_schema = rfl::json::to_schema<Config>(rfl::json::pretty);
  std::cout << json_schema << std::endl;
}

TEST_CASE("Sanity: validator", "[toml][demo]")
{
  using ProjectName = rfl::Pattern<R"([a-zA-Z0-9_]+)", "ProjectName">;
  using SpaceDim =
    rfl::Validator<std::uint32_t, rfl::AllOf<rfl::Minimum<1>, rfl::Maximum<3>>>;
  struct Config
  {
    ProjectName name;
    SpaceDim    dim;
  };
  auto       toml_str = R"(
    name = "hierbem"
    dim = 3
  )";
  const auto config   = rfl::toml::read<Config>(toml_str).value();
  REQUIRE(config.name == "hierbem");
  REQUIRE(config.dim == 3);
  auto err_toml_str = R"(
    name = "hierbem"
    dim = 0
  )";
  REQUIRE_THROWS(rfl::toml::read<Config>(err_toml_str).value());
  auto err_toml_str1 = R"(
    name = "%*%"
    dim = 2
  )";
  REQUIRE_THROWS(rfl::toml::read<Config>(err_toml_str1).value());
}

enum class Color
{
  Green = 2,
  Red   = 1,
  Blue  = 4,
  Black = 8
};
inline Color
operator|(Color a, Color b)
{
  return static_cast<Color>(static_cast<int>(a) | static_cast<int>(b));
}

TEST_CASE("Sanity: scoped enum", "[toml][demo]")
{
  enum class ProblemType
  {
    Interior,
    Exterior
  };

  struct Config
  {
    ProblemType          problem_type;
    std::optional<Color> color;
  };

  std::string toml_str = R"(
    problem_type = "Interior"
  )";
  const auto  config   = rfl::toml::read<Config>(toml_str).value();
  REQUIRE(config.problem_type == ProblemType::Interior);

  std::string toml_str1 = R"(
    problem_type = "Unknown"
  )";
  REQUIRE_THROWS(rfl::toml::read<Config>(toml_str1).value());

  const auto conf_obj  = Config{.problem_type = ProblemType::Exterior,
                                .color        = Color::Red | Color::Green};
  const auto toml_str2 = rfl::toml::write(conf_obj);
  std::cout << toml_str2 << std::endl;

  const auto conf_obj2 = rfl::toml::read<Config>(toml_str2).value();
  REQUIRE(conf_obj2.problem_type == ProblemType::Exterior);
  REQUIRE(conf_obj2.color.has_value());
  REQUIRE(conf_obj2.color.value() == (Color::Red | Color::Green));
}

TEST_CASE("Sanity: string literals", "[toml][demo]")
{
  using ProblemType = rfl::Literal<"interior", "exterior">;
  struct Config
  {
    ProblemType problem_type;
  };

  std::string toml_str = R"(
    problem_type = "interior"
  )";
  const auto  config   = rfl::toml::read<Config>(toml_str).value();
  std::cout << config.problem_type.name() << std::endl;
  std::cout << config.problem_type.value() << std::endl;
  REQUIRE(config.problem_type == "interior");
  REQUIRE(config.problem_type == ProblemType::value_of<"interior">());

  std::string toml_str_err = R"(
    problem_type = "unknown"
  )";
  REQUIRE_THROWS(rfl::toml::read<Config>(toml_str_err).value());
}

TEST_CASE("Sanity: processors", "[toml][demo]")
{
  struct Person
  {
    std::string first_name;
    std::string last_name;
    int         age;
  };
  const auto homer =
    Person{.first_name = "Homer", .last_name = "Simpson", .age = 45};

  const auto json_string = rfl::json::write<rfl::SnakeCaseToCamelCase>(homer);
  std::cout << json_string << std::endl;

  const auto homer2 =
    rfl::json::read<Person, rfl::SnakeCaseToCamelCase>(json_string).value();

  REQUIRE(homer2.first_name == "Homer");
  REQUIRE(homer2.last_name == "Simpson");
  REQUIRE(homer2.age == 45);

  struct SomeConf
  {
    int         a = 42;
    std::string b = "bar";
  };

  std::string toml_str = R"(
    a = 1
  )";

  const auto conf =
    rfl::toml::read<SomeConf, rfl::DefaultIfMissing>(toml_str).value();

  REQUIRE(conf.a == 1);
  REQUIRE(conf.b == "bar");

  std::string toml_str1 = R"(
    a = 1
    bb = "foo"
  )";
  REQUIRE_NOTHROW(
    rfl::toml::read<SomeConf, rfl::DefaultIfMissing>(toml_str1).value());
  REQUIRE_THROWS(
    rfl::toml::read<SomeConf, rfl::DefaultIfMissing, rfl::NoExtraFields>(
      toml_str1)
      .value());
}

TEST_CASE("Sanity: flatten", "[toml][demo]")
{
  struct Person
  {
    std::string first_name;
    std::string last_name;
    int         age;
  };

  struct Employee
  {
    rfl::Flatten<Person> person;
    std::string          employer;
    float                salary;
  };

  const auto employee =
    Employee{.person =
               Person{.first_name = "Homer", .last_name = "Simpson", .age = 45},
             .employer = "Mr. Burns",
             .salary   = 60000.0f};
  const auto toml_string = rfl::toml::write(employee);
  std::cout << toml_string << std::endl;

  auto someone = rfl::toml::read<Employee>(toml_string).value();
  REQUIRE(someone.person().first_name == "Homer");
  REQUIRE(someone.person.get().last_name == "Simpson");
  REQUIRE(someone.person.value_.age == 45);
  REQUIRE(someone.employer == "Mr. Burns");
  REQUIRE(someone.salary == 60000.0f);

  someone.person().first_name = "Bart";
  REQUIRE(someone.person.get().first_name == "Bart");
}

TEST_CASE("Sanity: optional field", "[toml][demo]")
{
  struct Person
  {
    std::string                        first_name;
    std::string                        last_name;
    std::optional<std::vector<Person>> children;
  };
  const auto wzy         = Person{.first_name = "Zhiyuan",
                                  .last_name  = "Wang",
                                  .children   = std::nullopt};
  const auto wyn         = Person{.first_name = "Yuning",
                                  .last_name  = "Wang",
                                  .children   = std::nullopt};
  const auto wxz         = Person{.first_name = "Xiaozhe",
                                  .last_name  = "Wang",
                                  .children   = std::vector<Person>({wzy, wyn})};
  const auto toml_string = rfl::toml::write(wxz);
  std::cout << toml_string << std::endl;

  const auto someone = rfl::toml::read<Person>(toml_string).value();
  REQUIRE(someone.first_name == "Xiaozhe");
  REQUIRE(someone.last_name == "Wang");
  REQUIRE(someone.children.has_value());
  REQUIRE(someone.children.value().size() == 2);
  REQUIRE(!someone.children.value()[0].children.has_value());
  REQUIRE(!someone.children.value()[1].children.has_value());
}

TEST_CASE("TOML serialize", "[toml][demo]")
{
  const auto config = ConfigHierBEM{
    .project = ConfigProject{.name = "test", .working_dir = "output"}};
  const std::string toml_str = rfl::toml::write(config);
  rfl::toml::save("serialized.toml", config);

  REQUIRE_THAT(toml_str, ContainsSubstring("name = 'test'"));
  REQUIRE_THAT(toml_str, ContainsSubstring("working_dir = 'output'"));
  REQUIRE_THAT(toml_str, ContainsSubstring("order_dirichlet = 1"));
  REQUIRE_THAT(toml_str, ContainsSubstring("order_neumann = 0"));
  REQUIRE_THAT(toml_str, ContainsSubstring("mapping_order_dirichlet = 1"));
  REQUIRE_THAT(toml_str, ContainsSubstring("mapping_order_neumann = 1"));
  REQUIRE_THAT(toml_str, ContainsSubstring("is_interior_problem = false"));
}

TEST_CASE("TOML deserialize", "[toml][demo]")
{
  const auto default_conf = ConfigHierBEM{};
  rfl::toml::write(default_conf, std::cout);
  const auto extern_conf =
    rfl::toml::load<ConfigHierBEM, rfl::DefaultIfMissing>(SOURCE_DIR
                                                          "/test_config.toml")
      .value();
  REQUIRE(extern_conf.project.name == "test");
  REQUIRE(extern_conf.project.working_dir == "output");
  REQUIRE(extern_conf.bem.order_dirichlet == 1);
  REQUIRE(extern_conf.bem.order_neumann == 0);
  REQUIRE(extern_conf.bem.mapping_order_dirichlet == 1);
  REQUIRE(extern_conf.bem.mapping_order_neumann == 1);
  REQUIRE(extern_conf.bem.is_interior_problem == true);
}
