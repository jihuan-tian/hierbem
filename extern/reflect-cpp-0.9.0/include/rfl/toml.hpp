#ifndef RFL_TOML_HPP_
#define RFL_TOML_HPP_

// XXX modified by wxz
#if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__)
// compiler is clang or GCC
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wmissing-braces"
#elif defined(_MSC_VER)
// compiler is MSVC
#pragma warning(push)
#pragma warning(disable : 4244)
#pragma warning(disable : 4101)
#endif
// XXX ends here

#include "toml/Parser.hpp"
#include "toml/Reader.hpp"
#include "toml/Writer.hpp"
#include "toml/load.hpp"
#include "toml/read.hpp"
#include "toml/save.hpp"
#include "toml/write.hpp"

// XXX modified by wxz
#if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__)
// compiler is clang or GCC
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
// compiler is MSVC
#pragma warning(pop)
#endif

#endif
