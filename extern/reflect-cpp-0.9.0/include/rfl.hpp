#ifndef RFL_RFL_HPP_
#define RFL_RFL_HPP_

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

#include "rfl/AllOf.hpp"
#include "rfl/AnyOf.hpp"
#include "rfl/Attribute.hpp"
#include "rfl/Box.hpp"
#include "rfl/Description.hpp"
#include "rfl/Field.hpp"
#include "rfl/Flatten.hpp"
#include "rfl/Literal.hpp"
#include "rfl/NamedTuple.hpp"
#include "rfl/OneOf.hpp"
#include "rfl/Pattern.hpp"
#include "rfl/PatternValidator.hpp"
#include "rfl/Ref.hpp"
#include "rfl/Rename.hpp"
#include "rfl/Size.hpp"
#include "rfl/TaggedUnion.hpp"
#include "rfl/Timestamp.hpp"
#include "rfl/Validator.hpp"
#include "rfl/Variant.hpp"
#include "rfl/always_false.hpp"
#include "rfl/as.hpp"
#include "rfl/comparisons.hpp"
#include "rfl/default.hpp"
#include "rfl/define_literal.hpp"
#include "rfl/define_named_tuple.hpp"
#include "rfl/define_tagged_union.hpp"
#include "rfl/define_variant.hpp"
#include "rfl/enums.hpp"
#include "rfl/extract_discriminators.hpp"
#include "rfl/field_type.hpp"
#include "rfl/fields.hpp"
#include "rfl/from_named_tuple.hpp"
#include "rfl/get.hpp"
#include "rfl/make_named_tuple.hpp"
#include "rfl/name_t.hpp"
#include "rfl/named_tuple_t.hpp"
#include "rfl/parsing/CustomParser.hpp"
#include "rfl/patterns.hpp"
#include "rfl/remove_fields.hpp"
#include "rfl/replace.hpp"
#include "rfl/to_named_tuple.hpp"
#include "rfl/to_view.hpp"
#include "rfl/type_name_t.hpp"
#include "rfl/visit.hpp"

// XXX modified by wxz
#if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__)
// compiler is clang or GCC
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
// compiler is MSVC
#pragma warning(pop)
#endif

#endif
