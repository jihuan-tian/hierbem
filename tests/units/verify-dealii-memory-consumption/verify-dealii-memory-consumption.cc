/**
 * @file verify-dealii-memory-consumption.cc
 * @brief Verify the memory consumption estimation function provided by deal.ii.
 *
 * @p MemoryConsumption::memory_consumption will not count the allocated memory
 * associated with each pointer.
 *
 * @ingroup testers
 * @author Jihuan Tian
 * @date 2024-02-18
 */

#include <deal.II/base/memory_consumption.h>
#include <deal.II/base/table_handler.h>

#include <cxxabi.h>

#include <iostream>
#include <typeinfo>
#include <vector>

using namespace std;
using namespace dealii;

template <typename T>
void
add_memory_consumption_table_row(TableHandler &table, const T &data)
{
  int   demangle_status = -1;
  char *demangled_type_name =
    abi::__cxa_demangle(typeid(data).name(), NULL, NULL, &demangle_status);
  table.add_value("Type", demangled_type_name);
  unsigned int type_size = sizeof(data);
  table.add_value("sizeof", type_size);
  table.add_value("Capacity", data.capacity());
  unsigned int memory_size = MemoryConsumption::memory_consumption(data);
  table.add_value("Memory consumption", memory_size);
  table.add_value("Data size", memory_size - type_size);
}

int
main()
{
  std::vector<int>   a(10);
  std::vector<int *> vector_of_arrays;

  for (unsigned int i = 0; i < 10; i++)
    {
      int *array_pointer = new int[10];
      vector_of_arrays.push_back(array_pointer);
    }

  TableHandler table;

  add_memory_consumption_table_row(table, a);
  add_memory_consumption_table_row(table, vector_of_arrays);

  table.write_text(std::cout, TableHandler::TextOutputFormat::org_mode_table);

  for (auto array_pointer : vector_of_arrays)
    {
      delete[] array_pointer;
    }

  return 0;
}
