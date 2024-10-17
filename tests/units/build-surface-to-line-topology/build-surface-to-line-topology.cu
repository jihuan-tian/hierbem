/**
 * @file build-surface-to-line-topology.cc
 * @brief
 *
 * @ingroup testers
 * @author
 * @date 2024-08-05
 */

#include "electric_field/ddm_efield.h"
#include "hbem_test_config.h"

using namespace HierBEM;

int
main(int argc, const char *argv[])
{
  (void)argc;
  (void)argv;

  {
    DDMEfield<1, 3> efield;
    efield.read_subdomain_topology(HBEM_TEST_MODEL_DIR
                                   "circle-immersed-in-two-squares.geo",
                                   HBEM_TEST_MODEL_DIR
                                   "circle-immersed-in-two-squares.msh");
    efield.get_subdomain_topology().print(std::cout);
  }

  {
    DDMEfield<1, 3> efield;
    efield.read_subdomain_topology(
      HBEM_TEST_MODEL_DIR
      "circle-immersed-in-two-squares-different-surface-orientations.geo",
      HBEM_TEST_MODEL_DIR
      "circle-immersed-in-two-squares-different-surface-orientations.msh");
    efield.get_subdomain_topology().print(std::cout);
  }

  return 0;
}
