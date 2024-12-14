/**
 * @file build-volume-to-surface-topology.cc
 * @brief
 *
 * @ingroup testers
 * @author
 * @date 2024-07-30
 */

#include "electric_field/ddm_efield.hcu"
#include "hbem_test_config.h"

using namespace HierBEM;

int
main(int argc, const char *argv[])
{
  (void)argc;
  (void)argv;

  DDMEfield<2, 3> efield;
  efield.read_subdomain_topology(HBEM_TEST_MODEL_DIR
                                 "sphere-immersed-in-two-boxes.brep",
                                 HBEM_TEST_MODEL_DIR
                                 "sphere-immersed-in-two-boxes.msh");
  efield.get_subdomain_topology().print(std::cout);

  return 0;
}
