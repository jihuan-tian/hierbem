/**
 * @file create-efield-subdomains.cu
 * @brief
 *
 * @ingroup test_cases
 * @author
 * @date 2024-08-06
 */

#include "electric_field/ddm_efield.h"
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
  // At the moment, we manually assign problem parameters.
  efield.initialize_parameters();

  efield.create_efield_subdomains_and_surfaces();

  return 0;
}
