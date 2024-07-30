/**
 * @file assign-entity-tag-to-cell.cc
 * @brief
 *
 * @ingroup testers
 * @author
 * @date 2024-07-28
 */

#include <fstream>
#include <iostream>

#include "electric_field/ddm_efield.h"
#include "grid_out_ext.h"
#include "hbem_test_config.h"

using namespace HierBEM;

int
main(int argc, const char *argv[])
{
  (void)argc;
  (void)argv;

  DDMEfield<2, 3> efield;
  efield.read_skeleton_mesh(HBEM_TEST_MODEL_DIR
                            "sphere-immersed-in-two-boxes.msh");

  return 0;
}
