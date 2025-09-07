/**
 * @file read-1d-skeleton-mesh.cc
 * @brief Verify reading 1D skeleton mesh, which is used in 2D BEM.
 *
 * @ingroup test_cases
 * @author Jihuan Tian
 * @date 2024-08-05
 */

#include "electric_field/ddm_efield.h"
#include "grid/grid_in_ext.h"
#include "hbem_test_config.h"

using namespace HierBEM;

int
main(int argc, const char *argv[])
{
  (void)argc;
  (void)argv;

  DDMEfield<1, 3> efield;
  read_msh(HBEM_TEST_MODEL_DIR
           "circle-immersed-in-two-squares-different-surface-orientations.msh",
           efield.get_triangulation(),
           false);

  return 0;
}
