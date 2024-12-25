#ifndef HIERBEM_INCLUDE_DOF_TO_CELL_TOPOLOGY_H_
#define HIERBEM_INCLUDE_DOF_TO_CELL_TOPOLOGY_H_

#include "config.h"

HBEM_NS_OPEN

template <int dim, int spacedim>
class DofToCellTopology
{
public:
  /**
   * The core structure describing the DoF-to-cell topology.
   */
  std::vector<
    std::vector<const typename DoFHandler<dim, spacedim>::cell_iterator *>>
    topology;

  /**
   * Maximum number of cells associated with a DoF.
   */
  unsigned int max_cells_per_dof;
};

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_DOF_TO_CELL_TOPOLOGY_H_