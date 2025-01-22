#ifndef HIERBEM_INCLUDE_DOF_TO_CELL_TOPOLOGY_H_
#define HIERBEM_INCLUDE_DOF_TO_CELL_TOPOLOGY_H_

#include <deal.II/dofs/dof_handler.h>

#include "config.h"

HBEM_NS_OPEN

using namespace dealii;

template <int dim, int spacedim>
class DoFToCellTopology
{
public:
  /**
   * The core structure describing the DoF-to-cell topology. The index for the
   * outer vector is a full and external DoF index, i.e. the original DoF
   * indices defined in a DoF handler. Each element in the outer vector is
   * another vector of cell iterator pointers. The related cells share this
   * common DoF.
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